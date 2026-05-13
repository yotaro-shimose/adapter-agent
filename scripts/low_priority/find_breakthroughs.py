"""find_breakthroughs.py — surface tasks that flipped from 0-hit to ≥1-hit.

For one RL run, group rollouts by (task_id, rl_step) and compute the
per-step `any_success` flag (= at least one of the num_samples rollouts
succeeded at that step). A task is a "breakthrough" when:

  - it failed (any_success = False) on every step before some step S, AND
  - it first succeeded at step S where S > 1.

This is the strictest version of "RL taught it something new" — the model
literally never produced a successful sample for that task in the early
training, then learned to.

Tasks that succeeded from step 1 are filtered out (already knew how).
Tasks that never succeeded are filtered out (still can't do it).

Run with:
    uv run scripts/find_breakthroughs.py
    uv run scripts/find_breakthroughs.py --simple-train-id <id>
    uv run scripts/find_breakthroughs.py --detail-task <task_id>
"""

import argparse
import asyncio
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from prisma import Prisma


# Defaults — overridable from the command line.
DEFAULT_SIMPLE_TRAIN_ID = "continue_rl_task_hisab_from_qra_v2_20260507_015123"
DEFAULT_SUITE_NAME = "gh_archive_rl"


async def _load_per_step_any_success(
    prisma: Prisma, simple_train_id: str, suite_name: str
) -> tuple[dict[str, dict[int, bool]], dict[str, str]]:
    """Aggregate rows into `{task_id: {rl_step: any_success}}` and a
    `{task_id: instruction}` map. Done in SQL so we don't ship 22k rows
    over the wire.
    """
    rows = await prisma.query_raw(
        """
        SELECT
            task_id,
            rl_step,
            BOOL_OR(success) AS any_success,
            MAX(instruction) AS instruction
        FROM simple_rl_rollouts
        WHERE simple_train_id = $1 AND suite_name = $2
        GROUP BY task_id, rl_step
        ORDER BY task_id, rl_step
        """,
        simple_train_id,
        suite_name,
    )
    per_task_step: dict[str, dict[int, bool]] = defaultdict(dict)
    instructions: dict[str, str] = {}
    for r in rows:
        tid = r["task_id"]
        per_task_step[tid][int(r["rl_step"])] = bool(r["any_success"])
        if tid not in instructions:
            instructions[tid] = r["instruction"] or ""
    return per_task_step, instructions


def _classify(steps_to_success: dict[int, bool]) -> tuple[str, int | None]:
    """Return (category, first_success_step).

    category ∈ {
        "always_failed",     # never succeeded
        "always_succeeded",  # succeeded at the earliest step we have
        "breakthrough",      # 0 successes before some S>min_step, then succeeded
    }
    """
    if not steps_to_success:
        return "always_failed", None
    sorted_steps = sorted(steps_to_success)
    first_step = sorted_steps[0]
    success_steps = [s for s in sorted_steps if steps_to_success[s]]
    if not success_steps:
        return "always_failed", None
    first_success = success_steps[0]
    if first_success == first_step:
        return "always_succeeded", first_success
    return "breakthrough", first_success


def _post_breakthrough_rate(
    steps_to_success: dict[int, bool], first_success: int
) -> float:
    """Fraction of steps at-or-after the first success that were any_success.
    Useful to distinguish "found it once and lost it" from "actually learned"."""
    after = [v for s, v in steps_to_success.items() if s >= first_success]
    if not after:
        return 0.0
    return sum(1 for v in after if v) / len(after)


def _plot_breakthroughs_per_step(
    breakthroughs: list[tuple[str, int, float]],
    *,
    all_steps: list[int],
    simple_train_id: str,
    suite_name: str,
    out_path: Path,
) -> None:
    """Bar chart: # of tasks that flipped from 0-hit → ≥1-hit at each step.

    Plots every step in the run's range (not just steps that produced a
    breakthrough), so quiet stretches show up as visible zeros instead of
    being silently collapsed.
    """
    import seaborn as sns
    from matplotlib import pyplot as plt

    counts: dict[int, int] = {s: 0 for s in all_steps}
    for _, fs, _ in breakthroughs:
        counts[fs] = counts.get(fs, 0) + 1
    steps = sorted(counts)
    values = [counts[s] for s in steps]

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(max(8, len(steps) * 0.5), 5))
    sns.barplot(x=steps, y=values, ax=ax, color="#2ecc71")
    for x, v in enumerate(values):
        if v > 0:
            ax.text(x, v + 0.05, str(v), ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("RL step at which the task first succeeded")
    ax.set_ylabel("Tasks flipped 0→≥1 hit")
    ax.set_title(
        f"Breakthroughs per step\n"
        f"{simple_train_id}  ·  suite={suite_name}  ·  total={len(breakthroughs)}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"\nSaved plot: {out_path}")


def _plot_breakthroughs_cumulative(
    breakthroughs: list[tuple[str, int, float]],
    *,
    all_steps: list[int],
    simple_train_id: str,
    suite_name: str,
    out_path: Path,
) -> None:
    """Cumulative line chart: total # of tasks that have flipped 0→≥1 hit by
    each step (running sum of the per-step counts). Reads 'how many new
    things has RL taught the model so far'."""
    import seaborn as sns
    from matplotlib import pyplot as plt

    counts: dict[int, int] = {s: 0 for s in all_steps}
    for _, fs, _ in breakthroughs:
        counts[fs] = counts.get(fs, 0) + 1
    steps = sorted(counts)
    cumulative: list[int] = []
    running = 0
    for s in steps:
        running += counts[s]
        cumulative.append(running)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(max(8, len(steps) * 0.5), 5))
    sns.lineplot(
        x=steps, y=cumulative, ax=ax, marker="o", color="#2ecc71", linewidth=2
    )
    for x, v in zip(steps, cumulative):
        ax.text(x, v + 0.6, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("RL step")
    ax.set_ylabel("Cumulative tasks flipped 0→≥1 hit")
    ax.set_title(
        f"Cumulative breakthroughs\n"
        f"{simple_train_id}  ·  suite={suite_name}  ·  total={len(breakthroughs)}",
        fontsize=12,
    )
    ax.set_xticks(steps)
    ax.set_ylim(0, max(cumulative) * 1.12 if cumulative else 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


async def _print_summary(
    prisma: Prisma,
    simple_train_id: str,
    suite_name: str,
    plot_path: Path | None = None,
) -> None:
    per_task, instructions = await _load_per_step_any_success(
        prisma, simple_train_id, suite_name
    )
    if not per_task:
        print(
            f"No rollouts found for simple_train_id='{simple_train_id}', "
            f"suite_name='{suite_name}'."
        )
        return

    all_steps = sorted({s for d in per_task.values() for s in d})
    print("=" * 80)
    print(f"Run        : {simple_train_id}")
    print(f"Suite      : {suite_name}")
    print(f"Steps seen : {all_steps[0]}..{all_steps[-1]} ({len(all_steps)} distinct)")
    print(f"Tasks      : {len(per_task)}")
    print("=" * 80)

    breakthroughs: list[tuple[str, int, float]] = []
    always_succeeded = 0
    always_failed = 0
    for tid, step_map in per_task.items():
        cat, first_success = _classify(step_map)
        if cat == "always_succeeded":
            always_succeeded += 1
        elif cat == "always_failed":
            always_failed += 1
        else:
            assert first_success is not None
            rate_after = _post_breakthrough_rate(step_map, first_success)
            breakthroughs.append((tid, first_success, rate_after))

    print()
    print("Counts:")
    print(f"  always_succeeded : {always_succeeded:4d}")
    print(f"  always_failed    : {always_failed:4d}")
    print(f"  breakthrough     : {len(breakthroughs):4d}")
    print()

    if not breakthroughs:
        print("No breakthroughs found.")
        return

    # Histogram of first_success step for the breakthrough tasks.
    hist: dict[int, int] = defaultdict(int)
    for _, fs, _ in breakthroughs:
        hist[fs] += 1
    print("First-success step histogram (breakthroughs only):")
    for s in sorted(hist):
        bar = "█" * hist[s]
        print(f"  step {s:>3}: {hist[s]:>3} {bar}")
    print()

    # Sort: latest breakthroughs first (more impressive); break ties with
    # higher post-breakthrough success rate.
    breakthroughs.sort(key=lambda x: (-x[1], -x[2]))
    print(f"Breakthrough tasks ({len(breakthroughs)}):")
    print("-" * 80)
    for tid, fs, rate in breakthroughs:
        instr = instructions.get(tid, "").strip().replace("\n", " ")
        instr_preview = instr[:160] + ("..." if len(instr) > 160 else "")
        print(
            f"  task={tid}  first_success_step={fs:>3}  "
            f"post_rate={rate:.0%}"
        )
        print(f"    {instr_preview}")
        print()

    if plot_path is not None:
        _plot_breakthroughs_per_step(
            breakthroughs,
            all_steps=all_steps,
            simple_train_id=simple_train_id,
            suite_name=suite_name,
            out_path=plot_path,
        )
        # Cumulative companion plot. Sits next to the per-step bar chart with
        # a `_cumulative` suffix so both charts live in the same directory
        # without colliding even when the user passes an explicit --plot path.
        cum_path = plot_path.with_name(
            f"{plot_path.stem}_cumulative{plot_path.suffix}"
        )
        _plot_breakthroughs_cumulative(
            breakthroughs,
            all_steps=all_steps,
            simple_train_id=simple_train_id,
            suite_name=suite_name,
            out_path=cum_path,
        )


async def _print_task_detail(
    prisma: Prisma,
    simple_train_id: str,
    suite_name: str,
    task_id: str,
) -> None:
    """Per-step trace for one task: any_success, success_count/num_samples,
    plus the first successful answer body so you can eyeball what it
    learned."""
    rows = await prisma.query_raw(
        """
        SELECT
            rl_step,
            COUNT(*)::int AS rollouts,
            SUM(CASE WHEN success THEN 1 ELSE 0 END)::int AS successes,
            BOOL_OR(success) AS any_success,
            MAX(instruction) AS instruction
        FROM simple_rl_rollouts
        WHERE simple_train_id = $1 AND suite_name = $2 AND task_id = $3
        GROUP BY rl_step
        ORDER BY rl_step ASC
        """,
        simple_train_id,
        suite_name,
        task_id,
    )
    if not rows:
        print(f"No rollouts for task_id='{task_id}'.")
        return

    instr = rows[0].get("instruction") or ""
    print("=" * 80)
    print(f"Task       : {task_id}")
    print(f"Run        : {simple_train_id}")
    print(f"Suite      : {suite_name}")
    print()
    print("Instruction:")
    print(instr.strip())
    print("=" * 80)
    for r in rows:
        marker = "✓" if r["any_success"] else "✗"
        print(
            f"  step {int(r['rl_step']):>3} {marker}  "
            f"{int(r['successes'])}/{int(r['rollouts'])} samples succeeded"
        )

    # Pull the earliest successful rollout's answer body.
    first_hit_rows = await prisma.query_raw(
        """
        SELECT rl_step, sample_idx, answer
        FROM simple_rl_rollouts
        WHERE simple_train_id = $1 AND suite_name = $2 AND task_id = $3
              AND success = TRUE
        ORDER BY rl_step ASC, sample_idx ASC
        LIMIT 1
        """,
        simple_train_id,
        suite_name,
        task_id,
    )
    if first_hit_rows:
        h = first_hit_rows[0]
        print()
        print("-" * 80)
        print(
            f"First successful answer (step {int(h['rl_step'])}, sample {int(h['sample_idx'])}):"
        )
        print("-" * 80)
        print(h.get("answer", ""))


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simple-train-id",
        default=DEFAULT_SIMPLE_TRAIN_ID,
        help="run id (default: %(default)s)",
    )
    parser.add_argument(
        "--suite-name",
        default=DEFAULT_SUITE_NAME,
        help="suite name (default: %(default)s)",
    )
    parser.add_argument(
        "--detail-task",
        default=None,
        help="If set, print per-step detail for this task_id instead of the summary.",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const="",  # bare --plot picks the default path
        default=None,
        help=(
            "Save a breakthroughs-per-step bar chart. With no value, writes "
            "to <repo>/output/breakthroughs_<simple_train_id>.png."
        ),
    )
    args = parser.parse_args()

    load_dotenv()
    prisma = Prisma()
    await prisma.connect()
    try:
        if args.detail_task:
            await _print_task_detail(
                prisma, args.simple_train_id, args.suite_name, args.detail_task
            )
        else:
            plot_path: Path | None = None
            if args.plot is not None:
                if args.plot:
                    plot_path = Path(args.plot)
                else:
                    # Default landing zone: <repo>/output/ (sibling of scripts/).
                    out_dir = Path(__file__).resolve().parent.parent / "output"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    plot_path = (
                        out_dir / f"breakthroughs_{args.simple_train_id}.png"
                    )
            await _print_summary(
                prisma,
                args.simple_train_id,
                args.suite_name,
                plot_path=plot_path,
            )
    finally:
        await prisma.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
