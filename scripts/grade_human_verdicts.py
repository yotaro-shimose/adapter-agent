"""Aggregate human verdicts against gemini_lite verdicts in the rollout
review markdown produced by `scripts/dump_canonical_rollouts.py`.

Parses each `## Rollout` section for the lines:

    **gemini_lite verdict:** SUCCESS|FAILURE
    **Human verdict:** OK|NG|(... anything else means ungraded)

and reports the 2×2 confusion matrix + agreement rate. Pass one or more
review markdown paths as args; defaults to the canonical hisab + numrs2
dumps under /tmp.

The relevant interpretation for reward-hacking analysis:
  - gemini SUCCESS / human NG = false positive (gemini accepted a wrong
    answer — the "hackable" cases that inflated the eval score)
  - gemini FAILURE / human OK = false negative (gemini rejected a correct
    answer — undercount, less damaging)
  - gemini SUCCESS / human OK and FAILURE / NG = agreements

Usage:
    uv run scripts/grade_human_verdicts.py
    uv run scripts/grade_human_verdicts.py /tmp/canonical_hisab_rollouts_review.md
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATHS = [
    Path("human_review/canonical_hisab"),
    Path("human_review/canonical_numrs2"),
]

# Per-file layout: each rollout markdown starts with `# Rollout NN — task_idx=X`.
# Legacy single-file layout (kept for backward compat) used `## Rollout NN`.
_ROLLOUT_RE = re.compile(r"^#{1,2} Rollout (\d+) — task_idx=(\d+)$")
_GEMINI_RE = re.compile(r"^\*\*gemini_lite verdict:\*\*\s*(\S+)")
_HUMAN_RE = re.compile(r"^\*\*Human verdict:\*\*\s*(.+?)\s*$")


@dataclass
class Row:
    rollout_idx: int
    task_idx: int
    gemini: str  # "SUCCESS" | "FAILURE"
    human_raw: str
    human: str | None  # "OK" | "NG" | None (ungraded)


def _normalize_human(raw: str) -> str | None:
    s = raw.strip().upper()
    # Anything matching the placeholder remains ungraded.
    if s.startswith("(") or s == "" or "FILL IN" in s:
        return None
    if s in ("OK", "○", "MARU", "MARU", "Y", "YES", "TRUE", "T"):
        return "OK"
    if s in ("NG", "X", "×", "BATSU", "N", "NO", "FALSE", "F"):
        return "NG"
    return None  # unrecognized → treat as ungraded


def parse_review(path: Path) -> list[Row]:
    rows: list[Row] = []
    cur_rollout: int | None = None
    cur_task: int | None = None
    cur_gemini: str | None = None
    cur_human_raw: str | None = None

    def _flush() -> None:
        nonlocal cur_rollout, cur_task, cur_gemini, cur_human_raw
        if cur_rollout is not None and cur_gemini is not None:
            human = _normalize_human(cur_human_raw or "")
            rows.append(
                Row(
                    rollout_idx=cur_rollout,
                    task_idx=cur_task if cur_task is not None else -1,
                    gemini=cur_gemini.upper(),
                    human_raw=(cur_human_raw or "").strip(),
                    human=human,
                )
            )
        cur_rollout = None
        cur_task = None
        cur_gemini = None
        cur_human_raw = None

    for line in path.read_text(encoding="utf-8").splitlines():
        m = _ROLLOUT_RE.match(line)
        if m:
            _flush()
            cur_rollout = int(m.group(1))
            cur_task = int(m.group(2))
            continue
        m = _GEMINI_RE.match(line)
        if m:
            cur_gemini = m.group(1)
            continue
        m = _HUMAN_RE.match(line)
        if m:
            cur_human_raw = m.group(1)
            continue
    _flush()
    return rows


def report(path: Path, rows: list[Row]) -> None:
    print(f"\n=== {path.name} ===")
    print(f"Total rollouts in file: {len(rows)}")

    graded = [r for r in rows if r.human is not None]
    ungraded = [r for r in rows if r.human is None]
    print(f"  graded: {len(graded)}   ungraded: {len(ungraded)}")

    if ungraded:
        ids = ", ".join(f"#{r.rollout_idx:02d}" for r in ungraded)
        print(f"  ungraded ids: {ids}")

    if not graded:
        print("  (no human verdicts present — nothing to aggregate)")
        return

    # Confusion matrix: rows = gemini, cols = human.
    tp = sum(1 for r in graded if r.gemini == "SUCCESS" and r.human == "OK")
    fp = sum(1 for r in graded if r.gemini == "SUCCESS" and r.human == "NG")
    fn = sum(1 for r in graded if r.gemini == "FAILURE" and r.human == "OK")
    tn = sum(1 for r in graded if r.gemini == "FAILURE" and r.human == "NG")

    print()
    print("  Confusion matrix (rows=gemini_lite, cols=human):")
    print(f"                 human OK   human NG")
    print(f"    gemini SUCCESS  {tp:>4}       {fp:>4}")
    print(f"    gemini FAILURE  {fn:>4}       {tn:>4}")

    agree = tp + tn
    total = tp + fp + fn + tn
    if total:
        print(f"\n  Agreement: {agree}/{total} = {agree / total:.0%}")
    if (tp + fp):
        prec = tp / (tp + fp)
        print(f"  gemini precision (P(human OK | gemini SUCCESS)): "
              f"{tp}/{tp + fp} = {prec:.0%}")
        if fp:
            print(f"    → gemini false-positive rate (reward-hacking proxy): "
                  f"{fp}/{tp + fp} = {fp / (tp + fp):.0%}")
    if (tp + fn):
        rec = tp / (tp + fn)
        print(f"  gemini recall    (P(gemini SUCCESS | human OK)): "
              f"{tp}/{tp + fn} = {rec:.0%}")

    # If everything is graded, also estimate the population stats.
    if not ungraded and graded:
        n = len(graded)
        gs = tp + fp
        gf = fn + tn
        print()
        print(
            f"  Sample composition (random 20): {gs} gemini SUCCESS + "
            f"{gf} gemini FAILURE."
        )
        print(f"  gemini-effective accuracy (TP+TN over {n}): "
              f"{(tp + tn)}/{n} = {(tp + tn) / n:.0%}")


def _collect_rows(path: Path) -> list[Row]:
    """Accept either a directory of `rollout_*.md` files or a single markdown
    file (legacy, multiple `## Rollout` sections in one file)."""
    if path.is_dir():
        rows: list[Row] = []
        for md in sorted(path.glob("rollout_*.md")):
            rows.extend(parse_review(md))
        return rows
    return parse_review(path)


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv[1:]] if len(argv) > 1 else DEFAULT_PATHS
    for path in paths:
        if not path.exists():
            print(f"[skip] not found: {path}")
            continue
        rows = _collect_rows(path)
        report(path, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
