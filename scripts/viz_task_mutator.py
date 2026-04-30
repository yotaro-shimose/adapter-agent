"""Visualize a Task → Task-variants mutator on gh_archive seed tasks.

Probe script for the "task augmentation" idea discussed: take a real
gh_archive task and have an LLM produce several similar-but-diverse
variants. Prints original + variants side-by-side so we can eyeball the
quality / diversity trade-off before investing in a real implementation.

Usage:
    uv run python scripts/viz_task_mutator.py
    uv run python scripts/viz_task_mutator.py --seeds 5 --variants 4
    uv run python scripts/viz_task_mutator.py --slice-start 10 --seeds 3
"""

import argparse
import asyncio
import logging
from dataclasses import dataclass

from dotenv import load_dotenv
from oai_utils.agent import AgentsSDKModel, AgentWrapper
from pydantic import BaseModel, Field
from rich.console import Console, Group
from rich.logging import RichHandler
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from adapter_agent.hierarchical.agent.base import BaseAgent
from adapter_agent.hierarchical.gh import load_gh_archive
from adapter_agent.hierarchical.types import Task
from adapter_agent.model_helper import get_gemini

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger(__name__)


_MUTATOR_PROMPT = """\
You are a task augmenter for an RL training pipeline. Given one programming task,
you produce several *similar but diverse* variants that exercise the same
underlying skills while differing meaningfully in surface details.

### Goals
- Preserve the core skill / API surface the original task exercises (e.g. if it
  uses `numrs2.array`, variants should still revolve around `numrs2.array`).
- Vary at least ONE of: input data shape, dtype, numerical scale, edge case
  handled, framing/domain ("audio buffer" vs "image tensor" vs "stock prices"),
  or the secondary operation chained after the primary one.
- Each variant must be self-contained and solvable on its own — do not refer
  back to the original task ("same as above but...").
- Avoid pure paraphrase. If a reader who solved the original would solve the
  variant by literally copying their solution, the variant is too close.
- Avoid drift into unrelated APIs or skills. If the variant no longer needs the
  original library/feature, it has drifted too far.

### Output
A list of variants. Each entry has:
- `instruction`: the new task statement, written in the same style as the original.
- `variation_axis`: a short tag describing what you changed (e.g. "dtype: f32→i64",
  "domain: audio→image", "edge case: empty input").
"""


class TaskVariant(BaseModel):
    instruction: str = Field(description="The mutated task instruction, self-contained.")
    variation_axis: str = Field(description="Short tag describing what was changed.")


class TaskVariantList(BaseModel):
    variants: list[TaskVariant]


@dataclass(kw_only=True)
class TaskMutator[T: AgentsSDKModel](BaseAgent[T]):
    """Produce N similar-but-diverse variants of a single Task."""

    async def mutate(self, task: Task, n_variants: int) -> list[TaskVariant]:
        agent = AgentWrapper[TaskVariantList].create(
            name="TaskMutator",
            instructions=_MUTATOR_PROMPT,
            model=self.model,
            output_type=TaskVariantList,
        )
        input_prompt = f"""\
Original task:
<Task>
{task.instruction}
</Task>

Produce exactly {n_variants} variants per the rules above.
"""
        try:
            result = await agent.run(input_prompt, time_out_seconds=120.0)
            output = result.final_output()
            return output.variants
        except Exception as e:
            logger.exception(f"Mutation failed for task {task.id}: {e}")
            return []


def _render_seed(idx: int, task: Task, variants: list[TaskVariant]) -> None:
    """Render one seed + its variants as stacked panels under a section rule."""
    console.print()
    console.print(
        Rule(
            Text.assemble(
                ("SEED #", "bold white"),
                (f"{idx}", "bold cyan"),
                ("  id=", "dim"),
                (task.id, "dim"),
            ),
            style="cyan",
        )
    )

    original_panel = Panel(
        Text(task.instruction.strip(), style="white"),
        title=Text("ORIGINAL", style="bold cyan"),
        border_style="cyan",
        padding=(1, 2),
    )

    if not variants:
        empty = Panel(
            Text("(mutator returned no variants — check logs)", style="red"),
            border_style="red",
            padding=(1, 2),
        )
        console.print(Group(original_panel, empty))
        return

    variant_panels = []
    for j, v in enumerate(variants):
        body = Text.assemble(
            ("axis: ", "dim"),
            (v.variation_axis, "italic yellow"),
            ("\n\n", ""),
            (v.instruction.strip(), "white"),
        )
        variant_panels.append(
            Panel(
                body,
                title=Text(f"VARIANT #{idx}.{j}", style="bold green"),
                border_style="green",
                padding=(1, 2),
            )
        )

    console.print(Group(original_panel, *variant_panels))


async def _mutate_one(
    mutator: TaskMutator, task: Task, n_variants: int, idx: int
) -> tuple[int, Task, list[TaskVariant]]:
    variants = await mutator.mutate(task, n_variants)
    return idx, task, variants


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3, help="Number of seed tasks to mutate.")
    parser.add_argument("--variants", type=int, default=3, help="Variants per seed task.")
    parser.add_argument("--slice-start", type=int, default=0, help="Start index into load_gh_archive().")
    parser.add_argument("--concurrency", type=int, default=8, help="Parallel mutation calls.")
    args = parser.parse_args()

    load_dotenv()

    archive = load_gh_archive()
    seeds = archive[args.slice_start : args.slice_start + args.seeds]
    logger.info(
        f"Loaded {len(archive)} gh_archive tasks; mutating "
        f"[{args.slice_start}:{args.slice_start + args.seeds}] "
        f"({len(seeds)} seeds × {args.variants} variants)."
    )

    mutator = TaskMutator(model=get_gemini())

    sem = asyncio.Semaphore(args.concurrency)

    async def _bounded(idx: int, task: Task) -> tuple[int, Task, list[TaskVariant]]:
        async with sem:
            return await _mutate_one(mutator, task, args.variants, idx)

    results = await asyncio.gather(
        *[_bounded(i, t) for i, t in enumerate(seeds)]
    )

    total_variants = 0
    for idx, task, variants in sorted(results, key=lambda r: r[0]):
        _render_seed(idx, task, variants)
        total_variants += len(variants)

    console.print()
    console.print(
        Rule(
            Text.assemble(
                ("Done. ", "bold"),
                (f"{len(seeds)}", "cyan"),
                (" seeds → ", "bold"),
                (f"{total_variants}", "green"),
                (" variants (target ", "bold"),
                (f"{len(seeds) * args.variants}", "yellow"),
                (")", "bold"),
            ),
            style="bold",
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
