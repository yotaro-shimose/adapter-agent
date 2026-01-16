import json
import random
import sys
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel


def display_random_qra(qra_dir: Path):
    json_files = list(qra_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {qra_dir}")
        return

    random_file = random.choice(json_files)
    with open(random_file, "r") as f:
        data = json.load(f)

    console = Console()

    console.print(f"\n[bold blue]Showing File:[/bold blue] {random_file.name}")
    console.print(f"[bold blue]Slug:[/bold blue] {data.get('slug', 'N/A')}")
    console.print(
        f"[bold blue]Concept:[/bold blue] [italic]{data.get('concept', 'N/A')}[/italic]"
    )
    console.print("-" * 50)

    # Question
    console.print(
        Panel(
            Markdown(data.get("question", "")),
            title="[bold yellow]Question[/bold yellow]",
            border_style="yellow",
        )
    )

    # Reasoning
    console.print(
        Panel(
            data.get("reasoning", ""),
            title="[bold cyan]Reasoning[/bold cyan]",
            border_style="cyan",
        )
    )

    # Answer
    console.print(
        Panel(
            Markdown(data.get("answer", "")),
            title="[bold green]Answer[/bold green]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    qra_dir = Path("data/qra/experiment_generic_aligned")
    if len(sys.argv) > 1:
        qra_dir = Path(sys.argv[1])

    display_random_qra(qra_dir)
