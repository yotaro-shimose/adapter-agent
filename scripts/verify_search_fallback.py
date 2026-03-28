import asyncio
from pathlib import Path

import tinker
from oai_utils.tinker import setup_tinkermodel

from adapter_agent.hierarchical.process.rewire import ss_solve_verify
from adapter_agent.hierarchical.agent.rewirer import log_trajectory
from adapter_agent.rl.env.session_result import (
    RewireSessionResultNormal,
)
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.library.knowledge_db import KnowledgeDB
from adapter_agent.model_helper import get_gemini
from adapter_agent.rl.env.runtime_settings import RuntimeSettings


async def main():
    service_client = tinker.ServiceClient()
    # Use a cheap and fast model for testing fallback if possible,
    # but Qwen is what was used in experiment_oc.py.
    model, _tokenizer, _renderer = setup_tinkermodel(
        service_client=service_client,
        model_name="Qwen/Qwen3-8B",
    )

    # 1. Initialize Tools and Verifier
    # Ensure numrs (or numrs2) exists. The original script uses numrs2.
    rust_doc_analyzer = await AsyncRustDocAnalyzer.create_from_libdir(
        Path("repositories/numrs")
    )
    verifier_model = get_gemini()

    # 2. Initialize KnowledgeDB and seed it with INAPPROPRIATE knowledge
    knowledge_db = KnowledgeDB()
    await knowledge_db.initialize()
    await knowledge_db.clear()
    await knowledge_db.initialize()

    print("🧠 Seeding KnowledgeDB with INAPPROPRIATE knowledge...")
    junk_query = "zeros"
    junk_content = "To create zeros in standard Rust, you can use `vec![0; 25]`."
    await knowledge_db.add_knowledge(junk_query, junk_content)

    # 3. Create task: 5x5 array using numrs2
    instruction = "Create a 5x5 array of zeros using the `numrs2` library and print it."
    task = Task.from_instruction(instruction)

    runtime_settings = RuntimeSettings(
        type="docker",
        image_uri="coder-mcp-numrs2:latest",
    )

    # 4. Run Solve (Knowledge exists but is junk)
    print("=" * 80)
    print("🤖 [FALLBACK TEST] Solving with INAPPROPRIATE knowledge in DB")
    print(
        "Expected behavior: SearchTool finds 'zeros' in DB, realizes it's irrelevant, calls `no_relevant_knowledge`, and falls back to RustDocs."
    )
    print("=" * 80)

    ret = await ss_solve_verify(
        solver_model=model,
        verifier_model=verifier_model,
        rust_doc_analyzer=rust_doc_analyzer,
        task=task,
        max_turns=10,
        qwen_no_think=True,
        runtime_settings=runtime_settings,
        knowledge_db=knowledge_db,
    )

    if isinstance(ret, RewireSessionResultNormal):
        print(f"✅ Conclusion: {ret.conclusion}")
        print("\n-- Trajectory Log --")
        log_trajectory(ret.trials, flip_tag=True)

        from rich.console import Console
        from rich.panel import Panel
        from rich.markdown import Markdown
        console = Console()
        console.print()
        console.print(
            Panel(
                Markdown(ret.knowledge or "No knowledge extracted."),
                title="🧠 Acquired Knowledge",
                border_style="magenta",
                expand=False
            )
        )
        
        # Check if it succeeded
        if ret.conclusion == "success":
            print("\n🎉 SUCCESS! The agent was able to solve the task despite junk knowledge in DB.")
            print("This confirms that the fallback to RustDoc search occurred.")
        else:
            print(f"\n❌ FAILED: Conclusion was {ret.conclusion}")
            if ret.reasoning:
                print(f"🧐 Reasoning: {ret.reasoning}")
    else:
        print(f"❌ Error Log: {ret}")


if __name__ == "__main__":
    asyncio.run(main())
