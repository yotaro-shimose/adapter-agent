import asyncio
from pathlib import Path

import tinker
from oai_utils.tinker import setup_tinkermodel

from adapter_agent.hierarchical.process.rewire import ss_solve_verify
from adapter_agent.hierarchical.agent.rewirer import log_trajectory
from adapter_agent.rl.env.session_result import (
    RewireSessionResultNormal,
    RewireSessionResultSuccess,
)
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.knowledge_db import KnowledgeDB
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.model_helper import get_gemini
from adapter_agent.rl.env.runtime_settings import RuntimeSettings


async def main():
    service_client = tinker.ServiceClient()
    model, _tokenizer, _renderer = setup_tinkermodel(
        service_client=service_client,
        model_name="Qwen/Qwen3-8B",
    )

    # 1. Initialize Tools and Verifier
    rust_doc_analyzer = await AsyncRustDocAnalyzer.create_from_libdir(Path("repositories/numrs"))
    verifier_model = get_gemini()
    
    # 2. Initialize KnowledgeDB and explicitly clear it for this experiment
    knowledge_db = KnowledgeDB()
    await knowledge_db.initialize()
    await knowledge_db.clear()
    await knowledge_db.initialize() # re-create index after clearing

    # 3. Create simple task: 5x5 array
    instruction = "Create a 5x5 array of zeros using the `numrs2` library and print it."
    task = Task.from_instruction(instruction)

    runtime_settings = RuntimeSettings(
        type="docker",
        image_uri="coder-mcp-numrs2:latest",
    )

    # 4. First Solve (Empty Knowledge) - loop until success
    success = False
    run_idx = 1
    while not success:
        print("=" * 80)
        print(f"🤖 [INITIAL RUN {run_idx}] Solving with EMPTY knowledge base (Expect: rust_doc_search fallback)")
        print("=" * 80)
        
        ret1 = await ss_solve_verify(
            solver_model=model,
            verifier_model=verifier_model,
            rust_doc_analyzer=rust_doc_analyzer,
            task=task,
            max_turns=10,
            qwen_no_think=True,
            runtime_settings=runtime_settings,
            knowledge_db=knowledge_db,
        )

        if isinstance(ret1, RewireSessionResultNormal):
            print(f"✅ Initial Run {run_idx} Conclusion: {ret1.conclusion}")
            print("\\n-- Trajectory Log --")
            log_trajectory(ret1.trials, flip_tag=True)
            # 軌跡ログの保存
            with open(f"trajectory_initial_{run_idx}.txt", "w") as f:
                for trial in ret1.trials:
                    f.write(str(trial) + "\\n")
            
            if ret1.conclusion == "success":
                success = True
                print("🎉 Initial run succeeded! Knowledge has been acquired.")
                if isinstance(ret1, RewireSessionResultSuccess):
                    from rich.console import Console
                    from rich.panel import Panel
                    from rich.markdown import Markdown
                    
                    console = Console()
                    console.print()
                    console.print(
                        Panel(
                            Markdown(ret1.knowledge or "No knowledge extracted."),
                            title="🧠 Acquired Knowledge",
                            border_style="magenta",
                            expand=False
                        )
                    )

                    if ret1.oc_trials:
                        print("\n-- OC-Converted (Closed Book) Trajectory Log --")
                        log_trajectory(ret1.oc_trials, flip_tag=True)
                        with open(f"trajectory_oc_initial_{run_idx}.txt", "w") as f:
                            for trial in ret1.oc_trials:
                                f.write(str(trial) + "\n")
                        print(f"📄 OC trajectory saved to trajectory_oc_initial_{run_idx}.txt")
            else:
                print(f"🔄 Initial run failed (Conclusion: {ret1.conclusion}).")
                if ret1.reasoning:
                    print(f"🧐 Reasoning: {ret1.reasoning}")
                print("Retrying...")
                run_idx += 1
                await knowledge_db.clear()
                await knowledge_db.initialize()
        else:
            print(f"❌ Initial Run {run_idx} Error Log: {ret1}")
            print("🔄 Retrying...")
            run_idx += 1
            await knowledge_db.clear()
            await knowledge_db.initialize()

    # 5. Subsequent Solves
    for i in range(1, 3):
        print("\\n\\n" + "=" * 80)
        print(f"🤖 [FOLLOWUP RUN {i}] Solving again with ACQUIRED knowledge base (Expect: search_knowledge_db hit)")
        print("=" * 80)
        
        ret2 = await ss_solve_verify(
            solver_model=model,
            verifier_model=verifier_model,
            rust_doc_analyzer=rust_doc_analyzer,
            task=task,
            max_turns=10,
            qwen_no_think=True,
            runtime_settings=runtime_settings,
            knowledge_db=knowledge_db,
        )

        if isinstance(ret2, RewireSessionResultNormal):
            print(f"✅ Followup Run {i} Conclusion: {ret2.conclusion}")
            if ret2.reasoning:
                print(f"🧐 Reasoning: {ret2.reasoning}")
            print("\\n-- Trajectory Log --")
            log_trajectory(ret2.trials, flip_tag=True)
            # 軌跡ログの保存
            with open(f"trajectory_followup_{i}.txt", "w") as f:
                for trial in ret2.trials:
                    f.write(str(trial) + "\n")
            
            if isinstance(ret2, RewireSessionResultSuccess) and ret2.oc_trials:
                print("\n-- OC-Converted (Closed Book) Trajectory Log --")
                log_trajectory(ret2.oc_trials, flip_tag=True)
                with open(f"trajectory_oc_followup_{i}.txt", "w") as f:
                    for trial in ret2.oc_trials:
                        f.write(str(trial) + "\n")
                print(f"📄 OC trajectory saved to trajectory_oc_followup_{i}.txt")
        else:
            print(f"❌ Followup Run {i} Error Log: {ret2}")


if __name__ == "__main__":
    asyncio.run(main())
