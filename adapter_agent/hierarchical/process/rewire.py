import logging
import re
from typing import cast

from agents.extensions.models.litellm_model import LitellmModel
from oai_utils.tinker import TinkerModel
from tinker_cookbook.renderers.base import Message as TinkerMessage
from tinker_cookbook.renderers.base import TextPart, ThinkingPart

from adapter_agent.data import QA
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.process.rewire_session import (
    RewireSessionResult,
    log_trajectory_if_debug,
    metrics_with_prefix,
)
from adapter_agent.hierarchical.types import Task
from adapter_agent.rl.completer import TinkerMessageCompleter
from adapter_agent.rl.env.reward import LLMAsAJudge
from adapter_agent.rl.env.simplified_solver import (
    SimplifiedSolverEnv,
    SimplifiedSolverEnvState,
    build_simplified_solver_env,
)
from adapter_agent.rl.litellm_completer import LitellmMessageCompleter
from adapter_agent.util.exception import MaximumContextExceeded

logger = logging.getLogger(__name__)


async def _rewire_solution(
    solver_model: TinkerModel,
    verifier: Verifier,
    rewirer_model: LitellmModel | TinkerModel,
    task: Task,
    msg_env: SimplifiedSolverEnv,
    i: int,
    init_metrics: dict,
    last_assistant_msg: TinkerMessage,
    trials: list[list[TinkerMessage]],
) -> RewireSessionResult | None:

    if isinstance(last_assistant_msg["content"], str):
        final_text = last_assistant_msg["content"]
    else:
        final_text = "".join(
            part["text"]
            for part in last_assistant_msg["content"]
            if part["type"] == "text"
        )

    code_match_1 = re.search(r"```rust(.*?)```", final_text, flags=re.DOTALL)
    if not code_match_1:
        logger.debug("Rewiring solver successful but failed to extract rust code.")
        return None
    original_code = code_match_1.group(1).strip()

    PROMPT2 = """\
<Role>
You are an expert Rust engineer generating a training dataset for other AI agents. You are tasked with writing a first-person reasoning and explanation for solving a user's instruction.
</Role>

<Context>
You are given a user's instruction and the known correct answer which contains a correct Rust code block achieving the user's request.
</Context>

<HowTo>
You must output your response using `<reasoning>` and `<answer>` XML tags to simulate how a knowledgeable human engineer would solve the problem without using any external tools or documentation.

- `<reasoning>`: The reasoning process which naturally leads the answer. This must start from "Okay, let's see." followed by briefly explaining the user's request. Then you start recalling the necessary knowledge to solve the problem. The reasoning should be described in the first person as if you remembered the knowledge by yourself without any external search. Do NOT mention any search tools, documentation searches, code execution result or trial-and-error process.
- `<answer>`: The final answer to the question (including both explanation and the code block). The Rust code enclosed in ```rust ... ``` must be EXACTLY the same as the one provided in the known answer. Do not change a single character of the provided source code.
</HowTo>

<OutputFormat>
You must output exactly in this format:
<reasoning>
Okay, let's see. [Your reasoning here...]
</reasoning>
<answer>
[Your explanation here]
```rust
// <the exact code from the provided answer>
```
</answer>
</OutputFormat>\
"""
    if isinstance(rewirer_model, LitellmModel):
        completer2 = LitellmMessageCompleter(
            model=rewirer_model.model,
            renderer=solver_model.renderer,
            tools=[],
        )
    else:
        completer2 = TinkerMessageCompleter(
            rewirer_model.sampling_client,
            renderer=rewirer_model.renderer,
            max_tokens=None,  # type: ignore
        )

    internalizer_messages = [
        TinkerMessage(role="system", content=PROMPT2),
        TinkerMessage(
            role="user",
            content=f"Instruction:\n{task.instruction}\n\nKnown Answer:\n{final_text}",
        ),
    ]

    ac_message2 = await completer2(internalizer_messages)

    if isinstance(ac_message2["content"], str):
        final_text2 = ac_message2["content"]
    else:
        final_text2 = "".join(
            part["text"] for part in ac_message2["content"] if part["type"] == "text"
        )

    reasoning_match = re.search(
        r"<reasoning>(.*?)</reasoning>", final_text2, flags=re.DOTALL
    )
    answer_match = re.search(r"<answer>(.*?)</answer>", final_text2, flags=re.DOTALL)
    if not answer_match:
        answer_match = re.search(r"<answer>(.*)", final_text2, flags=re.DOTALL)

    if not reasoning_match or not answer_match:
        logger.debug(
            "Internalizer failed to parse <reasoning> or <answer> from final_text."
        )
        return None

    reasoning_str = reasoning_match.group(1).strip()
    answer_str = answer_match.group(1).strip()

    code_match_2 = re.search(r"```rust(.*?)```", answer_str, flags=re.DOTALL)
    if not code_match_2 or code_match_2.group(1).strip() != original_code:
        logger.debug("Internalizer produced different rust code.")
        return None

    execution_output, _ = await msg_env.rust_env.run_cargo()
    main_rs_content = await msg_env.rust_env.view_file("src/main.rs")

    qa = QA(question=task.instruction, answer=answer_str)

    verification_result = await verifier.verify(
        qa=qa,
        tree_structure=msg_env.reward_fn.tree_structure,
        execution_output=execution_output,
        main_rs_content=main_rs_content,
    )

    if verification_result.success:
        whole_traj = list(msg_env.initial_messages)

        new_assistant_content: list = [
            ThinkingPart(type="thinking", thinking=reasoning_str),
            TextPart(type="text", text=answer_str),
        ]

        whole_traj.append(
            TinkerMessage(
                role="assistant",
                content=new_assistant_content,
            )
        )

        log_trajectory_if_debug(whole_traj)
        metrics = dict(
            no_text_content=0.0,
            no_code_found=0.0,
            code_did_not_compile=0.0,
            verifier_failed=0.0,
            verifier_error=0.0,
        )
        logger.info(f"Successfully solved and rewired after {i + 1} rewirings")
        return RewireSessionResult.success(
            task=task,
            messages=whole_traj,
            num_rewire=i + 1,
            metrics=init_metrics | metrics_with_prefix(metrics, f"rewire{i + 1}"),
            trials=trials,
        )
    else:
        logger.debug(
            f"Verification of rewired answer failed: {verification_result.reasoning}"
        )
        return None


async def run_rewiring_loop(
    solver_model: TinkerModel,
    verifier: Verifier,
    rewirer_model: LitellmModel | TinkerModel,
    task: Task,
    max_rewire: int,
    max_turns: int,
    init_metrics: dict | None = None,
    init_trials: list[list[TinkerMessage]] | None = None,
    qwen_no_think: bool = False,
) -> RewireSessionResult:
    if init_metrics is None:
        init_metrics = {}
    if init_trials is not None:
        trials = init_trials
    else:
        trials: list[list[TinkerMessage]] = []
    for i in range(max_rewire):
        ss_state = SimplifiedSolverEnvState.numrs2(
            task=task, qwen_no_think=qwen_no_think
        )
        async with build_simplified_solver_env(
            env_state=ss_state,
            renderer=solver_model.renderer,
            verifier=verifier,
            rust_doc_analyzer=verifier.rust_doc_analyzer,
            max_turns=max_turns,
        ) as env:
            msg_env = cast(SimplifiedSolverEnv, env.message_env)
            if isinstance(rewirer_model, LitellmModel):
                completer = LitellmMessageCompleter(
                    model=rewirer_model.model,
                    renderer=solver_model.renderer,
                    tools=list(msg_env.tools.values()),
                )
            else:
                completer = TinkerMessageCompleter(
                    rewirer_model.sampling_client,
                    renderer=rewirer_model.renderer,
                    max_tokens=None,  # type: ignore
                )

            ob = await msg_env.initial_observation()
            while True:
                try:
                    ac_message = await completer(ob)
                except MaximumContextExceeded:
                    logger.debug(
                        "Maximum context length exceeded during the rewiring loop"
                    )
                    step_result = None
                    ob[-1] = TinkerMessage(
                        role="tool",
                        content="The result of your action resulted in context length overflow.",
                    )
                    break

                step_result = await msg_env.step(ac_message)
                ob = step_result.next_messages
                if step_result.episode_done:
                    break
            trials.append(ob)

            if step_result is not None and LLMAsAJudge.is_successful_reward(
                step_result.reward
            ):
                # Retrieve the final valid Rust code from the environment history
                # Assuming the last assistant message before reward contains the code
                last_assistant_msg = next(
                    (m for m in reversed(ob) if m["role"] == "assistant")
                )
                result = await _rewire_solution(
                    solver_model=solver_model,
                    verifier=verifier,
                    rewirer_model=rewirer_model,
                    task=task,
                    msg_env=msg_env,
                    i=i,
                    init_metrics=init_metrics,
                    last_assistant_msg=last_assistant_msg,
                    trials=trials,
                )
                if result is not None:
                    return result
                continue

            else:
                log_trajectory_if_debug(ob)
                logger.info(f"Failed to solve after {i + 1} rewirings")
    logger.info("Failed to solve after max_rewire")
    return RewireSessionResult.error(
        task=task,
        error_message="Failed to solve after max_rewire",
        trials=trials,
    )
