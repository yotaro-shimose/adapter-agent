import logging
import re
from typing import cast

from coder_mcp.runtime import CoderMCPRuntimeError
from oai_utils.tinker import TinkerModel
from tinker_cookbook.renderers.base import Message as TinkerMessage
from tinker_cookbook.rl.types import Trajectory, Transition

from adapter_agent.data import QA, QRA
from adapter_agent.hierarchical.agent.knowledge_normalizer import (
    AgentsSDKModel,
    KnowledgeNormalizer,
    verify_normalized_knowledge,
)
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.process.rewire_session import (
    RewireSessionResult,
    RewireSessionResultError,
    RewireSessionResultFailure,
    RewireSessionResultSuccess,
)
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer
from adapter_agent.rl.completer import TinkerMessageCompleter
from adapter_agent.rl.env.reward import LLMAsAJudge
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.rl.env.simplified_solver import (
    SimplifiedSolverEnv,
    SimplifiedSolverEnvState,
    build_simplified_solver_env,
)
from adapter_agent.util.exception import CodingEnvironmentError, MaximumContextExceeded

CONTEXT_EXCEEDED_MESSAGE = TinkerMessage(
    role="tool",
    content="The result of your action resulted in context length overflow.",
)
logger = logging.getLogger(__name__)


async def _rewire_solution(
    rewirer_model: TinkerModel,
    verifier: Verifier,
    task: Task,
    msg_env: SimplifiedSolverEnv,
    last_assistant_msg: TinkerMessage,
) -> QRA | None:
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

    try:
        execution_output, _ = await msg_env.rust_env.run_cargo()
        main_rs_content = await msg_env.rust_env.view_file("src/main.rs")
    except CoderMCPRuntimeError as e:
        logger.error(f"Environment error during rewiring execution: {e}")
        return None

    qa = QA(question=task.instruction, answer=answer_str)
    try:
        verification_result = await verifier.verify(
            qa=qa,
            tree_structure=msg_env.reward_fn.tree_structure,
            execution_output=execution_output,
            main_rs_content=main_rs_content,
        )
    except Exception as e:
        raise CodingEnvironmentError(
            f"Environment error during rewiring verification: {e}"
        ) from e

    if verification_result.success:
        qra = QRA(
            question=task.instruction,
            answer=answer_str,
            reasoning=reasoning_str,
        )

        logger.info("Successfully solved and rewired")
        return qra
    else:
        logger.debug(
            f"Verification of rewired answer failed: {verification_result.reasoning}"
        )
        return None


async def ss_solve_verify(
    solver_model: TinkerModel,
    verifier_model: AgentsSDKModel,
    rust_doc_analyzer: RustDocAnalyzer,
    task: Task,
    max_turns: int,
    runtime_settings: RuntimeSettings,
    qwen_no_think: bool = False,
    knowledges: str | None = None,
) -> RewireSessionResult:
    verifier = Verifier(model=verifier_model, rust_doc_analyzer=rust_doc_analyzer)
    ss_state = SimplifiedSolverEnvState.numrs2(
        task=task, qwen_no_think=qwen_no_think, knowledge=knowledges
    )
    try:
        async with runtime_settings.build_runtime() as runtime:
            async with build_simplified_solver_env(
                env_state=ss_state,
                renderer=solver_model.renderer,
                verifier=verifier,
                rust_doc_analyzer=verifier.rust_doc_analyzer,
                max_turns=max_turns,
                runtime=runtime,
            ) as env:
                msg_env = cast(SimplifiedSolverEnv, env.message_env)

                completer = TinkerMessageCompleter(
                    solver_model.sampling_client,
                    renderer=solver_model.renderer,
                    max_tokens=None,  # type: ignore
                )

                ob = await msg_env.initial_observation()
                transitions = []
                while True:
                    try:
                        ac_message = await completer(ob)
                        ac_with_logprobs = ac_message["tokens_with_logprobs"]
                    except MaximumContextExceeded:
                        logger.debug(
                            "Maximum context length exceeded during the rewiring loop"
                        )
                        ob.append(CONTEXT_EXCEEDED_MESSAGE)
                        trajectory = Trajectory(
                            transitions=transitions,
                            final_ob=solver_model.renderer.build_generation_prompt(ob),
                        )
                        return RewireSessionResultFailure(
                            task=task,
                            trials=ob,
                            conclusion="context_length_exceeded",
                            trajectory=trajectory,
                        )

                    model_input = solver_model.renderer.build_generation_prompt(ob)
                    step_result = await msg_env.step(ac_message)
                    transition = Transition(
                        ob=model_input,
                        ac=ac_with_logprobs,
                        reward=step_result.reward,
                        episode_done=step_result.episode_done,
                        metrics=getattr(step_result, "metrics", {}),
                        logs=getattr(step_result, "logs", {}),
                    )
                    transitions.append(transition)

                    ob = step_result.next_messages
                    if step_result.episode_done:
                        break

                trajectory = Trajectory(
                    transitions=transitions,
                    final_ob=solver_model.renderer.build_generation_prompt(ob),
                )

                if LLMAsAJudge.is_successful_reward(step_result.reward):
                    normalizer = KnowledgeNormalizer(model=verifier_model)
                    normalized_knowledge = await normalizer.normalize(ob)

                    verification_result = await verify_normalized_knowledge(
                        runtime=runtime,
                        normalized_knowledge=normalized_knowledge,
                        model=verifier_model,
                    )

                    if not verification_result.success:
                        logger.warning(
                            f"Knowledge verification failed: {verification_result.reasoning}"
                        )
                        return RewireSessionResultFailure(
                            task=task,
                            trials=ob,
                            conclusion="verification_failed",
                            trajectory=trajectory,
                        )

                    return RewireSessionResultSuccess(
                        task=task,
                        trials=ob,
                        trajectory=trajectory,
                        knowledge=normalized_knowledge,
                    )
                else:
                    return RewireSessionResultFailure(
                        task=task,
                        trials=ob,
                        conclusion=step_result.conclusion,
                        trajectory=trajectory,
                    )
    except (CodingEnvironmentError, CoderMCPRuntimeError) as e:
        logger.error(f"Environment error: {e}")
        return RewireSessionResultError(
            task=task,
            conclusion="environment_error",
        )
