"""Source-navigating solver loop.

Parallel to `ss_solve_verify` (which uses rustdoc-JSON + wiki). This variant
gives the solver agent direct read-only access to the library source via
`<grep>`, `<read>`, `<ls>` plus `<write_and_run>` and `<submit>`. The turn
loop, verifier interaction, and result types are all the same — only the
env / tool surface differs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import tinker
from agents.extensions.models.litellm_model import LitellmModel
from coder_mcp.runtime import CoderMCPRuntimeError
from oai_utils.tinker import TinkerModel
from tinker_cookbook.renderers.base import Message as TinkerMessage
from tinker_cookbook.renderers.base import TextPart
from tinker_cookbook.rl.types import TokensWithLogprobs, Trajectory, Transition

from adapter_agent.hierarchical.agent.knowledge_normalizer import AgentsSDKModel
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.types import Task
from adapter_agent.rl.completer import LiteLLMMessageCompleter
from adapter_agent.rl.env.reward import LLMAsAJudge
from adapter_agent.rl.env.runtime_pool import RuntimePool
from adapter_agent.rl.env.session_result import (
    RewireSessionResult,
    RewireSessionResultError,
    RewireSessionResultFailure,
    RewireSessionResultSuccess,
)
from adapter_agent.rl.env.simplified_solver import SSStepResult
from adapter_agent.rl.env.source_solver import (
    SourceSolverEnvState,
    build_source_solver_msg_env,
)
from adapter_agent.rl.solved_subtask import SolvedSubtask
from adapter_agent.util.exception import CodingEnvironmentError, MaximumContextExceeded

logger = logging.getLogger(__name__)

CONTEXT_EXCEEDED_MESSAGE = TinkerMessage(
    role="tool",
    content="The result of your action resulted in context length overflow.",
)

# LiteLLM consumes the stop sequence and DOES NOT include it in the response.
# Stopping at closing tags forces the agent to commit to ONE tool action per
# turn instead of pre-rolling an imagined multi-step session in one message.
# After generation we restore the eaten closing tag so the env regex matches.
#
# IMPORTANT: Gemini's `stopSequences` field is silently IGNORED when the list
# has 5 or more entries (verified empirically with gemini-3-flash-preview).
# We have 5 tool tags total, so when search is enabled we drop `</submit>` —
# submit is the final action of a session, so trailing junk after it doesn't
# matter (the regex extracts the body regardless). The other four are
# critical for preventing autoregressive multi-tag hallucination mid-session.
def _stop_sequences_for(
    enable_search_tools: bool, enable_write_and_run: bool = True
) -> tuple[list[str], list[str]]:
    """Return `(stop_sequences, tag_names_for_close_restoration)`.

    Gemini's `stopSequences` field is silently IGNORED when the list has 5+
    entries; the helper picks the busiest 4 closers based on which tools
    are enabled, dropping `</submit>` last (it's the terminal action so
    trailing junk after it doesn't matter — the regex extracts the body
    regardless).
    """
    closers: list[str] = []
    tag_names: list[str] = ["submit"]  # always recoverable
    if enable_write_and_run:
        closers.append("</write_and_run>")
        tag_names.append("write_and_run")
    if enable_search_tools:
        closers.extend(["</grep>", "</read>", "</ls>"])
        tag_names.extend(["grep", "read", "ls"])
    if len(closers) < 4:
        # Have room for </submit> too — include it for tighter stopping.
        closers.append("</submit>")
    return closers, tag_names


def _restore_closing_tag(text: str, tag_names: list[str]) -> str:
    """If exactly one tool tag was opened but its closer was eaten by the
    stop sequence, append the matching `</tag>`."""
    for name in tag_names:
        opener = f"<{name}>"
        closer = f"</{name}>"
        if opener in text and closer not in text:
            return text + closer
    return text


def _flatten_content(msg: TinkerMessage) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if p.get("type") == "text")
    return "" if content is None else str(content)


async def solve_verify(
    solver_model: TinkerModel | LitellmModel,
    verifier_model: AgentsSDKModel,
    task: Task,
    libdir: Path,
    library_name: str,
    runtime_pool: RuntimePool,
    max_turns: int,
    qwen_no_think: bool = False,
    solved_subtasks: list[SolvedSubtask] | None = None,
    reference_knowledge: str | None = None,
    enable_search_tools: bool = True,
    enable_write_and_run: bool = True,
) -> RewireSessionResult:
    """Drive a `SourceSolverEnv` session to completion.

    Mirrors the public surface of `ss_solve_verify`:
      - same `RewireSessionResult` return type;
      - same support for both `TinkerModel` and `LitellmModel` solvers;
      - same verifier path (`Verifier` against the submission).

    Differences from `ss_solve_verify`:
      - `libdir: Path` replaces the rustdoc-analyzer + wiki-manager pair;
      - the env exposes `<grep>` / `<read>` / `<ls>` instead of
        `<search_library_doc>` / `<wiki_*>`;
      - no `internalized_knowledge` prefill or blocked-knowledge plumbing.
    """
    verifier = Verifier(model=verifier_model, library_name=library_name)
    env_state = SourceSolverEnvState.for_library(
        task=task,
        library_name=library_name,
        libdir=libdir,
        qwen_no_think=qwen_no_think,
        solved_subtasks=solved_subtasks,
        reference_knowledge=reference_knowledge,
    )

    try:
        renderer = (
            solver_model.renderer if isinstance(solver_model, TinkerModel) else None
        )
        msg_env = await build_source_solver_msg_env(
            env_state=env_state,
            verifier=verifier,
            runtime_pool=runtime_pool,
            max_turns=max_turns,
            renderer=renderer,
            enable_search_tools=enable_search_tools,
            enable_write_and_run=enable_write_and_run,
        )
        litellm_stop_sequences, litellm_tag_names = _stop_sequences_for(
            enable_search_tools, enable_write_and_run
        )

        ob_messages = await msg_env.initial_observation()

        if isinstance(solver_model, TinkerModel):
            assert renderer is not None, "TinkerModel must have a renderer"
            transitions: list[Transition] = []
            stop_condition = renderer.get_stop_sequences()
            step_result: SSStepResult | None = None
            while True:
                model_input = renderer.build_generation_prompt(ob_messages)
                try:
                    sample_result = await solver_model.sampling_client.sample_async(
                        prompt=model_input,
                        num_samples=1,
                        sampling_params=tinker.SamplingParams(
                            stop=stop_condition,
                            temperature=1.0,
                        ),
                    )
                    sampled_tokens = sample_result.sequences[0].tokens
                    maybe_logprobs = sample_result.sequences[0].logprobs
                except tinker.APIStatusError as e:
                    if (
                        e.status_code == 400
                        and "Prompt length plus max_tokens exceeds the model's context window"
                        in e.message
                    ):
                        logger.debug("Maximum context length exceeded during the loop")
                        msg_env.history.append(CONTEXT_EXCEEDED_MESSAGE)
                        return RewireSessionResultFailure(
                            task=task,
                            trials=msg_env.history,
                            conclusion="context_length_exceeded",
                            trajectory=Trajectory(
                                transitions=transitions,
                                final_ob=renderer.build_generation_prompt(msg_env.history),
                            ),
                        )
                    else:
                        raise

                assistant_message, parse_success = renderer.parse_response(sampled_tokens)
                if not parse_success:
                    transitions.append(
                        Transition(
                            ob=model_input,
                            ac=TokensWithLogprobs(
                                tokens=sampled_tokens, maybe_logprobs=maybe_logprobs
                            ),
                            reward=-1.0,
                            episode_done=True,
                            metrics={"parse_error": 1.0},
                            logs={},
                        )
                    )
                    step_result = SSStepResult(
                        reward=-1.0,
                        episode_done=True,
                        next_messages=msg_env.history,
                        conclusion="parse_failed",
                    )
                    break

                step_result = await msg_env.step(assistant_message)
                ob_messages = step_result.next_messages

                transitions.append(
                    Transition(
                        ob=model_input,
                        ac=TokensWithLogprobs(
                            tokens=sampled_tokens, maybe_logprobs=maybe_logprobs
                        ),
                        reward=step_result.reward,
                        episode_done=step_result.episode_done,
                        metrics=step_result.metrics,
                        logs={},
                    )
                )
                if step_result.episode_done:
                    break

            assert step_result is not None
            trajectory = Trajectory(
                transitions=transitions,
                final_ob=renderer.build_generation_prompt(msg_env.history),
            )
            trials = msg_env.history

        else:
            litellm_completer = LiteLLMMessageCompleter.from_litellm_model(
                solver_model,
                stop_condition=litellm_stop_sequences,
            )
            step_result = None
            while True:
                try:
                    action = await litellm_completer(ob_messages)
                except MaximumContextExceeded:
                    logger.debug("Maximum context length exceeded during the loop")
                    msg_env.history.append(CONTEXT_EXCEEDED_MESSAGE)
                    return RewireSessionResultFailure(
                        task=task,
                        trials=msg_env.history,
                        conclusion="context_length_exceeded",
                        trajectory=None,
                    )

                # `action` from LiteLLMMessageCompleter is a MessageWithLogprobs;
                # the underlying TinkerMessage is the message field.
                if hasattr(action, "message"):
                    msg_to_step = action.message  # type: ignore[attr-defined]
                else:
                    msg_to_step = action  # type: ignore[assignment]

                # Restore the closing tag the stop sequence ate.
                raw_text = _flatten_content(msg_to_step)
                restored = _restore_closing_tag(raw_text, litellm_tag_names)
                if restored != raw_text:
                    msg_to_step = TinkerMessage(
                        role=msg_to_step.get("role", "assistant"),
                        content=restored,
                    )

                step_result = await msg_env.step(msg_to_step)
                ob_messages = step_result.next_messages

                if step_result.episode_done:
                    break

            trials = msg_env.history
            trajectory = None

        assert step_result is not None
        if not LLMAsAJudge.is_successful_reward(step_result.reward):
            return RewireSessionResultFailure(
                task=task,
                trials=trials,
                conclusion=step_result.conclusion,
                trajectory=trajectory,
            )

        return RewireSessionResultSuccess(
            task=task,
            trials=trials,
            trajectory=trajectory,
        )

    except (CodingEnvironmentError, CoderMCPRuntimeError) as e:
        logger.error(f"Environment error: {e}")
        return RewireSessionResultError(
            task=task,
            conclusion="environment_error",
        )


# `TextPart` import retained for symmetry with ss_solve_verify in case
# downstream wraps prefill content; not used today.
_ = TextPart
