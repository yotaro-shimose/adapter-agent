"""PoC: synthesize Knowledge-grounded QRAs via ss_solve_verify.

Per-knowledge flow:
  1. Have Gemini propose a Question (via GeneratorAgent.generate_sft).
  2. Hand that Question to ss_solve_verify with the original Knowledge content
     pasted in as `<ReferenceKnowledge>` and wiki disabled.
  3. The solver agent iteratively writes Rust, runs cargo, fixes errors, and
     finally `<submit>`s a working program. The verifier (which now rejects
     stub-shadowing AND facade-only library usage) decides if the submission
     actually solves the question.
  4. Successful sessions are the candidates for SFT/RL data — failures get
     dropped.

Run with:
    uv run scripts/poc_synthesize_qra.py
"""

import asyncio
import logging
import re
from dataclasses import dataclass

from agents import set_tracing_disabled
from dotenv import load_dotenv
from prisma import Prisma

from adapter_agent.data import PydanticTinkerBaseMessage
from adapter_agent.hierarchical.agent.generator import GeneratorAgent
from adapter_agent.hierarchical.process.rewire import ss_solve_verify
from adapter_agent.hierarchical.types import Knowledge, Task
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.library.library_spec import LibrarySpec
from adapter_agent.model_helper import get_gemini, get_gemini_lite
from adapter_agent.rl.env.session_result import (
    RewireSessionResultSuccess,
)
from adapter_agent.simple_internalizer.data_sources import load_granular_knowledge

from prisma import Json

set_tracing_disabled(True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


HISAB_GRANULAR_ID = "granular_prep_hisab_20260504_073544"
NUM_KNOWLEDGES = 3
MAX_TURNS = 16
POC_CACHE_ID = "PoC"
# How many synthesis trials to run concurrently. Bounded so we don't overwhelm
# the local docker runtime / Gemini quota.
TRIAL_CONCURRENCY = 3


@dataclass
class TrialOutcome:
    knowledge_title: str
    question: str
    success: bool
    conclusion: str
    submit_code: str | None
    verifier_reasoning: str | None
    trials: list | None  # JSON-serializable list of TinkerMessage dicts
    reward: float | None


def _extract_submit(trials) -> str | None:
    """Pull the contents of the last `<submit>...</submit>` in the trial log."""
    for msg in reversed(trials):
        content = msg.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(p.get("text", "") for p in content if p.get("type") == "text")
        else:
            continue
        m = re.search(r"<submit>(.*?)</submit>", text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return None


def _serialize_trials(trials) -> list:
    """Convert a list of TinkerMessage TypedDicts/dicts into JSON-friendly form
    via the Pydantic wrapper used by the trajectories table."""
    return [
        PydanticTinkerBaseMessage.model_validate(m).model_dump(
            mode="json", exclude_none=True
        )
        for m in trials
    ]


async def _try_one(
    knowledge: Knowledge,
    *,
    generator: GeneratorAgent,
    solver_model,
    verifier_model,
    rust_doc_analyzer: AsyncRustDocAnalyzer,
    runtime_settings,
    library_name: str,
) -> TrialOutcome:
    logger.info(f"--- knowledge: {knowledge.title} ---")

    qra = await generator.generate_sft(knowledge)
    if qra is None:
        return TrialOutcome(
            knowledge_title=knowledge.title,
            question="(generation failed)",
            success=False,
            conclusion="generator_returned_none",
            submit_code=None,
            verifier_reasoning=None,
            trials=None,
            reward=None,
        )

    logger.info(f"Question: {qra.question[:200]}...")

    result = await ss_solve_verify(
        solver_model=solver_model,
        verifier_model=verifier_model,
        rust_doc_analyzer=rust_doc_analyzer,
        task=Task(instruction=qra.question),
        max_turns=MAX_TURNS,
        runtime_settings=runtime_settings,
        wiki_manager=None,
        reference_knowledge=knowledge.content,
        library_name=library_name,
    )

    success = isinstance(result, RewireSessionResultSuccess)
    submit_code = None
    reasoning = getattr(result, "reasoning", None)
    trials_serialized: list | None = None
    if hasattr(result, "trials") and result.trials is not None:
        submit_code = _extract_submit(result.trials)
        try:
            trials_serialized = _serialize_trials(result.trials)
        except Exception:
            logger.exception(f"Failed to serialize trials for '{knowledge.title}'")
            trials_serialized = None
    reward = getattr(result, "reward", None)

    return TrialOutcome(
        knowledge_title=knowledge.title,
        question=qra.question,
        success=success,
        conclusion=getattr(result, "conclusion", "unknown"),
        submit_code=submit_code,
        verifier_reasoning=reasoning,
        trials=trials_serialized,
        reward=reward,
    )


async def _persist_outcome(
    prisma: Prisma,
    outcome: TrialOutcome,
    knowledge_id: str,
) -> None:
    """Write the synthesis outcome to `sft_cache_items`. Failed trials are
    kept too (verified=false) so the graphvis tab can show what got rejected."""
    answer = outcome.submit_code or ""
    data: dict = {
        "cache_id": POC_CACHE_ID,
        "knowledge_id": knowledge_id,
        "knowledge_title": outcome.knowledge_title,
        "question": outcome.question,
        "reasoning": "",
        "answer": answer,
        "verified": outcome.success,
        "verifier_reasoning": outcome.verifier_reasoning or "",
        "conclusion": outcome.conclusion,
    }
    if outcome.trials is not None:
        data["trials_json"] = Json(outcome.trials)
    if outcome.reward is not None:
        data["reward"] = outcome.reward
    await prisma.sftcacheitem.create(data=data)


async def main() -> None:
    load_dotenv()
    library_spec = LibrarySpec.hisab()

    if not (library_spec.libdir / "target/doc" / f"{library_spec.name}.json").exists():
        raise SystemExit(
            f"RustDoc JSON missing for {library_spec.name}. Run rustdoc first."
        )

    outcomes: list[TrialOutcome] = []

    prisma = Prisma()
    await prisma.connect()
    try:
        all_knowledge = await load_granular_knowledge(prisma, HISAB_GRANULAR_ID)

        # Wipe any prior run of this PoC cache so each invocation starts fresh.
        # Cascade drops the items rows along with the parent.
        existing = await prisma.sftcache.find_unique(where={"id": POC_CACHE_ID})
        if existing is not None:
            await prisma.sftcache.delete(where={"id": POC_CACHE_ID})
            logger.info(f"Cleared existing '{POC_CACHE_ID}' SFT cache.")
        await prisma.sftcache.create(
            data={
                "id": POC_CACHE_ID,
                "granular_id": HISAB_GRANULAR_ID,
                "library_name": library_spec.name,
                "description": "Knowledge-grounded QRA synthesis via ss_solve_verify (PoC).",
            }
        )

        knowledge_sample = all_knowledge[:NUM_KNOWLEDGES]
        logger.info(
            f"Loaded {len(all_knowledge)} knowledge rows; trialing first {len(knowledge_sample)}."
        )

        solver_model = get_gemini()
        verifier_model = get_gemini_lite()
        generator = GeneratorAgent(model=get_gemini())

        rust_doc_analyzer = await AsyncRustDocAnalyzer.create_from_libdir(library_spec.libdir)
        runtime_settings = library_spec.docker_runtime()

        sem = asyncio.Semaphore(TRIAL_CONCURRENCY)

        async def _run_and_persist(k):
            async with sem:
                try:
                    outcome = await _try_one(
                        k,
                        generator=generator,
                        solver_model=solver_model,
                        verifier_model=verifier_model,
                        rust_doc_analyzer=rust_doc_analyzer,
                        runtime_settings=runtime_settings,
                        library_name=library_spec.name,
                    )
                except Exception:
                    logger.exception(f"Trial crashed for '{k.title}'")
                    outcome = TrialOutcome(
                        knowledge_title=k.title,
                        question="(crashed before completion)",
                        success=False,
                        conclusion="exception",
                        submit_code=None,
                        verifier_reasoning=None,
                    )
                await _persist_outcome(prisma, outcome, knowledge_id=k.id)
                return outcome

        async with rust_doc_analyzer:
            outcomes.extend(
                await asyncio.gather(*[_run_and_persist(k) for k in knowledge_sample])
            )
    finally:
        await prisma.disconnect()

    print("\n" + "=" * 80)
    print("PoC SUMMARY")
    print("=" * 80)
    success_count = sum(1 for o in outcomes if o.success)
    print(f"Success: {success_count}/{len(outcomes)}\n")
    for i, o in enumerate(outcomes, 1):
        print(f"[{i}] {o.knowledge_title} — {'✓' if o.success else '✗'} ({o.conclusion})")
        print(f"  Q: {o.question[:200]}{'...' if len(o.question) > 200 else ''}")
        if o.verifier_reasoning:
            r = o.verifier_reasoning
            print(f"  Verifier: {r[:300]}{'...' if len(r) > 300 else ''}")
        if o.submit_code:
            preview = o.submit_code if len(o.submit_code) <= 600 else o.submit_code[:600] + "..."
            print("  Submit:")
            print("  " + preview.replace("\n", "\n  "))
        print()


if __name__ == "__main__":
    asyncio.run(main())
