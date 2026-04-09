import asyncio
import logging
from dataclasses import dataclass, field

from ray.actor import ActorProxy

from adapter_agent.data import QRA
from adapter_agent.hierarchical.agent.generator import GeneratorAgent
from adapter_agent.hierarchical.agent.verifier import VerificationResult, Verifier
from adapter_agent.hierarchical.gh import Library
from adapter_agent.hierarchical.types import Knowledge
from adapter_agent.internalize.global_state import GlobalState
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.model_helper import get_gemini
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.util.logger_util import setup_base_loglevel

logger = logging.getLogger(__name__)


@dataclass
class QRAGenerator:
    global_state: ActorProxy[GlobalState]
    library: Library
    verifier: Verifier
    runtime_settings: RuntimeSettings
    num_concurrent_generations: int = 64

    generator: GeneratorAgent | None = field(init=False, default=None)
    doc_analyzer: AsyncRustDocAnalyzer | None = field(init=False, default=None)
    concurrency_limit: asyncio.Semaphore | None = field(init=False, default=None)

    async def setup(self) -> None:
        """Initialize generator agent and verifier."""
        setup_base_loglevel()
        logger.info("Initializing QRAGenerator internal components...")
        self.concurrency_limit = asyncio.Semaphore(self.num_concurrent_generations)
        self.doc_analyzer = await AsyncRustDocAnalyzer.create_from_libdir(
            self.library.local_path, skip_init=True
        )
        self.generator = GeneratorAgent(
            model=get_gemini(), rust_doc_analyzer=self.doc_analyzer
        )
        self.verifier = Verifier(
            model=get_gemini(), rust_doc_analyzer=self.doc_analyzer
        )

    async def run_loop(self) -> None:
        """
        Background loop: replenishment of task pools.
        """
        await self.setup()
        while True:
            await self.replenish_all_pools()
            await asyncio.sleep(5)

    async def replenish_all_pools(self) -> None:
        """Analyze pool status and trigger replenishment for all knowledges concurrently."""
        plan: list[
            tuple[Knowledge, int]
        ] = await self.global_state.get_replenishment_plan.remote()  # type: ignore[assignment]
        if not plan:
            return

        # Create a flat list of all generation tasks across all knowledges in the plan
        tasks = []
        for k, count in plan:
            for _ in range(count):
                tasks.append(self._generate_single_verified_qra(k))

        await asyncio.gather(*tasks)

    async def _generate_single_verified_qra(self, k: Knowledge) -> None:
        """Generate one QRA and verify it, pushing to GlobalState if successful."""
        assert self.concurrency_limit is not None
        async with self.concurrency_limit:
            await self.global_state.report_qra_generation_start.remote(k.id)  # type: ignore[attr-defined]
            qra_to_push = None
            try:
                assert self.generator and self.verifier
                qra = await self.generator.generate_sft(k)
                if not qra:
                    return

                verification_result = await self._verify_generated_qra(qra)
                if verification_result.success:
                    qra_to_push = qra
            finally:
                await self.global_state.report_qra_generation_result.remote(  # type: ignore[attr-defined]
                    k.id, qra_to_push
                )

    async def _verify_generated_qra(self, rollout: QRA) -> VerificationResult:
        """Run Rust code and verify for newly generated tasks."""
        assert self.verifier is not None
        try:
            async with self.runtime_settings.build_runtime() as runtime:
                await runtime.set_content("src/main.rs", rollout.answer)
                execution_output, exit_success = await runtime.run_cargo()

                if not exit_success:
                    return VerificationResult(
                        success=False, reasoning="Cargo execution failed."
                    )

                tree_output = await runtime.tree()

                return await self.verifier.verify(
                    qa=rollout,
                    tree_structure=tree_output,
                    execution_output=execution_output,
                    main_rs_content=rollout.answer,
                )
        except Exception as e:
            logger.error(f"Execution verification failed during QRA generation: {e}")
            return VerificationResult(success=False, reasoning=str(e))
