import logging
from coder_mcp.runtime import Runtime
from adapter_agent.rl.env.runtime_pool import RuntimePool
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.util.parsing import extract_rust_code
from adapter_agent.data import QRA
from .types import RuntimeExecutionResult, VerificationOutcome

logger = logging.getLogger(__name__)


class InternalizeExecutor:
    def __init__(self, runtime_pool: RuntimePool, verifier: Verifier):
        self.runtime_pool = runtime_pool
        self.verifier = verifier

    async def run_execution_and_verification(
        self, question: str, reasoning: str, answer_text: str
    ) -> VerificationOutcome:
        code = extract_rust_code(answer_text)
        if not code:
            return VerificationOutcome(
                success=False,
                execution_output="",
                verification_output="No Rust code block found.",
            )

        question_summary = (question[:80] + "..") if len(question) > 80 else question
        logger.debug(f"🚀 Executing task: {question_summary}")

        try:
            async def _run_closure(runtime: Runtime) -> RuntimeExecutionResult:
                url = runtime.get_api_url()
                logger.debug(f"  [Runtime: {url}] Setting content (src/main.rs)...")
                await runtime.set_content("src/main.rs", code)

                logger.debug(f"  [Runtime: {url}] Running cargo run...")
                execution_output, exit_success = await runtime.run_cargo()
                
                logger.debug(f"  [Runtime: {url}] Execution completed (success={exit_success})")
                tree_output = await runtime.tree()
                return RuntimeExecutionResult(
                    execution_output=execution_output,
                    tree_output=tree_output,
                    exit_success=exit_success,
                )

            exec_res = await self.runtime_pool.execute_with_retry(_run_closure)

            if not exec_res.exit_success:
                logger.debug(f"  [Result] Execution failed. Output: {exec_res.execution_output[:200]}...")
                return VerificationOutcome(
                    success=False,
                    execution_output=exec_res.execution_output,
                    verification_output="Compilation or execution failed.",
                )

            logger.debug(f"  [Result] Execution success. Verifying output length: {len(exec_res.execution_output)}")

            qa_data = QRA(question=question, reasoning=reasoning, answer=answer_text)
            res = await self.verifier.verify(
                qa=qa_data,
                tree_structure=exec_res.tree_output,
                execution_output=exec_res.execution_output,
                main_rs_content=answer_text,
            )
            return VerificationOutcome(
                success=res.success,
                execution_output=exec_res.execution_output,
                verification_output=res.reasoning,
            )
        except Exception as e:
            logger.error(f"Execution/verification failed: {e}")
            return VerificationOutcome(
                success=False, execution_output="", verification_output=str(e)
            )
