import logging
import os


class TruncatingLogFilter(logging.Filter):
    """
    Filter that truncates long log messages from openai.agents
    specifically targeting the dumped prompts in error runs.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "openai.agents":
            # Check if this is the error message with the huge prompt dump
            if (
                isinstance(record.msg, str)
                and "Error getting response; filtered.input=" in record.msg
            ):
                # Check args if present (the input list might be in args)
                if record.args:
                    new_args = []
                    modified = False
                    for arg in record.args:
                        arg_str = str(arg)
                        # Heuristic: if it looks like a huge list representation
                        if len(arg_str) > 1000 and (
                            arg_str.startswith("[{") or arg_str.startswith("{")
                        ):
                            new_args.append(
                                f"<Truncated Input of length {len(arg_str)}...>"
                            )
                            modified = True
                        else:
                            new_args.append(arg)  # type: ignore
                    if modified:
                        record.args = tuple(new_args)

            # Also check the message itself if it's already formatted
            if len(str(record.msg)) > 5000 and "filtered.input=" in str(record.msg):
                record.msg = (
                    record.msg[:1000]
                    + f" ... <Truncated Message of length {len(record.msg)}> ... "
                    + record.msg[-1000:]
                )
        return True


# Filter out specific openai.agents warning
class OpenAITracingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if (
            record.name == "openai.agents"
            and "Tracing client error 400" in record.getMessage()
        ):
            return False
        return True


def setup_base_loglevel():
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("mcp.client.streamable_http").setLevel(logging.ERROR)
    logging.getLogger("weave").setLevel(logging.ERROR)
    logging.getLogger("coder_mcp").setLevel(logging.WARNING)
    logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
    logging.getLogger("adapter_agent.hierarchical.process.rewire").setLevel(logging.WARNING)
    os.environ["WEAVE_LOG_LEVEL"] = "ERROR"

    # Suppress the specific extension warning from tinker_cookbook
    SuppressExtensionWarning.suppress_trainonwhat_warning()

    # Apply truncating filter to openai.agents
    agent_logger = logging.getLogger("openai.agents")
    # Avoid adding multiple filters if called multiple times
    if not any(isinstance(f, TruncatingLogFilter) for f in agent_logger.filters):
        agent_logger.addFilter(TruncatingLogFilter())
    if not any(isinstance(f, OpenAITracingFilter) for f in agent_logger.filters):
        agent_logger.addFilter(OpenAITracingFilter())


class SuppressExtensionWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "train_on_what=ALL_ASSISTANT_MESSAGES" not in record.getMessage()

    @classmethod
    def suppress_trainonwhat_warning(cls):
        logging.getLogger("tinker_cookbook.renderers.base").addFilter(
            SuppressExtensionWarning()
        )
