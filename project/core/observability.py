import logging
import config

logger = logging.getLogger(__name__)


class Observability:
    """
    Langfuse observability wrapper for LangChain/LangGraph.

    Uses the official LangChain CallbackHandler integration which automatically
    captures all LLM generations, tool calls, retriever spans, and graph nodes
    — including the Langfuse Agent Graph view.

    Per-request metadata (trace_name, session_id, user_id) is set by calling
    get_handler(session_id=..., trace_name=...) each turn, which creates a
    fresh scoped CallbackHandler while reusing the same authenticated client.

    Docs: https://langfuse.com/docs/integrations/langchain/tracing
    """

    def __init__(self):
        self._enabled = config.LANGFUSE_ENABLED
        self._client = None

        if not self._enabled:
            return

        if not config.LANGFUSE_PUBLIC_KEY or not config.LANGFUSE_SECRET_KEY:
            logger.warning("Langfuse enabled but API keys are missing — skipping")
            self._enabled = False
            return

        try:
            from langfuse import get_client
            self._client = get_client()

            if self._client.auth_check():
                print("✅ Langfuse client authenticated and ready.")
            else:
                print("⚠️  Langfuse authentication failed. Check your credentials and host.")
                self._enabled = False

        except Exception as exc:
            logger.warning("Could not initialize Langfuse client: %s", exc)
            self._enabled = False

    def get_handler(
        self,
        session_id: str | None = None,
        trace_name: str = "rag-chat",
        user_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ):
        """
        Return a per-request CallbackHandler with trace metadata.

        Creates a fresh handler each call so every LangGraph invocation gets
        its own named trace linked to the correct session in Langfuse.

        Args:
            session_id: Maps to the LangGraph thread_id so all turns in a
                        conversation are grouped under one Langfuse session.
            trace_name: Descriptive name shown in the Langfuse trace list
                        (e.g. "rag-chat", "rag-eval").
            user_id:    Optional user identifier for per-user filtering.
            tags:       Optional list of tags (e.g. ["production", "v2"]).
            metadata:   Optional dict of extra key/value metadata.
        """
        if not self._enabled:
            return None

        try:
            from langfuse.langchain import CallbackHandler
            handler = CallbackHandler(
                trace_name=trace_name,
                session_id=session_id,
                user_id=user_id,
                tags=tags or [],
                metadata=metadata or {},
            )
            print(f"[Langfuse] 📡 Tracing started → trace_name='{trace_name}' session_id='{session_id}'")
            return handler
        except Exception as exc:
            logger.warning("Could not create Langfuse CallbackHandler: %s", exc)
            return None

    def flush(self):
        """Flush pending traces to Langfuse before shutdown."""
        if self._client is not None:
            try:
                print("[Langfuse] 🚀 Flushing traces to Langfuse...")
                self._client.flush()
                print("[Langfuse] ✅ Traces flushed successfully.")
            except Exception as exc:
                print(f"[Langfuse] ⚠️  Flush failed: {exc}")

    @property
    def enabled(self) -> bool:
        return self._enabled