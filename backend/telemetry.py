import os
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_tracer: trace.Tracer | None = None

# Only LangChain agent spans are useful — drop everything Chainlit/HTTP emits
_KEEP_PREFIXES = (
    "langchain",
    "chat",
    "llm",
    "tool",
    "chain",
    "retriever",
    "agent",
    "food",
)


class _FilteringProcessor(SpanProcessor):
    """Allows only LangChain/agent spans through to Phoenix."""

    def __init__(self, delegate: SpanProcessor) -> None:
        self._delegate = delegate

    def _should_keep(self, name: str) -> bool:
        lower = name.lower()
        return any(lower.startswith(p) for p in _KEEP_PREFIXES)

    def on_start(self, span, parent_context=None):
        if self._should_keep(span.name):
            self._delegate.on_start(span, parent_context)

    def on_end(self, span):
        if self._should_keep(span.name):
            self._delegate.on_end(span)

    def shutdown(self):
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._delegate.force_flush(timeout_millis)


def setup_phoenix() -> trace.Tracer:
    """
    Start Phoenix and wire up OTel tracing for LangChain spans only.
    Call once at app startup; subsequent calls return the cached tracer.
    """
    global _tracer
    if _tracer is not None:
        return _tracer

    import phoenix as px
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as GRPCExporter,
    )

    # Persist to project-local .phoenix/ directory
    phoenix_dir = Path(__file__).parent.parent / ".phoenix"
    phoenix_dir.mkdir(exist_ok=True)
    os.environ["PHOENIX_WORKING_DIR"] = str(phoenix_dir)

    px.launch_app()

    project = os.getenv("PHOENIX_PROJECT", "food-log")

    # Build the provider ourselves — avoids register() adding an unfiltered processor
    provider = TracerProvider(
        resource=Resource.create({"project.name": project})
    )
    grpc_exporter = GRPCExporter(endpoint="localhost:4317", insecure=True)
    provider.add_span_processor(
        _FilteringProcessor(BatchSpanProcessor(grpc_exporter))
    )
    trace.set_tracer_provider(provider)

    LangChainInstrumentor().instrument(tracer_provider=provider)

    _tracer = trace.get_tracer("food_log")
    return _tracer


def get_tracer() -> trace.Tracer:
    if _tracer is None:
        raise RuntimeError("call setup_phoenix() before get_tracer()")
    return _tracer
