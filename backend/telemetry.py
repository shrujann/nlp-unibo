import os

from opentelemetry import trace
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

_tracer: trace.Tracer | None = None

# Chainlit internal events — not useful for food-log analysis
_SKIP_SPAN_NAMES = frozenset({
    "GET /user",
    "GET /project/settings",
    "GET /auth/config",
    "POST /message",
    "project/transition",
    "project/settings",
    "ui/token_budget",
    "connect",
    "disconnect",
})

_SKIP_PREFIXES = (
    "GET /",
    "POST /",
    "project/",
    "ui/",
)


class _FilteringProcessor(SpanProcessor):
    """Drops noisy Chainlit-internal spans before they reach the exporter."""

    def __init__(self, delegate: SpanProcessor) -> None:
        self._delegate = delegate

    def _should_skip(self, name: str) -> bool:
        return name in _SKIP_SPAN_NAMES or any(name.startswith(p) for p in _SKIP_PREFIXES)

    def on_start(self, span, parent_context=None):
        if not self._should_skip(span.name):
            self._delegate.on_start(span, parent_context)

    def on_end(self, span):
        if not self._should_skip(span.name):
            self._delegate.on_end(span)

    def shutdown(self):
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._delegate.force_flush(timeout_millis)


def setup_phoenix() -> trace.Tracer:
    """
    Start Phoenix locally and register the OTel tracer.
    Call once at app startup. Subsequent calls return the same tracer.
    """
    global _tracer
    if _tracer is not None:
        return _tracer

    import phoenix as px
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from phoenix.otel import register

    px.launch_app()

    tracer_provider: TracerProvider = register(
        project_name=os.getenv("PHOENIX_PROJECT", "food-log"),
        auto_instrument=False,
    )

    # Replace Phoenix's default SimpleSpanProcessor with our filtering wrapper.
    # Phoenix exports to http://localhost:6006/v1/traces over HTTP.
    phoenix_exporter = OTLPSpanExporter(
        endpoint="http://localhost:6006/v1/traces",
    )
    tracer_provider.add_span_processor(
        _FilteringProcessor(SimpleSpanProcessor(phoenix_exporter))
    )

    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    _tracer = trace.get_tracer("food_log")
    return _tracer


def get_tracer() -> trace.Tracer:
    """Return the active tracer. setup_phoenix() must have been called first."""
    if _tracer is None:
        raise RuntimeError("call setup_phoenix() before get_tracer()")
    return _tracer
