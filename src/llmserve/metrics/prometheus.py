from prometheus_client import Histogram, Gauge, Counter

METRIC_TTFT = Histogram("llmserve_ttft_seconds", "Time to first token")
METRIC_TPS  = Histogram("llmserve_decode_tps", "Decode tokens per second")
Q_PREFILL   = Gauge("llmserve_q_prefill_depth", "Prefill queue depth")
Q_DECODE    = Gauge("llmserve_q_decode_depth", "Decode queue depth")

# Speculative telemetry (sampled)
SPEC_ACCEPT  = Histogram("llmserve_spec_accept_ratio",
                         "Speculative acceptance ratio (0..1)",
                         buckets=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
SPEC_SPEEDUP = Histogram("llmserve_spec_speedup_ratio",
                         "Observed speculation speedup vs. baseline",
                         buckets=[0.5,0.75,1.0,1.25,1.5,2.0,3.0])

# Rate-limit telemetry
RATE_LIMIT_REJECTS = Counter("llmserve_rate_limit_rejects_total",
                             "Requests rejected due to rate limits",
                             ["tenant", "reason"])
RATE_LIMIT_RETRY = Counter("llmserve_rate_limit_retries_total",
                           "Decode reschedules due to rate limits",
                           ["tenant", "reason"])

# Per-backend TTFT (local | grpc | vllm). We also keep METRIC_TTFT (global).
BACKEND_TTFT = Histogram(
    "llmserve_ttft_seconds_by_backend",
    "Time to first token (seconds) by backend",
    labelnames=["backend"],
    buckets=(
        0.01, 0.02, 0.05, 0.075, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0,
        1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0
    ),
)

REQUESTS_ACCEPTED = Counter(
    "llmserve_requests_accepted_total",
    "Requests accepted by the router",
    labelnames=["backend", "tenant"],  # backend: local|grpc|vllm
)

STREAM_EVENTS = Counter(
    "llmserve_stream_events_total",
    "Number of streaming events (chunks/lines) observed",
    labelnames=["backend"],
)

STREAM_BYTES = Counter(
    "llmserve_stream_bytes_total",
    "Bytes streamed to the client",
    labelnames=["backend"],
)

STREAM_TOKENS = Counter(
    "llmserve_stream_tokens_total",
    "Estimated tokens streamed to the client (router-side approximation)",
    labelnames=["backend"],
)

STREAM_ERRORS = Counter(
    "llmserve_stream_errors_total",
    "Streaming errors by backend",
    labelnames=["backend", "reason"],
)

# vLLM PD–specific convenience metrics
VLLM_PD_STREAM_ERRORS = Counter(
    "llmserve_vllm_pd_stream_errors_total",
    "Errors when streaming from vLLM PD proxy/decode endpoint",
    labelnames=["reason"],
)

VLLM_PD_TOKENS_STREAMED = Counter(
    "llmserve_vllm_pd_tokens_streamed_total",
    "Estimated tokens streamed when backend is vLLM PD",
)