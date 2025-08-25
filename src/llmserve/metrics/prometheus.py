from prometheus_client import Histogram, Gauge, Counter, start_http_server

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