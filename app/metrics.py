from prometheus_client import Counter, Histogram

REQUESTS_TOTAL = Counter(
    "gateway_requests_total", "Total requests handled", ["backend", "status"]
)
REQUEST_LATENCY = Histogram(
    "gateway_request_latency_seconds", "End-to-end request latency", ["backend"]
)
CACHE_HITS = Counter("gateway_cache_hits_total", "Cache hits")
CACHE_MISSES = Counter("gateway_cache_misses_total", "Cache misses")
TOKENS_TOTAL = Counter(
    "gateway_tokens_total", "Tokens processed", ["backend", "direction"]  # direction: prompt|completion
)
FALLBACK_TOTAL = Counter(
    "gateway_fallback_total", "Times routing fell back from primary to secondary backend"
)
RATE_LIMITED_TOTAL = Counter(
    "gateway_rate_limited_total", "Requests rejected for exceeding rate limit", ["api_key"]
)
