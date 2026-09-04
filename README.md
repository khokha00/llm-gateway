# LLM Gateway

An open-source, OpenAI-compatible inference proxy: multi-backend, Redis-cached,
streaming, containerized, observable, and deployed through CI/CD.

Point any OpenAI-compatible client (`openai-python`, LangChain, etc.) at this
gateway's `base_url` and it works — same request/response schema as OpenAI's
`/v1/chat/completions`.

## Features

- **OpenAI-compatible API** — `/v1/chat/completions`, streaming and non-streaming
- **Pluggable backends** — talks to any OpenAI-compatible model server (vLLM,
  Ollama, TGI, etc.) through a common adapter interface
- **Exact-match caching** — Redis-backed, deterministic (`temperature=0`)
  requests only
- **Automatic fallback** — retries the primary backend once, then fails over
  to a secondary backend rather than erroring out
- **Rate limiting** — per-API-key, Redis token-bucket style, with usage
  tracking (`/usage/{api_key}`)
- **Observability** — Prometheus metrics (`/metrics`) + an auto-provisioned
  Grafana dashboard (latency percentiles, throughput, cache hit ratio, error
  rate, fallback rate, rate-limit rejections)
- **CI/CD** — GitHub Actions: test (mocked backend, no GPU) → build & push to
  GHCR → deploy → smoke test

## Stack

FastAPI · Redis · Docker Compose · Prometheus · Grafana · GitHub Actions ·
vLLM or Ollama (backend-agnostic)

## Quick start (no GPU required)

```bash
git clone <repo-url> llm-gateway && cd llm-gateway
docker compose up -d --build
docker compose exec ollama ollama pull llama3.2:3b
docker compose exec ollama-b ollama pull qwen2.5:1.5b

curl http://localhost:8000/health
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.2:3b","messages":[{"role":"user","content":"hi"}]}'
```

Full step-by-step instructions, including local dev without Docker, proving
fallback/rate-limiting under real conditions, load testing, and CI/CD setup:
see **[RUNBOOK.md](./RUNBOOK.md)**.

## Project layout

```
app/
├── main.py               # FastAPI entrypoint, wiring, lifespan
├── schemas.py             # OpenAI-compatible request/response models
├── routing.py              # /v1/chat/completions: cache, retry, fallback, streaming
├── cache.py                 # Redis exact-match cache
├── rate_limit.py             # Redis rate limiter + usage tracking
├── metrics.py                 # Prometheus metric definitions
└── adapters/
    ├── base.py                 # Abstract backend interface
    └── vllm_adapter.py          # OpenAI-wire-format HTTP adapter (works for vLLM or Ollama)
tests/test_routing.py       # Mocked-backend test suite (CI-safe, no GPU)
docker/Dockerfile.gateway    # Multi-stage image for the gateway service
docker-compose.yml            # Full stack: gateway + redis + 2 backends + prometheus + grafana
grafana/                       # Auto-provisioned datasource + dashboard
prometheus.yml                  # Scrape config
locustfile.py                    # Load test
.github/workflows/deploy.yml      # CI/CD pipeline
```

## Swapping in a real GPU backend later

The adapter (`app/adapters/vllm_adapter.py`) only assumes an OpenAI-compatible
HTTP endpoint — it doesn't care whether that's vLLM or Ollama. To move from
the CPU-friendly Ollama setup to real vLLM on a rented GPU:

1. Swap the `ollama` / `ollama-b` services in `docker-compose.yml` for
   `vllm/vllm-openai:latest`, with a GPU `deploy.resources.reservations` block.
2. Point `VLLM_URL` / `VLLM_URL_B` at ports `8001` / `8002` instead of
   Ollama's `11434`.
3. Everything else — routing, caching, fallback, rate limiting, metrics — is
   unchanged.

## License

MIT (or your choice — update this section).