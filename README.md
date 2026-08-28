# LLM Gateway

Open-source, OpenAI-compatible inference proxy. Multi-backend (vLLM), Redis-cached,
streaming, containerized, observable (Prometheus/Grafana), and deployed via GitHub Actions.

## Local dev (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# terminal 1 -- serve a model with vLLM (needs a GPU)
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --port 8001

# terminal 2 -- redis
docker run -p 6379:6379 redis:7-alpine

# terminal 3 -- gateway
export REDIS_URL=redis://localhost:6379
export VLLM_URL=http://localhost:8001
export PRIMARY_MODEL_ID=Qwen/Qwen2.5-7B-Instruct-AWQ
uvicorn app.main:app --reload --port 8000
```

## Run tests (no GPU required)

```bash
pytest -v
```

## Full stack via Docker Compose (on a GPU box)

```bash
docker compose up -d --build
curl http://localhost:8000/health
```

## Try it

```bash
# non-streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-key" \
  -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "messages": [{"role": "user", "content": "Say hi in 5 words."}],
        "temperature": 0
      }'

# repeat the exact same request -> look for "X-Cache: HIT" in the response headers
curl -i http://localhost:8000/v1/chat/completions ... # same payload

# streaming
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct-AWQ","messages":[{"role":"user","content":"count to 5"}],"stream":true}'
```

## Load test

```bash
pip install locust
locust -f locustfile.py --host http://localhost:8000 \
  --users 100 --spawn-rate 5 --run-time 5m --csv results/run1 --headless
```

## Observability

- `/metrics` — Prometheus scrape target
- Prometheus UI: `http://localhost:9090`
- Grafana: `http://localhost:3000` (default admin / admin, override with `GRAFANA_PASSWORD`)
  - Add Prometheus (`http://prometheus:9090`) as a data source
  - Panels: p50/p95/p99 latency (`histogram_quantile(0.95, rate(gateway_request_latency_seconds_bucket[5m]))`),
    requests/sec (`rate(gateway_requests_total[1m])`), cache hit ratio, error rate

## CI/CD

`.github/workflows/deploy.yml` runs on every push to `main`:
`test` (mocked backend, no GPU) → `build` (push image to GHCR) → `deploy` (SSH into the GPU box, `docker compose pull && up -d`) → `smoke` (curl `/health`).

Required repo secrets: `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY`.

## Adding a second backend (Week 6)

1. Instantiate a second `VLLMAdapter` in `app/main.py` pointed at `VLLM_URL_B`.
2. It's auto-registered as the fallback for the primary model — `routing.py`
   retries the primary once (via `tenacity`), then falls over automatically
   and increments `gateway_fallback_total`.
