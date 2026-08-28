import os
import sys
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import Response
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.adapters.vllm_adapter import VLLMAdapter
from app.cache import ResponseCache
from app.rate_limit import RateLimiter
from app.routing import router as chat_router

logger.remove()
logger.add(sys.stdout, serialize=True, level=os.getenv("LOG_LEVEL", "INFO"))

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8001")
VLLM_URL_B = os.getenv("VLLM_URL_B")  # optional second backend
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    primary = VLLMAdapter(name="vllm-primary", base_url=VLLM_URL, model_id=os.getenv("PRIMARY_MODEL_ID", "default"))

    # model name -> primary adapter (routing.py resolves by request.model)
    app.state.backends = {os.getenv("PRIMARY_MODEL_ID", "default"): primary}
    app.state.fallback_backends = {}

    if VLLM_URL_B:
        secondary = VLLMAdapter(
            name="vllm-secondary", base_url=VLLM_URL_B, model_id=os.getenv("SECONDARY_MODEL_ID", "default-b")
        )
        app.state.fallback_backends[os.getenv("PRIMARY_MODEL_ID", "default")] = secondary
        app.state.backends[os.getenv("SECONDARY_MODEL_ID", "default-b")] = secondary

    app.state.cache = ResponseCache(REDIS_URL)
    app.state.rate_limiter = RateLimiter(REDIS_URL, limit_per_minute=RATE_LIMIT_PER_MIN)

    logger.info(f"gateway started, backends={list(app.state.backends.keys())}")
    yield

    await app.state.cache.close()
    await app.state.rate_limiter.close()


app = FastAPI(title="LLM Gateway", version="1.0.0", lifespan=lifespan)
app.include_router(chat_router)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    with logger.contextualize(request_id=request_id):
        logger.bind(path=request.url.path).info("request received")
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
