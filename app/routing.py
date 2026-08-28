import json
import time

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import httpx

from app.cache import ResponseCache
from app.metrics import (
    CACHE_HITS,
    CACHE_MISSES,
    FALLBACK_TOTAL,
    RATE_LIMITED_TOTAL,
    REQUEST_LATENCY,
    REQUESTS_TOTAL,
    TOKENS_TOTAL,
)
from app.schemas import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ErrorDetail,
    ErrorResponse,
)

router = APIRouter()


@retry(
    stop=stop_after_attempt(2),
    wait=wait_fixed(1),
    retry=retry_if_exception_type(httpx.HTTPError),
    reraise=True,
)
async def _call_with_retry(adapter, request: ChatCompletionRequest):
    return await adapter.generate(request)


async def _resolve_backend(app, model: str):
    adapter = app.state.backends.get(model)
    if adapter is None:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error=ErrorDetail(message=f"Unknown model '{model}'", code="model_not_found")
            ).model_dump(),
        )
    return adapter


@router.post("/v1/chat/completions")
async def chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
    authorization: str = Header(default="anonymous"),
):
    app = request.app
    cache: ResponseCache = app.state.cache
    rate_limiter = app.state.rate_limiter
    api_key = authorization.replace("Bearer ", "")

    allowed, count = await rate_limiter.check(api_key)
    if not allowed:
        RATE_LIMITED_TOTAL.labels(api_key=api_key).inc()
        raise HTTPException(
            status_code=429,
            detail=ErrorResponse(
                error=ErrorDetail(message="Rate limit exceeded", code="rate_limit_exceeded")
            ).model_dump(),
        )

    primary = await _resolve_backend(app, payload.model)
    fallback = app.state.fallback_backends.get(payload.model)

    if payload.stream:
        return StreamingResponse(
            _stream_response(app, payload, primary, fallback, api_key),
            media_type="text/event-stream",
        )

    cached = await cache.get(payload)
    if cached is not None:
        CACHE_HITS.inc()
        from fastapi import Response
        return Response(
            content=cached.model_dump_json(),
            media_type="application/json",
            headers={"X-Cache": "HIT"},
        )
    CACHE_MISSES.inc()

    start = time.monotonic()
    backend_used = primary.name
    try:
        try:
            response = await _call_with_retry(primary, payload)
        except httpx.HTTPError:
            if fallback is None:
                raise
            logger.warning(f"primary backend '{primary.name}' failed, falling back to '{fallback.name}'")
            FALLBACK_TOTAL.inc()
            backend_used = fallback.name
            response = await fallback.generate(payload)

        REQUESTS_TOTAL.labels(backend=backend_used, status="success").inc()
        TOKENS_TOTAL.labels(backend=backend_used, direction="prompt").inc(response.usage.prompt_tokens)
        TOKENS_TOTAL.labels(backend=backend_used, direction="completion").inc(response.usage.completion_tokens)
        await rate_limiter.record_usage(api_key, response.usage)
        await cache.set(payload, response)
        return response

    except httpx.HTTPError as e:
        REQUESTS_TOTAL.labels(backend=backend_used, status="error").inc()
        logger.error(f"backend call failed: {e}")
        raise HTTPException(
            status_code=502,
            detail=ErrorResponse(
                error=ErrorDetail(message="Upstream backend error", code="backend_error")
            ).model_dump(),
        )
    finally:
        REQUEST_LATENCY.labels(backend=backend_used).observe(time.monotonic() - start)


async def _stream_response(app, payload: ChatCompletionRequest, primary, fallback, api_key: str):
    """
    Adapters yield raw text deltas -- this layer owns re-wrapping into the
    gateway's own ChatCompletionChunk + SSE framing, so a second backend's
    wire format never leaks to clients.
    """
    backend_used = primary.name
    source = primary
    try:
        # Probe primary health before committing the whole stream to it.
        if fallback is not None and not await primary.health():
            logger.warning(f"primary '{primary.name}' unhealthy, streaming from fallback")
            FALLBACK_TOTAL.inc()
            backend_used = fallback.name
            source = fallback
    except Exception:
        pass

    start = time.monotonic()
    try:
        async for delta in source.stream(payload):
            chunk = ChatCompletionChunk(
                model=payload.model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(content=delta),
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        final_chunk = ChatCompletionChunk(
            model=payload.model,
            choices=[ChatCompletionChunkChoice(index=0, delta=ChatCompletionChunkDelta(), finish_reason="stop")],
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        REQUESTS_TOTAL.labels(backend=backend_used, status="success").inc()
    except Exception as e:
        logger.error(f"stream failed on '{backend_used}': {e}")
        REQUESTS_TOTAL.labels(backend=backend_used, status="error").inc()
        error_payload = json.dumps({"error": {"message": "stream interrupted", "code": "stream_error"}})
        yield f"data: {error_payload}\n\n"
    finally:
        REQUEST_LATENCY.labels(backend=backend_used).observe(time.monotonic() - start)


@router.get("/usage/{api_key}")
async def get_usage(api_key: str, request: Request):
    return await request.app.state.rate_limiter.get_usage(api_key)
