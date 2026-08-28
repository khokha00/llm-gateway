import hashlib
import json
import os
from typing import Optional

import redis.asyncio as redis

from app.schemas import ChatCompletionRequest, ChatCompletionResponse

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))


def _cache_key(request: ChatCompletionRequest) -> str:
    """
    Only cache-eligible fields go into the hash. `stream` and `user` are
    excluded on purpose -- they don't change the content of the answer,
    just how it's delivered / attributed.
    """
    payload = {
        "model": request.model,
        "messages": [m.model_dump() for m in request.messages],
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
        "stop": request.stop,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    return f"cache:exact:{digest}"


class ResponseCache:
    def __init__(self, redis_url: str):
        self._redis = redis.from_url(redis_url, decode_responses=True)

    def is_cacheable(self, request: ChatCompletionRequest) -> bool:
        # Caching high-temperature creative requests produces confusing
        # duplicate-answer bugs -- only cache deterministic (temp=0) calls.
        return request.cache and request.temperature == 0

    async def get(self, request: ChatCompletionRequest) -> Optional[ChatCompletionResponse]:
        if not self.is_cacheable(request):
            return None
        raw = await self._redis.get(_cache_key(request))
        if raw is None:
            return None
        return ChatCompletionResponse.model_validate_json(raw)

    async def set(self, request: ChatCompletionRequest, response: ChatCompletionResponse) -> None:
        if not self.is_cacheable(request):
            return
        await self._redis.setex(
            _cache_key(request),
            CACHE_TTL_SECONDS,
            response.model_dump_json(),
        )

    async def close(self) -> None:
        await self._redis.aclose()
