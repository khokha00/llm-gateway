import redis.asyncio as redis

from app.schemas import Usage


class RateLimiter:
    """
    Fixed-window counter per API key using INCR + EXPIRE. Simple, not
    perfectly smooth (bursts at window edges), but predictable and cheap --
    fine for a v1. Swap for a sliding-window / leaky-bucket Lua script if
    edge bursts become a real problem.
    """

    def __init__(self, redis_url: str, limit_per_minute: int = 60):
        self._redis = redis.from_url(redis_url, decode_responses=True)
        self.limit = limit_per_minute

    async def check(self, api_key: str) -> tuple[bool, int]:
        key = f"ratelimit:{api_key}"
        count = await self._redis.incr(key)
        if count == 1:
            await self._redis.expire(key, 60)
        return count <= self.limit, count

    async def record_usage(self, api_key: str, usage: Usage) -> None:
        await self._redis.hincrby(f"usage:{api_key}", "prompt_tokens", usage.prompt_tokens)
        await self._redis.hincrby(f"usage:{api_key}", "completion_tokens", usage.completion_tokens)
        await self._redis.hincrby(f"usage:{api_key}", "requests", 1)

    async def get_usage(self, api_key: str) -> dict:
        data = await self._redis.hgetall(f"usage:{api_key}")
        return {k: int(v) for k, v in data.items()} or {
            "prompt_tokens": 0, "completion_tokens": 0, "requests": 0
        }

    async def close(self) -> None:
        await self._redis.aclose()
