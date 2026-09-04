"""
Runs entirely against a fake in-memory backend + fakeredis-style stub --
no vLLM, no GPU, no network. This is exactly what CI runs on every push.
"""
import fakeredis.aioredis
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.adapters.base import ModelBackend
from app.cache import ResponseCache
from app.main import app
from app.rate_limit import RateLimiter
from app.schemas import ChatCompletionChoice, ChatCompletionResponse, ChatMessage, Usage


class FakeBackend(ModelBackend):
    name = "fake"

    async def generate(self, request):
        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0, message=ChatMessage(role="assistant", content="hello from fake backend")
                )
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        )

    async def stream(self, request):
        for word in ["hello", " ", "world"]:
            yield word

    async def health(self) -> bool:
        return True


@pytest_asyncio.fixture
async def client():
    app.state.backends = {"test-model": FakeBackend()}
    app.state.fallback_backends = {}
    fake_redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    app.state.cache = ResponseCache.__new__(ResponseCache)
    app.state.cache._redis = fake_redis_client
    app.state.rate_limiter = RateLimiter.__new__(RateLimiter)
    app.state.rate_limiter._redis = fake_redis_client
    app.state.rate_limiter.limit = 60

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_chat_completion_non_streaming(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "hello from fake backend"
    assert body["usage"]["total_tokens"] == 10


@pytest.mark.asyncio
async def test_cache_hit_on_repeated_deterministic_request(client):
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0,
    }
    first = await client.post("/v1/chat/completions", json=payload)
    assert first.headers.get("X-Cache") != "HIT"

    second = await client.post("/v1/chat/completions", json=payload)
    assert second.headers.get("X-Cache") == "HIT"


@pytest.mark.asyncio
async def test_unknown_model_returns_404(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "nonexistent", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_health_endpoint(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
