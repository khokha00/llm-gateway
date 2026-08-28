import json
from typing import AsyncGenerator

import httpx
from loguru import logger

from app.adapters.base import ModelBackend
from app.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Usage,
)


class VLLMAdapter(ModelBackend):
    """
    Talks to a vLLM instance running in OpenAI-compatible server mode
    (`python -m vllm.entrypoints.openai.api_server`). Implemented as an
    explicit adapter -- not a transparent proxy -- so the gateway's own
    schema stays authoritative even though vLLM's wire format happens to
    already match OpenAI's.
    """

    def __init__(self, name: str, base_url: str, model_id: str, timeout: float = 60.0):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        payload = {
            "model": self.model_id,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "stop": request.stop,
            "stream": False,
        }
        resp = await self._client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})
        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=choice["message"]["content"],
                    ),
                    finish_reason=choice.get("finish_reason", "stop"),
                )
            ],
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        payload = {
            "model": self.model_id,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "stop": request.stop,
            "stream": True,
        }
        async with self._client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content")
                    if delta:
                        yield delta
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    logger.warning(f"[{self.name}] malformed stream chunk skipped: {e}")

    async def health(self) -> bool:
        try:
            resp = await self._client.get("/health", timeout=3.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
