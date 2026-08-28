from abc import ABC, abstractmethod
from typing import AsyncGenerator

from app.schemas import ChatCompletionRequest, ChatCompletionResponse


class ModelBackend(ABC):
    """
    Every serving backend (vLLM, TGI, a second vLLM instance, a hosted API,
    etc.) implements this interface. Routing code never talks to a backend's
    native wire format directly -- only through generate()/stream() -- so
    swapping or adding a backend never touches routing.py.
    """

    name: str

    @abstractmethod
    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        ...

    @abstractmethod
    async def stream(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """Yields raw text deltas (not SSE-framed, not JSON) -- the routing
        layer owns re-wrapping into ChatCompletionChunk + SSE framing."""
        ...

    @abstractmethod
    async def health(self) -> bool:
        ...
