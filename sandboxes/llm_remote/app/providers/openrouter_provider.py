"""OpenRouter native SDK provider implementation."""

import os
from typing import Any, Dict

from openrouter import OpenRouter

from app.providers.base import BaseLLMProvider, ChatCompletionRequest


class OpenRouterProvider(BaseLLMProvider):
    """Provider connecting to OpenRouter via official openrouter Python SDK."""

    def __init__(
        self,
        vendor_name: str = "openrouter",
        model_name: str = "meta-llama/llama-3.3-70b-instruct",
        api_key: str = "",
        base_url: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(vendor_name=vendor_name, model_name=model_name, **kwargs)
        resolved_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        resolved_base_url = (
            base_url
            or os.getenv("OPENROUTER_BASE_URL", "")
            or kwargs.get("base_url")
            or None
        )
        init_kwargs: Dict[str, Any] = {"api_key": resolved_key}
        if resolved_base_url:
            init_kwargs["server_url"] = resolved_base_url
        self.client = OpenRouter(**init_kwargs)

    def chat_completion(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Execute chat completion request via openrouter SDK."""
        model = request.model or self.model_name
        send_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": request.messages,
        }
        if request.temperature is not None:
            send_kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            send_kwargs["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            send_kwargs["top_p"] = request.top_p

        response = self.client.chat.send(**send_kwargs)
        if hasattr(response, "model_dump"):
            return response.model_dump()  # type: ignore[no-any-return]
        return response  # type: ignore[return-value]
