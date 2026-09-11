"""OpenRouter provider implementation using OpenAI SDK."""

import os
from typing import Any, Dict

from openai import OpenAI

from app.providers.base import BaseLLMProvider, ChatCompletionRequest


class OpenRouterProvider(BaseLLMProvider):
    """Provider connecting to OpenRouter via official OpenAI SDK."""

    def __init__(
        self,
        vendor_name: str = "openrouter",
        model_name: str = "meta-llama/llama-3.3-70b-instruct",
        api_key: str = "",
        base_url: str = "https://openrouter.ai/api/v1",
        **kwargs: Any,
    ) -> None:
        super().__init__(vendor_name=vendor_name, model_name=model_name, **kwargs)
        resolved_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        resolved_base_url = (
            base_url
            or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            or kwargs.get("base_url")
            or "https://openrouter.ai/api/v1"
        )
        self.client = OpenAI(
            api_key=resolved_key,
            base_url=resolved_base_url,
        )

    def chat_completion(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Execute chat completion request via OpenRouter endpoint."""
        model = request.model or self.model_name
        create_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": request.messages,
        }
        if request.temperature is not None:
            create_kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            create_kwargs["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            create_kwargs["top_p"] = request.top_p

        response = self.client.chat.completions.create(**create_kwargs)
        return response.model_dump()  # type: ignore[no-any-return]
