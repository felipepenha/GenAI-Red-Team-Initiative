"""OpenAI native SDK provider implementation."""

import os
from typing import Any, Dict

from openai import OpenAI

from app.providers.base import (
    BaseLLMProvider,
    ChatCompletionRequest,
    sanitize_credential,
    sanitize_url,
)


class OpenAIProvider(BaseLLMProvider):
    """Provider integrating directly with official openai Python SDK."""

    def __init__(
        self,
        vendor_name: str = "openai",
        model_name: str = "gpt-4o-mini",
        api_key: str = "",
        base_url: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(vendor_name=vendor_name, model_name=model_name, **kwargs)
        resolved_key = sanitize_credential(api_key or os.getenv("OPENAI_API_KEY", ""))
        resolved_base_url = sanitize_url(
            base_url or os.getenv("OPENAI_BASE_URL", "") or kwargs.get("base_url")
        )
        init_kwargs: Dict[str, Any] = {"api_key": resolved_key}
        if resolved_base_url:
            init_kwargs["base_url"] = resolved_base_url
        self.client = OpenAI(**init_kwargs)

    def chat_completion(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Execute chat completion request via openai SDK."""
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
