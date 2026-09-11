"""Mistral native SDK provider implementation."""

import os
from typing import Any, Dict

from mistralai.client import Mistral

from app.providers.base import (
    BaseLLMProvider,
    ChatCompletionRequest,
    format_openai_chat_response,
)


class MistralProvider(BaseLLMProvider):
    """Provider integrating directly with official mistralai Python SDK."""

    def __init__(
        self,
        vendor_name: str = "mistral",
        model_name: str = "mistral-small-latest",
        api_key: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(vendor_name=vendor_name, model_name=model_name, **kwargs)
        resolved_key = api_key or os.getenv("MISTRAL_API_KEY", "")
        self.client = Mistral(api_key=resolved_key)

    def chat_completion(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Execute chat completion request via mistralai SDK."""
        model = request.model or self.model_name
        complete_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": request.messages,
        }
        if request.temperature is not None:
            complete_kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            complete_kwargs["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            complete_kwargs["top_p"] = request.top_p

        response = self.client.chat.complete(**complete_kwargs)

        if hasattr(response, "model_dump"):
            return response.model_dump()  # type: ignore[no-any-return]

        # Fallback manual formatting
        content = ""
        if response.choices and response.choices[0].message:
            content = str(response.choices[0].message.content or "")

        prompt_tokens = (
            int(response.usage.prompt_tokens)
            if (response.usage and response.usage.prompt_tokens is not None)
            else 0
        )
        completion_tokens = (
            int(response.usage.completion_tokens)
            if (response.usage and response.usage.completion_tokens is not None)
            else 0
        )

        return format_openai_chat_response(
            model=model,
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
