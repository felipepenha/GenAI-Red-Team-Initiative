"""TrueFoundry AI Gateway provider implementation using OpenAI SDK."""

import os
from typing import Any, Dict

from openai import OpenAI

from app.providers.base import (
    BaseLLMProvider,
    ChatCompletionRequest,
    sanitize_credential,
    sanitize_url,
)


class TrueFoundryProvider(BaseLLMProvider):
    """Provider connecting to TrueFoundry AI Gateway via official OpenAI SDK."""

    def __init__(
        self,
        vendor_name: str = "truefoundry",
        model_name: str = "openai/gpt-4o-mini",
        api_key: str = "",
        base_url: str = "https://gateway.truefoundry.ai",
        **kwargs: Any,
    ) -> None:
        super().__init__(vendor_name=vendor_name, model_name=model_name, **kwargs)
        resolved_key = sanitize_credential(
            api_key
            or os.getenv("TRUEFOUNDRY_API_KEY", "")
            or os.getenv("TFY_API_KEY", "")
        )
        resolved_base_url = (
            sanitize_url(
                base_url
                or os.getenv("TRUEFOUNDRY_BASE_URL", "")
                or os.getenv("TFY_BASE_URL", "")
                or kwargs.get("base_url")
            )
            or "https://gateway.truefoundry.ai"
        )
        self.client = OpenAI(
            api_key=resolved_key,
            base_url=resolved_base_url,
        )

    def chat_completion(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Execute chat completion request via TrueFoundry AI Gateway."""
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
