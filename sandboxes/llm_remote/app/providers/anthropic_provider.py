"""Anthropic native SDK provider implementation."""

import os
from typing import Any, Dict, List

from anthropic import Anthropic

from app.providers.base import (
    BaseLLMProvider,
    ChatCompletionRequest,
    format_openai_chat_response,
    sanitize_credential,
    sanitize_url,
)


class AnthropicProvider(BaseLLMProvider):
    """Provider integrating directly with official anthropic Python SDK."""

    def __init__(
        self,
        vendor_name: str = "anthropic",
        model_name: str = "claude-3-5-haiku-latest",
        api_key: str = "",
        base_url: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(vendor_name=vendor_name, model_name=model_name, **kwargs)
        resolved_key = sanitize_credential(
            api_key
            or os.getenv("ANTHROPIC_API_KEY", "")
            or os.getenv("ANTHROPIC_AUTH_TOKEN", "")
        )
        resolved_base_url = sanitize_url(
            base_url or os.getenv("ANTHROPIC_BASE_URL", "") or kwargs.get("base_url")
        )
        init_kwargs: Dict[str, Any] = {"api_key": resolved_key}
        if resolved_base_url:
            init_kwargs["base_url"] = resolved_base_url
        self.client = Anthropic(**init_kwargs)

    def chat_completion(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Execute chat completion request via anthropic SDK."""
        model = request.model or self.model_name

        system_prompts: List[str] = []
        anthropic_messages: List[Dict[str, Any]] = []

        for msg in request.messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_prompts.append(str(content))
            elif role in ("user", "assistant"):
                anthropic_messages.append({"role": role, "content": content})

        # Anthropic requires max_tokens to be explicitly provided
        max_tokens = request.max_tokens if request.max_tokens is not None else 1024

        create_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
        }
        if system_prompts:
            create_kwargs["system"] = "\n\n".join(system_prompts)
        if request.temperature is not None:
            create_kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            create_kwargs["top_p"] = request.top_p

        response = self.client.messages.create(**create_kwargs)

        # Extract text content from Anthropic content blocks
        extracted_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                extracted_text += block.text

        prompt_tokens = response.usage.input_tokens if response.usage else 0
        completion_tokens = response.usage.output_tokens if response.usage else 0
        finish_reason = response.stop_reason or "stop"

        return format_openai_chat_response(
            model=model,
            content=extracted_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )
