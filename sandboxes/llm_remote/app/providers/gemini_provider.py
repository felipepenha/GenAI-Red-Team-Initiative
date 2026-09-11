"""Gemini native SDK provider implementation."""

import os
from typing import Any, Dict, List

from google import genai
from google.genai import types

from app.providers.base import (
    BaseLLMProvider,
    ChatCompletionRequest,
    format_openai_chat_response,
)


class GeminiProvider(BaseLLMProvider):
    """Provider integrating directly with official google-genai Python SDK."""

    def __init__(
        self,
        vendor_name: str = "gemini",
        model_name: str = "gemini-2.5-flash",
        api_key: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(vendor_name=vendor_name, model_name=model_name, **kwargs)
        resolved_key = (
            api_key
            or os.getenv("GEMINI_API_KEY", "")
            or os.getenv("GOOGLE_API_KEY", "")
        )
        self.client = genai.Client(api_key=resolved_key)

    def chat_completion(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Execute chat completion request via google-genai SDK."""
        model = request.model or self.model_name

        system_instruction: List[str] = []
        contents: Any = []

        for msg in request.messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_instruction.append(str(content))
            elif role == "user":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=str(content))],
                    )
                )
            elif role in ("assistant", "model"):
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=str(content))],
                    )
                )

        config_kwargs: Dict[str, Any] = {}
        if system_instruction:
            config_kwargs["system_instruction"] = "\n\n".join(system_instruction)
        if request.temperature is not None:
            config_kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            config_kwargs["max_output_tokens"] = request.max_tokens
        if request.top_p is not None:
            config_kwargs["top_p"] = request.top_p

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        extracted_text = response.text or ""
        prompt_tokens = 0
        completion_tokens = 0
        if response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count or 0
            completion_tokens = response.usage_metadata.candidates_token_count or 0

        finish_reason = "stop"
        if response.candidates and response.candidates[0].finish_reason:
            finish_reason = str(response.candidates[0].finish_reason)

        return format_openai_chat_response(
            model=model,
            content=extracted_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )
