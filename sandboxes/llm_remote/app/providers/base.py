"""Base provider interface and shared data models for LLM Remote Sandbox."""

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request schema."""

    model: Optional[str] = Field(
        default=None,
        description="Model identifier to use. If omitted, uses vendor default configured in model.toml.",
    )
    messages: List[Dict[str, Any]] = Field(
        ...,
        description="List of chat messages formatted as dictionaries with 'role' and 'content'.",
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Sampling temperature.",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens to generate in the completion.",
    )
    top_p: Optional[float] = Field(
        default=None,
        description="Nucleus sampling threshold.",
    )
    stream: Optional[bool] = Field(
        default=False,
        description="Whether to stream response chunks.",
    )


def format_openai_chat_response(
    model: str,
    content: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    finish_reason: str = "stop",
) -> Dict[str, Any]:
    """Format standard OpenAI-compatible chat completion response dictionary."""
    created_ts = int(time.time())
    resp_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    total_tokens = prompt_tokens + completion_tokens

    return {
        "id": resp_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


class BaseLLMProvider(ABC):
    """Abstract Base Class for remote LLM providers."""

    def __init__(self, vendor_name: str, model_name: str, **kwargs: Any) -> None:
        self.vendor_name = vendor_name
        self.model_name = model_name
        self.extra_config = kwargs

    @abstractmethod
    def chat_completion(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Execute chat completion request against the remote provider SDK.

        Args:
            request: ChatCompletionRequest containing messages and parameters.

        Returns:
            Dict[str, Any]: OpenAI-compatible chat completion response dictionary.
        """
        pass
