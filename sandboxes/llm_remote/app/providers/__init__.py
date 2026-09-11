"""Provider factory and FastAPI routes for LLM Remote Sandbox."""

import logging
import os
import time
from typing import Any, Dict, Optional, Type

from fastapi import APIRouter, Depends, Header, HTTPException

from app.providers.anthropic_provider import AnthropicProvider
from app.providers.base import BaseLLMProvider, ChatCompletionRequest
from app.providers.detector import (
    VendorConfigurationError,
    detect_active_vendor_and_model,
)
from app.providers.gemini_provider import GeminiProvider
from app.providers.mistral_provider import MistralProvider
from app.providers.openai_provider import OpenAIProvider
from app.providers.openrouter_provider import OpenRouterProvider
from app.providers.truefoundry_provider import TrueFoundryProvider

logger = logging.getLogger(__name__)

PROVIDER_CLASSES: Dict[str, Type[BaseLLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "mistral": MistralProvider,
    "openrouter": OpenRouterProvider,
    "truefoundry": TrueFoundryProvider,
}

router = APIRouter()


def verify_api_key(authorization: Optional[str] = Header(None)) -> str:
    """Validate incoming API key from client.

    Supports 'sk-mock-key', 'sk-remote-key', or a custom key defined in SANDBOX_API_KEY.
    If SANDBOX_REQUIRE_AUTH is set to 'false', allows unauthenticated calls.
    """
    require_auth = os.getenv("SANDBOX_REQUIRE_AUTH", "true").lower() != "false"
    if not require_auth:
        return "auth_disabled"

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header (Expected 'Bearer <token>')",
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authentication scheme")

    token = authorization.split(" ")[1]
    expected_token = os.getenv("SANDBOX_API_KEY", "sk-mock-key")
    accepted_tokens = {expected_token, "sk-mock-key", "sk-remote-key"}

    if token not in accepted_tokens:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return token


def get_provider(
    vendor: Optional[str] = None,
    model: Optional[str] = None,
) -> BaseLLMProvider:
    """Instantiate and return the active provider using direct native SDK."""
    detected_vendor, default_model, vendor_config = detect_active_vendor_and_model()
    active_vendor = (vendor or detected_vendor).lower()
    active_model = model or default_model

    provider_cls = PROVIDER_CLASSES.get(active_vendor)
    if not provider_cls:
        raise VendorConfigurationError(
            f"Unsupported vendor: {active_vendor}. Available: {list(PROVIDER_CLASSES.keys())}"
        )

    return provider_cls(
        vendor_name=active_vendor,
        model_name=active_model,
        **vendor_config,
    )


@router.get("/v1/models")
def list_models(token: str = Depends(verify_api_key)) -> Dict[str, Any]:
    """OpenAI-compatible models list endpoint returning the active vendor model."""
    try:
        vendor, model, _ = detect_active_vendor_and_model()
        return {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": vendor,
                }
            ],
            "active_vendor": vendor,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/chat/completions")
def chat_completions(
    request: ChatCompletionRequest,
    token: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """OpenAI-compatible chat completions endpoint routing to active remote LLM vendor.

    Invokes the vendor's official SDK directly without orchestration frameworks.
    """
    logger.debug(
        f"Received chat completion request for model: {request.model}, messages count: {len(request.messages)}"
    )
    try:
        provider = get_provider(model=request.model)
        response = provider.chat_completion(request)
        return response
    except VendorConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Remote provider error: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Remote LLM Provider Error: {str(e)}",
        )
