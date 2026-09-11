"""Unit and integration tests for LLM Remote Sandbox vendor detection and routing."""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.providers import get_provider
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.base import ChatCompletionRequest
from app.providers.detector import (
    VENDOR_KEY_MAPPING,
    VendorConfigurationError,
    detect_active_vendor_and_model,
)
from app.providers.gemini_provider import GeminiProvider
from app.providers.mistral_provider import MistralProvider
from app.providers.openai_provider import OpenAIProvider
from app.providers.openrouter_provider import OpenRouterProvider
from app.providers.truefoundry_provider import TrueFoundryProvider


@pytest.fixture(autouse=True)
def clean_env():
    """Clear all vendor-related env vars before each test."""
    all_keys = [
        "LLM_VENDOR",
        "LLM_MODEL",
        "SANDBOX_API_KEY",
        "SANDBOX_REQUIRE_AUTH",
    ]
    for keys in VENDOR_KEY_MAPPING.values():
        all_keys.extend(keys)

    old_env = {k: os.environ.get(k) for k in all_keys}
    for k in all_keys:
        os.environ.pop(k, None)

    yield

    for k, v in old_env.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


def test_detection_fails_when_no_keys_present():
    """Verify that detect_active_vendor_and_model raises error when no credentials exist."""
    with pytest.raises(VendorConfigurationError) as exc_info:
        detect_active_vendor_and_model()
    assert "No LLM vendor API keys were detected" in str(exc_info.value)


@pytest.mark.parametrize(
    "vendor,env_var",
    [
        ("openai", "OPENAI_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
        ("gemini", "GOOGLE_API_KEY"),
        ("mistral", "MISTRAL_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
        ("truefoundry", "TRUEFOUNDRY_API_KEY"),
        ("truefoundry", "TFY_API_KEY"),
    ],
)
def test_single_vendor_auto_detection(vendor, env_var):
    """Verify auto-detection works when exactly one vendor key is provided."""
    os.environ[env_var] = "test-secret-key-12345"
    detected_vendor, detected_model, _ = detect_active_vendor_and_model()
    assert detected_vendor == vendor
    assert detected_model is not None


def test_multiple_keys_precedence():
    """Verify precedence order when multiple vendor keys are exported without a pin."""
    os.environ["ANTHROPIC_API_KEY"] = "anthropic-key"
    os.environ["MISTRAL_API_KEY"] = "mistral-key"

    detected_vendor, _, _ = detect_active_vendor_and_model()
    assert detected_vendor == "anthropic"

    # If OpenAI is also added, OpenAI takes precedence over Anthropic
    os.environ["OPENAI_API_KEY"] = "openai-key"
    detected_vendor, _, _ = detect_active_vendor_and_model()
    assert detected_vendor == "openai"


def test_pinning_via_llm_vendor_env():
    """Verify LLM_VENDOR pins the active provider even when other keys exist."""
    os.environ["OPENAI_API_KEY"] = "openai-key"
    os.environ["MISTRAL_API_KEY"] = "mistral-key"
    os.environ["LLM_VENDOR"] = "mistral"

    detected_vendor, _, _ = detect_active_vendor_and_model()
    assert detected_vendor == "mistral"


def test_pinned_vendor_missing_key_raises_error():
    """Verify that pinning a vendor without setting its key raises descriptive error."""
    os.environ["OPENAI_API_KEY"] = "openai-key"
    os.environ["LLM_VENDOR"] = "anthropic"

    with pytest.raises(VendorConfigurationError) as exc_info:
        detect_active_vendor_and_model()
    assert "Vendor 'anthropic' was selected, but no API key was found" in str(
        exc_info.value
    )


def test_model_override_via_env():
    """Verify LLM_MODEL overrides vendor default model."""
    os.environ["OPENAI_API_KEY"] = "openai-key"
    os.environ["LLM_MODEL"] = "custom-test-model"

    _, detected_model, _ = detect_active_vendor_and_model()
    assert detected_model == "custom-test-model"


def test_truefoundry_defaults():
    """Verify TrueFoundry defaults to openai/gpt-4o-mini model and correct gateway URL."""
    os.environ["TRUEFOUNDRY_API_KEY"] = "tfy-test-key"
    vendor, model, config = detect_active_vendor_and_model()
    assert vendor == "truefoundry"
    assert model == "openai/gpt-4o-mini"
    assert config.get("base_url") == "https://gateway.truefoundry.ai"


def test_get_provider_classes():
    """Verify get_provider instantiates the correct SDK wrapper class."""
    os.environ["OPENAI_API_KEY"] = "test"
    provider = get_provider("openai")
    assert isinstance(provider, OpenAIProvider)

    os.environ["ANTHROPIC_API_KEY"] = "test"
    provider = get_provider("anthropic")
    assert isinstance(provider, AnthropicProvider)

    os.environ["GEMINI_API_KEY"] = "test"
    provider = get_provider("gemini")
    assert isinstance(provider, GeminiProvider)

    os.environ["MISTRAL_API_KEY"] = "test"
    provider = get_provider("mistral")
    assert isinstance(provider, MistralProvider)

    os.environ["OPENROUTER_API_KEY"] = "test"
    provider = get_provider("openrouter")
    assert isinstance(provider, OpenRouterProvider)

    os.environ["TRUEFOUNDRY_API_KEY"] = "test"
    provider = get_provider("truefoundry")
    assert isinstance(provider, TrueFoundryProvider)


def test_health_endpoint():
    """Test /health endpoint behavior."""
    client = TestClient(app)

    # Without keys: status is warning
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "warning"

    # With key: status is ok
    os.environ["OPENAI_API_KEY"] = "test-key"
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["vendor"] == "openai"


def test_auth_rejection():
    """Test that requests without valid Bearer token are rejected."""
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 401


def test_chat_completions_mock_call():
    """Test /v1/chat/completions endpoint dispatch with mocked provider."""
    client = TestClient(app)
    os.environ["OPENAI_API_KEY"] = "mock-key"

    with patch.object(OpenAIProvider, "chat_completion") as mock_chat:
        mock_chat.return_value = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! I am a remote LLM response.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "total_tokens": 15,
            },
        }

        resp = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-mock-key"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Hello!"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert (
            data["choices"][0]["message"]["content"]
            == "Hello! I am a remote LLM response."
        )
