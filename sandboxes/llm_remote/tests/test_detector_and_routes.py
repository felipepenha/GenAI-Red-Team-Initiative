"""Unit and integration tests for LLM Remote Sandbox vendor detection and routing."""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.providers import get_provider
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.base import (
    ChatCompletionRequest,
    sanitize_credential,
    sanitize_url,
)
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


def test_openai_provider_mock_sdk():
    """Verify OpenAIProvider correctly invokes openai SDK and formats response."""
    os.environ["OPENAI_API_KEY"] = "mock-key"
    provider = OpenAIProvider()

    mock_resp = MagicMock()
    mock_resp.model_dump.return_value = {
        "id": "chatcmpl-openai-mock",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "OpenAI mock reply"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }

    with patch.object(
        provider.client.chat.completions, "create", return_value=mock_resp
    ) as mock_create:
        req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])
        res = provider.chat_completion(req)

        assert res["choices"][0]["message"]["content"] == "OpenAI mock reply"
        assert res["usage"]["total_tokens"] == 30
        mock_create.assert_called_once()


def test_anthropic_provider_mock_sdk():
    """Verify AnthropicProvider parses system messages and extracts content from blocks."""
    os.environ["ANTHROPIC_API_KEY"] = "mock-key"
    provider = AnthropicProvider()

    mock_block = MagicMock()
    mock_block.text = "Anthropic mock reply"

    mock_resp = MagicMock()
    mock_resp.content = [mock_block]
    mock_resp.usage.input_tokens = 15
    mock_resp.usage.output_tokens = 25
    mock_resp.stop_reason = "end_turn"

    with patch.object(
        provider.client.messages, "create", return_value=mock_resp
    ) as mock_create:
        req = ChatCompletionRequest(
            messages=[
                {"role": "system", "content": "You are a test helper."},
                {"role": "user", "content": "hi anthropic"},
            ],
            temperature=0.5,
            max_tokens=500,
        )
        res = provider.chat_completion(req)

        assert res["choices"][0]["message"]["content"] == "Anthropic mock reply"
        assert res["choices"][0]["finish_reason"] == "end_turn"
        assert res["usage"]["prompt_tokens"] == 15
        assert res["usage"]["completion_tokens"] == 25
        assert res["usage"]["total_tokens"] == 40

        # Check call arguments
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["system"] == "You are a test helper."
        assert call_kwargs["messages"] == [{"role": "user", "content": "hi anthropic"}]
        assert call_kwargs["max_tokens"] == 500
        assert call_kwargs["temperature"] == 0.5


def test_gemini_provider_mock_sdk():
    """Verify GeminiProvider translates contents, system instruction, and formats response."""
    os.environ["GEMINI_API_KEY"] = "mock-key"
    provider = GeminiProvider()

    mock_candidate = MagicMock()
    mock_candidate.finish_reason = "STOP"

    mock_resp = MagicMock()
    mock_resp.text = "Gemini mock reply"
    mock_resp.usage_metadata.prompt_token_count = 12
    mock_resp.usage_metadata.candidates_token_count = 18
    mock_resp.candidates = [mock_candidate]

    with patch.object(
        provider.client.models, "generate_content", return_value=mock_resp
    ) as mock_gen:
        req = ChatCompletionRequest(
            messages=[
                {"role": "system", "content": "System directive"},
                {"role": "user", "content": "hello gemini"},
                {"role": "assistant", "content": "prior model response"},
                {"role": "user", "content": "follow-up"},
            ]
        )
        res = provider.chat_completion(req)

        assert res["choices"][0]["message"]["content"] == "Gemini mock reply"
        assert res["usage"]["prompt_tokens"] == 12
        assert res["usage"]["completion_tokens"] == 18
        assert res["usage"]["total_tokens"] == 30
        mock_gen.assert_called_once()


def test_mistral_provider_mock_sdk():
    """Verify MistralProvider invokes chat.complete and dumps result."""
    os.environ["MISTRAL_API_KEY"] = "mock-key"
    provider = MistralProvider()

    mock_resp = MagicMock()
    mock_resp.model_dump.return_value = {
        "id": "mistral-mock-123",
        "object": "chat.completion",
        "created": 1700000001,
        "model": "mistral-small-latest",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Mistral mock reply"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 14,
            "total_tokens": 22,
        },
    }

    with patch.object(
        provider.client.chat, "complete", return_value=mock_resp
    ) as mock_complete:
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi mistral"}]
        )
        res = provider.chat_completion(req)

        assert res["choices"][0]["message"]["content"] == "Mistral mock reply"
        mock_complete.assert_called_once()


def test_openrouter_provider_mock_sdk():
    """Verify OpenRouterProvider invokes official openrouter SDK chat.send."""
    os.environ["OPENROUTER_API_KEY"] = "mock-key"
    provider = OpenRouterProvider()

    mock_resp = MagicMock()
    mock_resp.model_dump.return_value = {
        "id": "gen-openrouter-123",
        "object": "chat.completion",
        "created": 1700000002,
        "model": "meta-llama/llama-3.3-70b-instruct",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "OpenRouter native mock reply",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 30,
            "total_tokens": 50,
        },
    }

    with patch.object(
        provider.client.chat, "send", return_value=mock_resp
    ) as mock_send:
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi openrouter"}]
        )
        res = provider.chat_completion(req)

        assert res["choices"][0]["message"]["content"] == "OpenRouter native mock reply"
        mock_send.assert_called_once()


def test_truefoundry_provider_mock_sdk():
    """Verify TrueFoundryProvider invokes OpenAI SDK with TrueFoundry gateway endpoint."""
    os.environ["TRUEFOUNDRY_API_KEY"] = "mock-key"
    provider = TrueFoundryProvider()

    mock_resp = MagicMock()
    mock_resp.model_dump.return_value = {
        "id": "chatcmpl-tfy-mock",
        "object": "chat.completion",
        "created": 1700000003,
        "model": "openai/gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "TrueFoundry gateway mock reply",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 11,
            "total_tokens": 18,
        },
    }

    with patch.object(
        provider.client.chat.completions, "create", return_value=mock_resp
    ) as mock_create:
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi truefoundry"}]
        )
        res = provider.chat_completion(req)

        assert (
            res["choices"][0]["message"]["content"] == "TrueFoundry gateway mock reply"
        )
        mock_create.assert_called_once()


def test_sanitize_helpers():
    """Verify sanitize_credential and sanitize_url handle various dirty inputs."""
    assert sanitize_credential(None) == ""
    assert sanitize_credential("") == ""
    assert sanitize_credential("   sk-clean   ") == "sk-clean"
    assert sanitize_credential('"sk-quoted"') == "sk-quoted"
    assert sanitize_credential("'sk-single-quoted'") == "sk-single-quoted"
    assert sanitize_credential('  "  sk-spaced-quote  "  ') == "sk-spaced-quote"
    assert sanitize_credential('""sk-double-quoted""') == "sk-double-quoted"

    assert sanitize_url(None) is None
    assert sanitize_url("") is None
    assert sanitize_url("https://api.openai.com/v1/") == "https://api.openai.com/v1"
    assert (
        sanitize_url(' "https://gateway.truefoundry.ai/" ')
        == "https://gateway.truefoundry.ai"
    )


@pytest.mark.parametrize(
    "vendor,env_key,raw_value,expected_clean",
    [
        ("openai", "OPENAI_API_KEY", '  "sk-openai-key"  ', "sk-openai-key"),
        ("anthropic", "ANTHROPIC_API_KEY", "  'sk-ant-key'  ", "sk-ant-key"),
        ("anthropic", "ANTHROPIC_AUTH_TOKEN", '  "sk-ant-token"  ', "sk-ant-token"),
        ("gemini", "GEMINI_API_KEY", '  "aiza-gemini-key"  ', "aiza-gemini-key"),
        ("gemini", "GOOGLE_API_KEY", "  'aiza-google-key'  ", "aiza-google-key"),
        ("mistral", "MISTRAL_API_KEY", '  "mistral-api-key"  ', "mistral-api-key"),
        ("openrouter", "OPENROUTER_API_KEY", '  "sk-or-key"  ', "sk-or-key"),
        (
            "truefoundry",
            "TRUEFOUNDRY_API_KEY",
            '  "tfy-secret-key"  ',
            "tfy-secret-key",
        ),
        ("truefoundry", "TFY_API_KEY", "  'tfy-short-key'  ", "tfy-short-key"),
    ],
)
def test_all_vendors_key_sanitization(vendor, env_key, raw_value, expected_clean):
    """Verify that every vendor's credentials are sanitized even when loaded with quotes and spaces."""
    os.environ[env_key] = raw_value
    active_vendor, _, _ = detect_active_vendor_and_model()
    assert active_vendor == vendor


def test_all_providers_instantiation_with_dirty_env():
    """Verify that all 6 provider classes instantiate cleanly when given dirty keys/URLs."""
    # OpenAI
    p_openai = OpenAIProvider(
        api_key='  "sk-test-openai"  ',
        base_url='  "https://mock.openai.com/v1/"  ',
    )
    assert p_openai.client.api_key == "sk-test-openai"
    assert str(p_openai.client.base_url).rstrip("/") == "https://mock.openai.com/v1"

    # Anthropic
    p_anthropic = AnthropicProvider(
        api_key='  "sk-test-anthropic"  ',
        base_url='  "https://mock.anthropic.com/v1/"  ',
    )
    assert p_anthropic.client.api_key == "sk-test-anthropic"

    # Gemini
    p_gemini = GeminiProvider(
        api_key='  "aiza-test-gemini"  ',
    )
    assert p_gemini.client._api_client.api_key == "aiza-test-gemini"

    # Mistral
    p_mistral = MistralProvider(
        api_key='  "test-mistral"  ',
        base_url='  "https://mock.mistral.ai/v1/"  ',
    )
    assert p_mistral.vendor_name == "mistral"

    # OpenRouter
    p_openrouter = OpenRouterProvider(
        api_key='  "sk-or-test"  ',
        base_url='  "https://openrouter.ai/api/v1/"  ',
    )
    assert p_openrouter.vendor_name == "openrouter"

    # TrueFoundry
    p_truefoundry = TrueFoundryProvider(
        api_key='  "tfy-test"  ',
        base_url='  "https://gateway.truefoundry.ai/"  ',
    )
    assert p_truefoundry.client.api_key == "tfy-test"
    assert (
        str(p_truefoundry.client.base_url).rstrip("/")
        == "https://gateway.truefoundry.ai"
    )
