"""Automated test runner for LLM Remote Sandbox.

This module loads test prompts from config/prompts.toml and runs automated tests
against the sandbox API gateway, displaying results and summary statistics.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
    _env_f = Path(__file__).resolve().parent.parent / ".env"
    if _env_f.is_file():
        load_dotenv(dotenv_path=_env_f)
except ImportError:
    pass

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

# Configuration paths
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"

# Load test prompts
prompts_path = CONFIG_DIR / "prompts.toml"
with open(prompts_path, "rb") as f:
    prompts_config = tomllib.load(f)

# Load client configuration
client_config_path = CONFIG_DIR / "client_config.toml"
with open(client_config_path, "rb") as f:
    client_config = tomllib.load(f)

# Load model configuration
model_config_path = CONFIG_DIR / "model.toml"
with open(model_config_path, "rb") as f:
    model_config = tomllib.load(f)

API_BASE_URL = os.getenv("SANDBOX_API_URL", "http://localhost:8000")
API_KEY = os.getenv("SANDBOX_API_KEY", "sk-mock-key")


def test_prompt(
    prompt: str,
    category: str = "test",
    model: str = "",
) -> Dict[str, Any]:
    """Send a prompt through the remote sandbox API and return test results.

    Args:
        prompt: User message string.
        category: Test category name.
        model: Optional model identifier.

    Returns:
        Dict[str, Any]: Result dictionary with status, latency, response, and error.
    """
    pre_prompt = client_config.get("client", {}).get("pre_prompt", "")
    messages: List[Dict[str, str]] = []
    if pre_prompt:
        messages.append({"role": "system", "content": pre_prompt})
    messages.append({"role": "user", "content": prompt})

    payload: Dict[str, Any] = {
        "messages": messages,
    }
    if model:
        payload["model"] = model

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    start_time = time.time()
    try:
        resp = requests.post(
            f"{API_BASE_URL}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=120,
        )
        elapsed = round(time.time() - start_time, 2)

        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices", [])
            content = (
                choices[0].get("message", {}).get("content", "") if choices else ""
            )
            usage = data.get("usage", {})
            return {
                "category": category,
                "prompt": prompt,
                "success": True,
                "response": content,
                "latency_sec": elapsed,
                "usage": usage,
                "error": None,
            }
        else:
            return {
                "category": category,
                "prompt": prompt,
                "success": False,
                "response": None,
                "latency_sec": elapsed,
                "error": f"HTTP {resp.status_code}: {resp.text}",
            }
    except Exception as e:
        elapsed = round(time.time() - start_time, 2)
        return {
            "category": category,
            "prompt": prompt,
            "success": False,
            "response": None,
            "latency_sec": elapsed,
            "error": str(e),
        }


if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing LLM Remote Sandbox with Configured Prompts")
    print(f"Target API Endpoint: {API_BASE_URL}")
    print("=" * 80)

    # Check server health first
    try:
        health_resp = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            print(
                f"✅ Connected to Sandbox! Active Vendor: {health_data.get('vendor', 'unknown')}, "
                f"Model: {health_data.get('model', 'unknown')}"
            )
        else:
            print(
                f"⚠️ Health check returned HTTP {health_resp.status_code}: {health_resp.text}"
            )
    except Exception as e:
        print(f"❌ Failed to connect to sandbox at {API_BASE_URL}: {e}")
        print(
            "Please ensure the server is running (e.g. 'make up' or 'uvicorn app.main:app')."
        )
        sys.exit(1)

    all_results: List[Dict[str, Any]] = []
    total_tests: int = 0
    passed_tests: int = 0

    for category, prompts in prompts_config.get("test_prompts", {}).items():
        if not prompts:
            continue

        print(f"\n📋 Testing category: {category.upper()}")
        print("-" * 80)

        for i, prompt in enumerate(prompts, 1):
            total_tests += 1
            truncated_prompt = prompt[:60] + "..." if len(prompt) > 60 else prompt
            print(f"\n[{i}/{len(prompts)}] Prompt: {truncated_prompt}")

            result = test_prompt(prompt, category)
            all_results.append(result)

            if result["success"]:
                passed_tests += 1
                resp_text = result["response"] or ""
                truncated_resp = (
                    resp_text[:100] + "..." if len(resp_text) > 100 else resp_text
                )
                print(f"✅ Success ({result['latency_sec']}s): {truncated_resp}")
            else:
                print(f"❌ Failed ({result['latency_sec']}s): {result['error']}")

    print("\n" + "=" * 80)
    print("📊 Test Summary")
    print("=" * 80)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests} ✅")
    print(f"Failed: {total_tests - passed_tests} ❌")
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    print("=" * 80)

    sys.exit(0 if passed_tests == total_tests else 1)
