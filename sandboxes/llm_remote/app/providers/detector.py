"""Vendor detection and configuration management for Remote LLM Sandbox."""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv

    # Load from current working directory or sandbox root
    load_dotenv()
    _sandbox_env_file = Path(__file__).resolve().parent.parent.parent / ".env"
    if _sandbox_env_file.is_file():
        load_dotenv(dotenv_path=_sandbox_env_file)
except ImportError:
    pass


def _normalize_environ() -> None:
    """Normalize environment variables by stripping leading/trailing whitespace and surrounding quotes."""
    for k, v in list(os.environ.items()):
        clean_k = k.strip()
        clean_v = v.strip()
        if (clean_v.startswith('"') and clean_v.endswith('"')) or (
            clean_v.startswith("'") and clean_v.endswith("'")
        ):
            clean_v = clean_v[1:-1].strip()
        if clean_k != k:
            os.environ.pop(k, None)
            os.environ[clean_k] = clean_v
        elif clean_v != v:
            os.environ[k] = clean_v


_normalize_environ()

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

# Mapping of vendor name to potential environment variable names for API keys
VENDOR_KEY_MAPPING: Dict[str, List[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "truefoundry": ["TRUEFOUNDRY_API_KEY", "TFY_API_KEY"],
}

# Precedence order when multiple vendor keys are detected and no vendor is explicitly pinned
DEFAULT_VENDOR_PRECEDENCE: List[str] = [
    "openai",
    "anthropic",
    "gemini",
    "mistral",
    "openrouter",
    "truefoundry",
]


class VendorConfigurationError(Exception):
    """Raised when vendor configuration or credentials cannot be resolved."""

    pass


def load_model_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from config/model.toml."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "model.toml"

    if not config_path.exists():
        logger.warning(
            f"Config file not found at {config_path}, using internal defaults."
        )
        return {}

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def get_detected_vendor_keys() -> Dict[str, str]:
    """Scan os.environ for known vendor API keys.

    Returns:
        Dict[str, str]: Mapping of vendor_name -> detected env_var_name.
    """
    _normalize_environ()
    detected: Dict[str, str] = {}
    for vendor, env_vars in VENDOR_KEY_MAPPING.items():
        for var_name in env_vars:
            val = os.getenv(var_name)
            if val and val.strip():
                detected[vendor] = var_name
                break
    return detected


def resolve_vendor_credential(vendor: str) -> str:
    """Retrieve API key for specified vendor or raise an error."""
    _normalize_environ()
    env_vars = VENDOR_KEY_MAPPING.get(vendor, [])
    for var in env_vars:
        val = os.getenv(var)
        if val and val.strip():
            return val.strip().strip("'\"")

    vars_str = " or ".join(env_vars)
    raise VendorConfigurationError(
        f"Vendor '{vendor}' was selected, but no API key was found in environment. "
        f"Please set {vars_str}."
    )


def detect_active_vendor_and_model(
    config_path: Optional[Path] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """Resolve active vendor, model name, and vendor-specific configuration.

    Resolution strategy:
    1. Explicit Pinning (highest priority):
       - If environment variable LLM_VENDOR is set (and != 'auto'), use it.
       - Otherwise, if config/model.toml has [default].vendor != 'auto', use it.
    2. Auto-Detection (when vendor is 'auto'):
       - Scan os.environ for present API keys.
       - If 1 vendor key is present: auto-activate that vendor.
       - If >1 vendor keys are present: select highest-priority vendor in DEFAULT_VENDOR_PRECEDENCE
         and log an informative warning with remediation instructions.
       - If 0 vendor keys are present: raise VendorConfigurationError listing all supported keys.
    3. Model Resolution:
       - Environment variable LLM_MODEL takes highest priority.
       - Next, [default].model if set.
       - Next, vendor-specific model from [vendor].model in model.toml.
       - Default fallback if none found.

    Returns:
        Tuple[str, str, Dict[str, Any]]: (vendor_name, model_name, vendor_section_config)
    """
    config = load_model_config(config_path)
    default_section = config.get("default", {})

    # 1. Determine vendor selection
    env_vendor = os.getenv("LLM_VENDOR", "").strip().lower()
    config_vendor = default_section.get("vendor", "auto").strip().lower()

    detected_keys = get_detected_vendor_keys()
    selected_vendor: Optional[str] = None

    if env_vendor and env_vendor != "auto":
        selected_vendor = env_vendor
    elif config_vendor and config_vendor != "auto":
        selected_vendor = config_vendor
    else:
        # Auto-detect mode
        if not detected_keys:
            all_keys = [k for keys in VENDOR_KEY_MAPPING.values() for k in keys]
            raise VendorConfigurationError(
                "No LLM vendor API keys were detected in the environment!\n"
                f"Please export one of the supported keys: {', '.join(all_keys)}\n"
                "Or explicitly pin the vendor using LLM_VENDOR=<vendor>."
            )
        elif len(detected_keys) == 1:
            selected_vendor = list(detected_keys.keys())[0]
            logger.info(
                f"Auto-detected single vendor API key ({detected_keys[selected_vendor]}). "
                f"Activating vendor: '{selected_vendor}'."
            )
        else:
            # Multiple keys detected: apply precedence
            for candidate in DEFAULT_VENDOR_PRECEDENCE:
                if candidate in detected_keys:
                    selected_vendor = candidate
                    break
            if not selected_vendor:
                selected_vendor = list(detected_keys.keys())[0]

            logger.warning(
                f"Multiple vendor API keys detected: {list(detected_keys.keys())}. "
                f"Defaulting to '{selected_vendor}' based on precedence order.\n"
                f"To switch to a different provider, set LLM_VENDOR=<vendor> or edit config/model.toml."
            )

    if selected_vendor not in VENDOR_KEY_MAPPING:
        raise VendorConfigurationError(
            f"Unsupported vendor '{selected_vendor}'. Supported vendors: {list(VENDOR_KEY_MAPPING.keys())}"
        )

    # Validate credential presence
    resolve_vendor_credential(selected_vendor)

    # 2. Determine model selection
    vendor_section = config.get(selected_vendor, {})
    env_model = os.getenv("LLM_MODEL", "").strip()
    config_default_model = default_section.get("model", "").strip()
    vendor_default_model = vendor_section.get("model", "").strip()

    if env_model:
        selected_model = env_model
    elif config_default_model:
        selected_model = config_default_model
    elif vendor_default_model:
        selected_model = vendor_default_model
    else:
        # Fallback defaults
        fallbacks = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-5-haiku-latest",
            "gemini": "gemini-3.6-flash",
            "mistral": "mistral-small-latest",
            "openrouter": "meta-llama/llama-3.3-70b-instruct",
            "truefoundry": "openai/gpt-4o-mini",
        }
        selected_model = fallbacks.get(selected_vendor, "gpt-4o-mini")

    return selected_vendor, selected_model, vendor_section
