"""FastAPI main application module for LLM Remote Sandbox.

This module initializes the FastAPI application and mounts remote LLM provider routers.
It serves as the entry point for the LLM Remote API Server.
"""

from typing import Any, Dict

from fastapi import FastAPI

from app.providers import router
from app.providers.detector import (
    VendorConfigurationError,
    detect_active_vendor_and_model,
)

app = FastAPI(
    title="LLM Remote Sandbox API Server",
    description="Remote LLM API gateway for Red Teaming multi-vendor LLM providers",
    version="1.0.0",
)


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint to verify server status and active provider configuration.

    Returns:
        Dict[str, Any]: Status dictionary including 'status', 'vendor', and 'model'.
    """
    try:
        vendor, model, _ = detect_active_vendor_and_model()
        return {
            "status": "ok",
            "vendor": vendor,
            "model": model,
        }
    except VendorConfigurationError as e:
        return {
            "status": "warning",
            "message": "Server running but no vendor API key configured",
            "error": str(e),
        }


# Mount remote LLM provider router
app.include_router(router, tags=["Remote LLM Gateway"])
