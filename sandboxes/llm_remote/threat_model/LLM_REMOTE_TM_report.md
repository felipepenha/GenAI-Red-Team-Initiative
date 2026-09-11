# Threat Model Report: LLM Remote Sandbox

## 1. Overview & System Description

The **LLM Remote Sandbox** provides a standardized, OpenAI-compatible API gateway designed for Red Teaming remote, production Large Language Model (LLM) APIs. Unlike local sandbox environments (such as `sandboxes/llm_local`, which execute self-hosted models via Ollama), this sandbox bridges local security tooling and test harnesses with remote commercial LLM providers:
- **OpenAI** (via `openai` SDK)
- **Anthropic** (via `anthropic` SDK)
- **Google Gemini** (via `google-genai` SDK)
- **Mistral AI** (via `mistralai` SDK)
- **OpenRouter** (via `openai` SDK)
- **TrueFoundry AI Gateway** (via `openai` SDK)

The architecture is composed of:
1. **Client Tier**: Test clients (`client/main.py`), interactive Gradio UI (`client/gradio_app.py`), or external Red Teaming suites (e.g., PyRIT, Garak, Promptfoo).
2. **Gateway Tier**: FastAPI container running locally on `http://localhost:8000`, exposing standard `/v1/chat/completions` and `/v1/models` endpoints.
3. **Provider Integration Tier**: Vendor-specific modules translating OpenAI-compatible requests and invoking native vendor SDKs without intermediary orchestration frameworks.
4. **External Services Tier**: Public or VPC-hosted remote LLM endpoints over HTTPS.

---

## 2. Remote LLM Specific Risks & Threat Matrix

| Threat ID | Threat Category | Description | Severity | Mitigation in Sandbox |
|---|---|---|---|---|
| **TR-01** | **Denial of Wallet / Financial Exhaustion** | High-volume fuzzing, recursive prompt injections, or automated payload generators rapidly exhaust API credits or trigger exorbitant cloud bills. | **High** | Token limits (`max_tokens`), model selection controls in `config/model.toml`, and clear warnings before test execution. |
| **TR-02** | **Account Suspension & Guardrail Ban** | Probing for extreme harmful content or jailbreaks triggers remote provider Trust & Safety classifiers, leading to API key revocation or account termination. | **High** | Explicit red-teaming warnings in documentation; recommendation to use dedicated red-teaming test accounts and pre-cleared red-teaming allowances. |
| **TR-03** | **Sensitive Data Leakage via Remote Logs** | Proprietary corporate data, secret keys, or vulnerability details embedded in test prompts are transmitted to remote servers and logged according to provider retention policies. | **Medium** | Pre-prompt sanitization and documentation advising against using production confidential data in remote tests. |
| **TR-04** | **Credential Exposure & Leakage** | API keys (`OPENAI_API_KEY`, etc.) passed into container environments or logged during debug/trace output. | **High** | API keys are read directly from memory/environment; keys are masked and never echoed in `/health` or error traces; container runs as non-root user. |
| **TR-05** | **Rate Limiting & Throttling** | Cloud providers enforce Requests Per Minute (RPM) and Tokens Per Minute (TPM) limits, returning HTTP 429 and aborting security assessment campaigns. | **Medium** | Retry handling (`tenacity`), modular provider timeout configurations, and structured test reporting. |
| **TR-06** | **Non-Deterministic Behavior & Drift** | Remote provider updates, silent model revisions, or system-prompt injections from the cloud side cause inconsistent red-teaming findings. | **Low** | Pinned model versions in `config/model.toml` (e.g. specific model snapshot tags). |

---

## 3. STRIDE Analysis

### Spoofing
- **Threat**: Unauthorized clients submit prompts to the local gateway on port 8000.
- **Mitigation**: API key verification (`verify_api_key`) enforcing Bearer token authentication against configured secrets (`SANDBOX_API_KEY`).

### Tampering
- **Threat**: Manipulation of requests or models in transit between client and gateway.
- **Mitigation**: Communication over localhost network; container isolation with dedicated bridge network (`sec_remote_test_net`); TLS/HTTPS enforced for all outbound calls to remote cloud endpoints.

### Repudiation
- **Threat**: Actions taken against remote LLMs cannot be attributed to specific test suites.
- **Mitigation**: Gateway logging captures timestamps, requested model, latency, and status code for all transactions.

### Information Disclosure
- **Threat**: Vendor API keys or remote provider response headers leaked in error responses.
- **Mitigation**: Exception handling sanitizes error outputs to prevent exposing raw authorization headers or bearer credentials to the caller.

### Denial of Service
- **Threat**: Exhaustion of local gateway resources or remote provider rate limits.
- **Mitigation**: Non-root container execution, bounded timeouts on outbound requests (120s max), and clear HTTP 502/504 responses when remote providers fail.

### Elevation of Privilege
- **Threat**: Arbitrary code execution within the container via dependency vulnerabilities.
- **Mitigation**: Non-root container execution (`USER appuser`), minimal base image (`python:3.12-slim`), strict dependency pinning (`uv.lock`), and no shell execution of prompt inputs.

---

## 4. Best Practices for Red Teaming Remote LLMs

1. **Use Dedicated Red Teaming API Keys**: Never use production service credentials for adversarial testing.
2. **Apply Spend Limits**: Configure hard spending limits in provider dashboards (OpenAI, Anthropic, Google Cloud, Mistral) before launching automated scans.
3. **Review Provider Terms of Service**: When evaluating compliance or safety boundaries, check if your provider requires prior notification for penetration testing or red teaming activities.
4. **Monitor Throttling**: Use `config/prompts.toml` to baseline rate limits and latency before launching distributed or parallel fuzzing runs.
