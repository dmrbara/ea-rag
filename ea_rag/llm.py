from __future__ import annotations

import json
import re
import time
import random
from typing import Dict, Any, List, Optional

import httpx

from .config import env


def _build_structured_response_format(prompt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build OpenRouter structured outputs block using json_schema.
    If retrieval candidates include target_uri, constrain via enum.
    Ref: https://openrouter.ai/docs/features/structured-outputs
    """
    candidate_uris: List[str] = []
    for c in prompt.get("candidates", []) or []:
        if isinstance(c, dict) and "target_uri" in c and isinstance(c.get("target_uri"), str):
            candidate_uris.append(c["target_uri"]) 
    # de-duplicate while preserving order
    seen: set[str] = set()
    candidate_uris = [u for u in candidate_uris if not (u in seen or seen.add(u))]

    target_uri_schema: Dict[str, Any] = {"type": "string", "description": "Chosen target entity URI"}
    if candidate_uris:
        target_uri_schema["enum"] = candidate_uris
    else:
        target_uri_schema["pattern"] = r"^http://local/entity/.+"

    schema = {
        "name": "entity_alignment",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "target_uri": target_uri_schema,
                "score": {"type": "number", "minimum": 0, "maximum": 1, "description": "Confidence [0,1]"},
                "rationale": {"type": "string", "description": "Short explanation"},
            },
            "required": ["target_uri", "score"],
            "additionalProperties": False,
        },
    }
    return {"type": "json_schema", "json_schema": schema}


def prompt_to_text(p: Dict[str, Any]) -> str:
    """
    Serialize the prompt dict used for alignment into a single user message string.
    This is shared across providers to ensure identical prompting and to allow
    saving the exact prompt text alongside predictions.
    """
    name = p.get("name") or p.get("source_uri", "")
    lines: List[str] = []
    lines.append("Task: Select the best aligned target for the source entity.")
    lines.append(f"Source: {p.get('source_uri','')} (name: {name})")
    src_sents = p.get("source_sentences") or []
    if isinstance(src_sents, list) and src_sents:
        lines.append("Source context:")
        for idx, s in enumerate(src_sents[:10], start=1):
            s_txt = (s or "").strip()
            if len(s_txt) > 400:
                s_txt = s_txt[:400]
            lines.append(f"- {idx}. {s_txt}")
    cands = p.get("candidates") or []
    if cands and isinstance(cands, list) and isinstance(cands[0], dict) and "target_uri" in cands[0]:
        lines.append("Candidates:")
        for idx, c in enumerate(cands[:10], start=1):
            uri = c.get("target_uri", "")
            ctx = (c.get("context") or "").strip()
            if len(ctx) > 400:
                ctx = ctx[:400]
            lines.append(f"- {idx}. {uri}\n  context: {ctx}")
    elif cands and isinstance(cands, list) and isinstance(cands[0], dict) and "target_name" in cands[0]:
        lines.append("Candidate names:")
        for idx, c in enumerate(cands[:10], start=1):
            lines.append(f"- {idx}. {c.get('target_name','')}")
    lines.append("Return strictly valid JSON matching the required schema.")
    return "\n".join(lines)


def predict_alignment_openrouter(
    model: str,
    prompt: Dict[str, Any],
    temperature: Optional[float] = None,
    timeout: float = 60.0,
    max_tokens: Optional[int] = 256,
    json_mode: bool = True,
    *,
    client: Optional[httpx.Client] = None,
    max_retries: int = 3,
    backoff_base: float = 0.75,
) -> Dict[str, Any]:
    """
    Call OpenRouter chat completions endpoint and expect JSON object output.
    The prompt dict should include fields to be serialized into a user message.
    Returns a dict with keys: target_uri, score, rationale (when parseable), possibly raw.
    """
    base_url = env.openrouter_base_url.rstrip("/")
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {env.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    messages = [
        {"role": "system", "content": "You are an entity alignment assistant. Return strictly valid JSON matching the required schema."},
        {"role": "user", "content": prompt_to_text(prompt)},
    ]
    body = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if json_mode:
        body["response_format"] = _build_structured_response_format(prompt)
    def _post_and_parse(payload: Dict[str, Any], _client: httpx.Client) -> Dict[str, Any]:
        r = _client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        # Try to parse message content as JSON
        provider = data.get("provider") or (data.get("choices") or [{}])[0].get("provider")
        try:
            content = data["choices"][0]["message"].get("content")
            if content:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "target_uri" in parsed:
                    # surface provider for diagnostics
                    parsed["provider"] = provider
                    return parsed
        except Exception:
            pass
        return {"provider": provider, "raw": data}

    def _call_with_fallbacks(_client: httpx.Client) -> Dict[str, Any]:
        # Attempt with strict json_schema; on 400 try json_object; then no response_format
        try:
            return _post_and_parse(body, _client)
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code == 400:
                # Fallback 1: generic json_object
                try:
                    body_json_obj = dict(body)
                    body_json_obj["response_format"] = {"type": "json_object"}
                    return _post_and_parse(body_json_obj, _client)
                except httpx.HTTPStatusError:
                    # Fallback 2: remove response_format entirely
                    try:
                        body_no_rf = dict(body)
                        body_no_rf.pop("response_format", None)
                        return _post_and_parse(body_no_rf, _client)
                    except httpx.HTTPStatusError as exc2:
                        # Last resort: return error payload
                        try:
                            err_json = exc2.response.json()
                        except Exception:
                            err_json = {"text": exc2.response.text if exc2.response is not None else str(exc2)}
                        err_provider = err_json.get("provider") if isinstance(err_json, dict) else None
                        return {"provider": err_provider, "error": f"HTTP {exc2.response.status_code if exc2.response else 'error'}", "raw": err_json}
            # Non-400: return error payload
            try:
                err_json = exc.response.json() if exc.response is not None else {"text": str(exc)}
            except Exception:
                err_json = {"text": exc.response.text if exc.response is not None else str(exc)}
            err_provider = err_json.get("provider") if isinstance(err_json, dict) else None
            return {"provider": err_provider, "error": f"HTTP {exc.response.status_code if exc.response else 'error'}", "raw": err_json}

    # Retry loop for transient errors (408, 429, 5xx) and network timeouts
    created_client: Optional[httpx.Client] = None
    _client_to_use = client
    if _client_to_use is None:
        created_client = httpx.Client(timeout=timeout, http2=True)
        _client_to_use = created_client
    try:
        last_result: Optional[Dict[str, Any]] = None
        for attempt in range(max_retries):
            try:
                result = _call_with_fallbacks(_client_to_use)  # type: ignore[arg-type]
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                # Normalize transport/timeout errors into a retryable shape
                result = {"error": "timeout", "raw": {"error": {"message": str(exc)}}}
            last_result = result
            err = result.get("error") if isinstance(result, dict) else None
            if not err:
                return result
            # Parse error to detect retryable statuses
            status_code: Optional[int] = None
            if isinstance(err, str):
                m = re.search(r"HTTP (\d+)", err)
                if m:
                    try:
                        status_code = int(m.group(1))
                    except Exception:
                        status_code = None
            # Retry on 408 Request Timeout, 429 Too Many Requests, and 5xx server errors
            if status_code is not None and (status_code in {408, 429} or 500 <= status_code <= 599):
                # Exponential backoff with jitter and a reasonable cap
                sleep_s = min(30.0, backoff_base * (2 ** attempt)) * (0.5 + random.random())
                time.sleep(sleep_s)
                continue
            # Also retry on generic timeout-like errors without explicit status
            if isinstance(err, str) and ("timeout" in err.lower() or "timed out" in err.lower()):
                sleep_s = min(30.0, backoff_base * (2 ** attempt)) * (0.5 + random.random())
                time.sleep(sleep_s)
                continue
            # Non-retryable error
            return result
        # Exhausted retries
        return last_result or {"error": "Unknown error"}
    finally:
        if created_client is not None:
            try:
                created_client.close()
            except Exception:
                pass
