"""Independent re-derivation probe: ask the model, with tools off, to write the tool argument itself."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional

from . import analysis

_SYSTEM = ("You are re-deriving the next action of a coding agent from its conversation. "
           "Reply with ONLY the exact {field} the agent should pass to its `{tool}` tool next. "
           "No explanation, no code fences.")
_MAX_CONTEXT = 8000


def _flatten(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def context_transcript(messages: list[dict[str, Any]]) -> str:
    """Last user request plus everything after it, as plain text, including (truncated) tool output.

    Tool output matters: after a failed command the agent's next step depends on the error it saw.
    """
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=0)
    lines = []
    for m in messages[last_user:]:
        role, text = m.get("role"), _flatten(m.get("content"))
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                text += f"\n[called {fn.get('name')} {str(fn.get('arguments', ''))[:400]}]"
        if role == "tool":
            label, text = "TOOL OUTPUT", text.strip()[:1000]
        else:
            label, text = (role or "user").upper(), text.strip()[:2000]
        if text:
            lines.append(f"{label}: {text}")
    return "\n\n".join(lines)[-_MAX_CONTEXT:]


def build_messages(messages: list[dict[str, Any]], tool: str, field: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _SYSTEM.format(tool=tool, field=field)},
        {"role": "user", "content": context_transcript(messages) + f"\n\nWrite the exact {field} now."},
    ]


def probe(messages: list[dict[str, Any]], tool: str, field: str, *, base_url: str, api_key: str,
          model: str, timeout: float = 60.0) -> Optional[list[analysis.Token]]:
    """Return the probe answer's tokens, or None when the endpoint gives no logprobs."""
    body: dict[str, Any] = {
        "model": model, "messages": build_messages(messages, tool, field),
        "logprobs": True, "top_logprobs": 3, "max_tokens": 4000, "temperature": 0,
    }
    if "openrouter.ai" in base_url:
        body["provider"] = {"require_parameters": True}
        body["reasoning"] = {"effort": "low", "exclude": True}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    tokens = analysis.response_tokens(data)
    return tokens or None
