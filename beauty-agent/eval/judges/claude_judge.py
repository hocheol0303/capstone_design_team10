"""Claude (Anthropic) judge wrapper.

Uses claude-sonnet-4-6 to avoid self-preference bias (agent uses gpt-4o-mini).
Forces JSON output via system prompt; one retry on parse failure.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

import anthropic

JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "claude-sonnet-4-6")
_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set — required for judge.")
        _client = anthropic.Anthropic()
    return _client


def _extract_json(text: str) -> dict | list:
    """Pull the first JSON object/array out of `text`. Raises on failure."""
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    return json.loads(text)


def score(
    system_prompt: str,
    user_payload: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Send a single judge call. Returns {"parsed": <json>, "raw": <text>, "error": <str|None>}."""
    client = _get_client()
    msg = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_payload}],
    )
    raw = "".join(blk.text for blk in msg.content if getattr(blk, "type", None) == "text")
    try:
        return {"parsed": _extract_json(raw), "raw": raw, "error": None}
    except Exception as e:
        # One retry with explicit JSON-only nudge
        msg2 = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=max_tokens,
            temperature=0.0,
            system=system_prompt + "\n\nIMPORTANT: respond ONLY with a single JSON value, no prose, no markdown.",
            messages=[{"role": "user", "content": user_payload}],
        )
        raw2 = "".join(blk.text for blk in msg2.content if getattr(blk, "type", None) == "text")
        try:
            return {"parsed": _extract_json(raw2), "raw": raw2, "error": None}
        except Exception as e2:
            return {"parsed": None, "raw": raw + "\n\n--- RETRY ---\n" + raw2, "error": f"{e}; retry: {e2}"}
