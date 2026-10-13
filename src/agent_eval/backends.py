"""Optional real-LLM backend (Anthropic API).

Requires the ``anthropic`` extra and ``ANTHROPIC_API_KEY`` in the environment::

    pip install "agent-eval-harness[anthropic]"
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from .agent import SYSTEM_PROMPT, Tool


class AnthropicBackend:
    """Tool-calling backend using the Anthropic Messages API."""

    name = "anthropic"

    def __init__(self, model: str, max_tokens: int = 1024) -> None:
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                'pip install "agent-eval-harness[anthropic]" to use AnthropicBackend'
            ) from exc
        self.client = anthropic.Anthropic()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.model = model
        self.max_tokens = max_tokens

    @staticmethod
    def _to_anthropic_tools(tools: List[Tool]) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        k: {"type": "string", "description": v}
                        for k, v in t.parameters.items()
                    },
                },
            }
            for t in tools
        ]

    def complete(
        self, messages: List[Dict[str, Any]], tools: List[Tool]
    ) -> Dict[str, Any]:
        # The harness uses a system+user message list; the API takes system separately.
        system = next(
            (m["content"] for m in messages if m["role"] == "system"), SYSTEM_PROMPT
        )
        convo = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m["role"] in ("user", "assistant")
        ]
        # Ask for a strict JSON envelope so parsing stays trivial.
        convo = convo + [
            {
                "role": "user",
                "content": (
                    'Reply with ONLY one JSON object: either {"tool": "<name>", '
                    '"arguments": {...}} or {"answer": "<text>"}.'
                ),
            }
        ]
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=convo,
        )
        text = "".join(
            b.text for b in resp.content if getattr(b, "type", "") == "text"
        ).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {"type": "final", "text": text}
        if "answer" in data:
            return {"type": "final", "text": str(data["answer"])}
        if "tool" in data:
            return {
                "type": "tool_call",
                "tool_call": {
                    "name": data["tool"],
                    "arguments": data.get("arguments", {}),
                },
            }
        return {"type": "final", "text": text}
