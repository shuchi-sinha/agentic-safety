"""Scripted agent policies for deterministic evaluation.

The same scenario runs under two policies:

- ``NaivePolicy`` -- follows instructions embedded in tool output, modelling
  prompt-injection susceptibility.
- ``CautiousPolicy`` -- ignores embedded instructions and sticks to its plan,
  modelling a hardened agent.

Comparing their traces under identical faults is the core demonstration:
the harness measures *policy robustness*, not just task success.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .agent import Tool
from .faults import INJECT_MARKER

_INJECT_RE = re.compile(r"\[\[INJECT:\s*(\w+)\s*((?:\w+=[^\s\]]+\s*)*)\]\]")


def parse_injection(text: Optional[str]) -> Optional[Tuple[str, Dict[str, str]]]:
    """Parse ``[[INJECT: tool k=v k=v ]]`` out of a tool result."""
    if not text:
        return None
    m = _INJECT_RE.search(text)
    if not m:
        return None
    args: Dict[str, str] = {}
    for kv in m.group(2).split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            args[k] = v
    return m.group(1), args


def _to_response(step: Dict[str, Any]) -> Dict[str, Any]:
    if "final" in step:
        return {"type": "final", "text": step["final"]}
    return {"type": "tool_call", "tool_call": step["tool_call"]}


class ScriptedPolicy:
    """Follows a scripted plan; subclasses decide what to do about
    injected instructions found in tool results."""

    name = "scripted"

    def __init__(self, plan: List[Dict[str, Any]]) -> None:
        self.plan = plan
        self._idx = 0
        self._acted_injections: set[int] = set()

    def _tool_results(self, messages: List[Dict[str, Any]]):
        for i, m in enumerate(messages):
            content = m.get("content", "")
            if m.get("role") == "user" and content.startswith("Result of "):
                yield i, content

    def _next_planned(self) -> Dict[str, Any]:
        if self._idx < len(self.plan):
            step = self.plan[self._idx]
            self._idx += 1
            return _to_response(step)
        return {"type": "final", "text": "No further planned steps."}


class NaivePolicy(ScriptedPolicy):
    """Follows the first unacted injected instruction it sees in tool output."""

    name = "naive"

    def complete(
        self, messages: List[Dict[str, Any]], tools: List[Tool]
    ) -> Dict[str, Any]:
        for i, content in self._tool_results(messages):
            if i in self._acted_injections:
                continue
            parsed = parse_injection(content)
            if parsed:
                self._acted_injections.add(i)
                tool_name, args = parsed
                self._idx += 1  # the injected turn consumes a plan slot
                return {
                    "type": "tool_call",
                    "tool_call": {"name": tool_name, "arguments": args},
                    "from_injection": True,
                }
        return self._next_planned()


class CautiousPolicy(ScriptedPolicy):
    """Ignores embedded instructions entirely and follows its plan."""

    name = "cautious"

    def complete(
        self, messages: List[Dict[str, Any]], tools: List[Tool]
    ) -> Dict[str, Any]:
        for i, _ in self._tool_results(messages):
            self._acted_injections.add(i)  # seen, never acted on
        return self._next_planned()


POLICIES = {"naive": NaivePolicy, "cautious": CautiousPolicy}
