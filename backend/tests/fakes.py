"""A scripted stand-in for anthropic.Anthropic used by the agent-loop tests.

Only the model is faked: the loop, the tools and the database are real.
"""
from __future__ import annotations

import itertools
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

_ids = itertools.count(1)


def tool_use(name: str, **tool_input: Any) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=f"toolu_{next(_ids):04d}", name=name, input=tool_input)


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def response(*blocks: SimpleNamespace, stop_reason: str | None = None) -> SimpleNamespace:
    if stop_reason is None:
        stop_reason = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason)


class ScriptedClient:
    """`script` is a list of responses, or a callable(call_index, params) -> response."""

    def __init__(self, script: list[Any] | Callable[[int, dict[str, Any]], Any]):
        self.script = script
        self.requests: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **params: Any) -> Any:
        # Snapshot the conversation as it was sent (the loop mutates the list afterwards).
        self.requests.append({**params, "messages": [dict(m) for m in params["messages"]]})
        i = len(self.requests) - 1
        item = self.script(i, params) if callable(self.script) else self.script[i]
        if isinstance(item, Exception):
            raise item
        return item
