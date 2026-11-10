"""Token-budget tracking, pruning and compaction.

Budget tracking anchors on the provider's reported usage (exact) and only
estimates the messages appended since (chars/4), so drift stays small.

Two stages keep a session under budget:

  prune    (at ``prune_at`` of budget): shorten old, large tool results.
  compact  (at ``compact_at`` of budget): summarize everything before a cut
           point into one message and keep the recent tail verbatim.

The history is append-only otherwise. That matters for providers whose
reasoning blocks are bound to the exact prefix that produced them (Claude's
thinking blocks are): every rewrite strips ``ProviderBlock(reasoning=True)``
from the first edited message onward, and blocks before the edit stay. The
cut point always lands on an assistant message, so a tool call is never
separated from its result.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

from .conversation import (
    Message,
    ModelRequest,
    ProviderBlock,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    Usage,
    render_block,
)
from .providers.base import Provider
from .session import Session

log = logging.getLogger(__name__)

SUMMARY_PREFIX = "[Summary of the earlier conversation, written by the harness to save context]\n\n"


class ContextOverflow(Exception):
    pass


def estimate_tokens(text: str) -> int:
    return len(text) // 4 + 1


def estimate_message(m: Message) -> int:
    total = 4
    for b in m.content:
        if isinstance(b, ProviderBlock):
            total += estimate_tokens(json.dumps(b.data))
        else:
            total += estimate_tokens(render_block(b, max_chars=10**9)) + 4
    return total


def estimate_messages(messages: List[Message]) -> int:
    return sum(estimate_message(m) for m in messages)


@dataclass
class ContextConfig:
    max_context_tokens: Optional[int] = None  # None: the provider's window
    reserve_tokens: int = 32_000  # room for the model's reply
    prune_at: float = 0.6
    compact_at: float = 0.8
    keep_recent_tokens: int = 20_000  # tail kept verbatim on compaction
    protect_recent_messages: int = 6  # never pruned
    pruned_result_chars: int = 400


Summarizer = Callable[[List[Message]], str]


class LLMSummarizer:
    SYSTEM = (
        "You compress an AI agent's working transcript so it can continue the task with less context. "
        "Keep: the user's goals and constraints (quote requests verbatim), decisions made, files created or "
        "changed and why, facts discovered, errors hit and how they were resolved, and what remains to do. "
        "Drop: raw tool output that is no longer needed. Write plain prose and bullet lists, no preamble."
    )

    def __init__(self, provider: Provider, max_tokens: int = 8_000):
        self.provider = provider
        self.max_tokens = max_tokens

    def __call__(self, messages: List[Message]) -> str:
        transcript = "\n\n".join(
            "%s:\n%s" % (m.role.upper(), "\n".join(filter(None, (render_block(b) for b in m.content))))
            for m in messages
        )
        resp = self.provider.complete(
            ModelRequest(
                system=self.SYSTEM,
                messages=[Message.user("Summarize this transcript:\n\n" + transcript)],
                max_tokens=self.max_tokens,
            )
        )
        if not resp.text.strip():
            raise RuntimeError("summarizer returned no text (stop reason %s)" % resp.stop_reason.value)
        return resp.text


def mechanical_summary(messages: List[Message]) -> str:
    """Fallback when the summarizer fails: keep user text and tool-call names."""
    lines = []
    for m in messages:
        for b in m.content:
            if isinstance(b, TextBlock) and m.role == "user":
                lines.append("- user: " + b.text[:1000])
            elif not isinstance(b, (TextBlock, ToolResultBlock, ProviderBlock)):
                lines.append("- " + render_block(b, max_chars=200))
    return "Earlier steps (automatic outline):\n" + "\n".join(lines[-200:])


class ContextManager:
    def __init__(self, provider: Provider, config: Optional[ContextConfig] = None, summarizer: Optional[Summarizer] = None):
        self.config = config or ContextConfig()
        self.budget = self.config.max_context_tokens or provider.context_window
        self.summarizer = summarizer or LLMSummarizer(provider)
        self._anchor_tokens: Optional[int] = None
        self._anchor_len = 0

    # ------------------------------------------------------------ tracking

    def record_usage(self, session: Session, usage: Usage) -> None:
        """Call right after appending the assistant message the usage is for."""
        session.total_input_tokens += usage.input_tokens
        session.total_output_tokens += usage.output_tokens
        if usage.input_tokens:
            self._anchor_tokens = usage.input_tokens + usage.output_tokens
            self._anchor_len = len(session.messages)

    def estimate(self, session: Session, tools: List[ToolSpec]) -> int:
        if self._anchor_tokens is not None and self._anchor_len <= len(session.messages):
            return self._anchor_tokens + estimate_messages(session.messages[self._anchor_len:])
        fixed = estimate_tokens(session.system) + sum(
            estimate_tokens(t.name + t.description + json.dumps(t.parameters)) for t in tools
        )
        return fixed + estimate_messages(session.messages)

    def _invalidate(self) -> None:
        self._anchor_tokens = None

    # ---------------------------------------------------------- management

    def ensure_fits(self, session: Session, tools: List[ToolSpec]) -> List[str]:
        """Prune/compact as needed before a model call. Returns what was done."""
        cfg, actions = self.config, []
        used = self.estimate(session, tools)
        if used >= cfg.prune_at * self.budget:
            n = self.prune(session)
            if n:
                actions.append("pruned %d old tool results" % n)
                used = self.estimate(session, tools)
        if used >= cfg.compact_at * self.budget:
            self.compact(session)
            actions.append("compacted history (%d -> %d est. tokens)" % (used, self.estimate(session, tools)))
            used = self.estimate(session, tools)
        if used > self.budget - cfg.reserve_tokens:
            raise ContextOverflow("context needs ~%d tokens; budget is %d" % (used, self.budget))
        return actions

    def prune(self, session: Session) -> int:
        cfg = self.config
        limit = max(0, len(session.messages) - cfg.protect_recent_messages)
        first_edit, count = None, 0
        for i, m in enumerate(session.messages[:limit]):
            for j, b in enumerate(m.content):
                if isinstance(b, ToolResultBlock) and len(b.content) > cfg.pruned_result_chars * 2:
                    kept = b.content[: cfg.pruned_result_chars]
                    note = "\n...[pruned %d chars to save context; re-run the tool if needed]" % (len(b.content) - len(kept))
                    m.content[j] = ToolResultBlock(b.tool_call_id, kept + note, b.is_error)
                    first_edit = i if first_edit is None else first_edit
                    count += 1
        if first_edit is not None:
            strip_reasoning(session.messages[first_edit:])
            self._invalidate()
        return count

    def compact(self, session: Session) -> None:
        cut = self._cut_index(session.messages)
        if cut is None:
            raise ContextOverflow("nothing to compact: no completed assistant turn yet")
        old, recent = session.messages[:cut], session.messages[cut:]
        try:
            summary = self.summarizer(old)
        except Exception as e:  # never lose the session to a summarizer failure
            log.warning("summarizer failed (%s); using mechanical summary", e)
            summary = mechanical_summary(old)
        strip_reasoning(recent)
        session.messages = [Message("user", [TextBlock(SUMMARY_PREFIX + summary)])] + recent
        session.compactions += 1
        self._invalidate()

    def _cut_index(self, messages: List[Message]) -> Optional[int]:
        assistant_idx = [i for i, m in enumerate(messages) if m.role == "assistant" and i > 0]
        if not assistant_idx:
            return None
        for i in assistant_idx:  # earliest cut whose tail fits the keep budget
            if estimate_messages(messages[i:]) <= self.config.keep_recent_tokens:
                return i
        return assistant_idx[-1]


def strip_reasoning(messages: List[Message]) -> None:
    for m in messages:
        kept = [b for b in m.content if not (isinstance(b, ProviderBlock) and b.reasoning)]
        # A turn that was only reasoning must not vanish (roles would stop alternating).
        m.content = kept or [TextBlock("(reasoning omitted)")]
