"""Provider-neutral conversation types.

Every provider adapter translates to and from these types, so the loop, the
tool registry, the safety gate and the context manager never see a vendor SDK
object.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolCall:
    """A model's request to run a tool.

    ``parse_error`` is set when the provider could not decode the arguments
    (bad JSON, truncated stream). The registry turns that into an error result
    instead of running the tool.
    """

    id: str
    name: str
    arguments: Dict[str, Any]
    parse_error: Optional[str] = None
    type: str = "tool_call"


@dataclass
class ToolResultBlock:
    tool_call_id: str
    content: str
    is_error: bool = False
    type: str = "tool_result"


@dataclass
class ProviderBlock:
    """An opaque block only its own provider understands (thinking, fallback
    markers, citations...). It is replayed verbatim to the same provider and
    dropped for any other.

    ``reasoning`` marks blocks that are bound to the exact conversation prefix
    that produced them: any rewrite of earlier history must strip them.
    """

    provider: str
    data: Dict[str, Any]
    reasoning: bool = False
    type: str = "provider"


Block = Union[TextBlock, ToolCall, ToolResultBlock, ProviderBlock]


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: List[Block]

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls("user", [TextBlock(text)])

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_calls(self) -> List[ToolCall]:
        return [b for b in self.content if isinstance(b, ToolCall)]

    @property
    def tool_results(self) -> List[ToolResultBlock]:
        return [b for b in self.content if isinstance(b, ToolResultBlock)]


class StopReason(str, enum.Enum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    PAUSED = "paused"  # provider-side work paused; re-send to resume
    OTHER = "other"


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema (object)


@dataclass
class ModelRequest:
    system: str
    messages: List[Message]
    tools: List[ToolSpec] = field(default_factory=list)
    max_tokens: Optional[int] = None


@dataclass
class ModelResponse:
    content: List[Block]
    stop_reason: StopReason
    usage: Usage = field(default_factory=Usage)
    stop_details: Optional[Dict[str, Any]] = None

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_calls(self) -> List[ToolCall]:
        return [b for b in self.content if isinstance(b, ToolCall)]


# ---------------------------------------------------------------- serialization

_BLOCK_TYPES = {
    "text": TextBlock,
    "tool_call": ToolCall,
    "tool_result": ToolResultBlock,
    "provider": ProviderBlock,
}


def block_to_dict(block: Block) -> Dict[str, Any]:
    return dict(block.__dict__)


def block_from_dict(d: Dict[str, Any]) -> Block:
    cls = _BLOCK_TYPES[d["type"]]
    return cls(**d)


def message_to_dict(m: Message) -> Dict[str, Any]:
    return {"role": m.role, "content": [block_to_dict(b) for b in m.content]}


def message_from_dict(d: Dict[str, Any]) -> Message:
    return Message(d["role"], [block_from_dict(b) for b in d["content"]])


def render_block(block: Block, max_chars: int = 2000) -> str:
    """Plain-text rendering, used for summaries and token estimates."""
    if isinstance(block, TextBlock):
        return block.text
    if isinstance(block, ToolCall):
        return "[tool call %s] %s" % (block.name, json.dumps(block.arguments)[:max_chars])
    if isinstance(block, ToolResultBlock):
        tag = "tool error" if block.is_error else "tool result"
        body = block.content
        if len(body) > max_chars:
            body = body[:max_chars] + " ...[%d chars omitted]" % (len(body) - max_chars)
        return "[%s] %s" % (tag, body)
    return ""
