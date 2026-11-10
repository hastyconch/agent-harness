"""Schema-backed tool registry and dispatcher.

Dispatch is split in two so the safety gate can sit in between:

    prepared = registry.prepare(call)        # lookup + argument validation
    ...gate decides on prepared...
    result = registry.execute(prepared, ctx) # timeout, error capture, truncation

Nothing a tool does (bad arguments, exceptions, hangs, huge output) escapes as
an exception: every outcome becomes a ``ToolResultBlock`` the model can read
and recover from.
"""

from __future__ import annotations

import concurrent.futures
import enum
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from ..conversation import ToolCall, ToolResultBlock, ToolSpec
from .schema import validate

log = logging.getLogger(__name__)


class Risk(enum.IntEnum):
    """How much damage a call could do; the safety gate's default policy
    keys off this."""

    READ = 0
    WRITE = 1
    NETWORK = 2
    EXECUTE = 3
    DESTRUCTIVE = 4


class ToolError(Exception):
    """Raise from a handler for an expected failure; the message is shown to
    the model as-is."""


@dataclass
class ToolContext:
    """Whatever handlers need at run time (workspace, shell runner, search
    backend...). Built once per agent."""

    workspace: Any = None
    shell: Any = None
    search: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)


Handler = Callable[[Dict[str, Any], ToolContext], Any]


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Handler
    risk: Risk = Risk.READ
    timeout: float = 60.0

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)


@dataclass
class PreparedCall:
    call: ToolCall
    tool: Tool
    args: Dict[str, Any]


class ToolRegistry:
    def __init__(self, max_output_chars: int = 20_000, coerce_args: bool = True, max_workers: int = 8):
        self._tools: Dict[str, Tool] = {}
        self.max_output_chars = max_output_chars
        self.coerce_args = coerce_args
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tool")

    # ------------------------------------------------------- registration

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError("duplicate tool name %r" % tool.name)
        if tool.parameters.get("type") != "object":
            raise ValueError("tool %r: parameters must be a JSON Schema object" % tool.name)
        self._tools[tool.name] = tool
        return tool

    def tool(self, name: str, description: str, parameters: Dict[str, Any], risk: Risk = Risk.READ, timeout: float = 60.0):
        """Decorator form of ``register``."""

        def wrap(fn: Handler) -> Handler:
            self.register(Tool(name, description, parameters, fn, risk, timeout))
            return fn

        return wrap

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def specs(self) -> List[ToolSpec]:
        # Sorted so the tool list is byte-stable across requests (prompt caching).
        return [self._tools[n].spec() for n in sorted(self._tools)]

    # ----------------------------------------------------------- dispatch

    def prepare(self, call: ToolCall) -> Union[PreparedCall, ToolResultBlock]:
        tool = self._tools.get(call.name)
        if tool is None:
            return self.error(call, "Unknown tool %r. Available tools: %s" % (call.name, ", ".join(sorted(self._tools))))
        if call.parse_error:
            return self.error(call, "Could not parse arguments for %r: %s. Re-issue the call with valid JSON." % (call.name, call.parse_error))
        args, errors = validate(call.arguments, tool.parameters, coerce=self.coerce_args)
        if errors:
            return self.error(
                call,
                "Invalid arguments for %r:\n- %s\nExpected schema: %s"
                % (call.name, "\n- ".join(errors), json.dumps(tool.parameters)),
            )
        return PreparedCall(call, tool, args)

    def execute(self, prepared: PreparedCall, ctx: ToolContext) -> ToolResultBlock:
        call, tool = prepared.call, prepared.tool
        future = self._pool.submit(tool.handler, prepared.args, ctx)
        try:
            output = future.result(timeout=tool.timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            return self.error(call, "Tool %r timed out after %.0fs." % (tool.name, tool.timeout))
        except ToolError as e:
            return self.error(call, str(e))
        except Exception as e:  # never let a tool crash the loop
            log.exception("tool %s failed", tool.name)
            return self.error(call, "Tool %r failed: %s: %s" % (tool.name, type(e).__name__, e))
        return ToolResultBlock(call.id, self._truncate(self._stringify(output)))

    def error(self, call: ToolCall, message: str) -> ToolResultBlock:
        return ToolResultBlock(call.id, self._truncate(message), is_error=True)

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _stringify(output: Any) -> str:
        if output is None:
            return "(no output)"
        if isinstance(output, str):
            return output
        try:
            return json.dumps(output, indent=2, default=str)
        except (TypeError, ValueError):
            return repr(output)

    def _truncate(self, text: str) -> str:
        limit = self.max_output_chars
        if len(text) <= limit:
            return text
        head = int(limit * 0.7)
        tail = limit - head
        omitted = len(text) - head - tail
        return "%s\n\n...[%d characters omitted]...\n\n%s" % (text[:head], omitted, text[-tail:])
