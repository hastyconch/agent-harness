import itertools
import shutil
import tempfile
import unittest

from agent_harness import (
    Agent,
    AgentConfig,
    ContextManager,
    ModelResponse,
    PermissionGate,
    ScriptedProvider,
    StopReason,
    TextBlock,
    ToolCall,
    ToolContext,
    ToolRegistry,
    Usage,
    Workspace,
    register_builtin_tools,
)
from agent_harness.sandbox import LocalShellRunner
from agent_harness.safety import allow_all

_ids = itertools.count(1)


def text(t, usage=(100, 10)):
    return ModelResponse([TextBlock(t)], StopReason.END_TURN, Usage(*usage))


def calls(*specs, stop=StopReason.TOOL_USE, usage=(100, 10), prefix=()):
    """calls(("read_file", {"path": "a"}), ...) -> a tool-use response."""
    blocks = list(prefix) + [ToolCall("call_%d" % next(_ids), name, args) for name, args in specs]
    return ModelResponse(blocks, stop, Usage(*usage))


class WorkspaceTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="harness-test-")
        self.ws = Workspace(self.tmp)
        self.events = []

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_agent(self, steps, approver=allow_all, registry=None, config=None, context=None, gate=None, **kw):
        provider = ScriptedProvider(steps)
        registry = registry or register_builtin_tools(ToolRegistry())
        agent = Agent(
            provider,
            registry,
            gate or PermissionGate(approver=approver),
            tool_context=ToolContext(workspace=self.ws, shell=LocalShellRunner(self.ws, timeout=5)),
            context=context or ContextManager(provider),
            config=config or AgentConfig(retry_base_delay=0),
            on_event=lambda k, d: self.events.append((k, d)),
            sleep=lambda s: None,
            **kw
        )
        return agent, provider
