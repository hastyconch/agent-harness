"""One-call wiring of the default stack."""

from __future__ import annotations

from typing import List, Optional

from .agent import Agent, AgentConfig, EventHandler
from .context import ContextConfig, ContextManager
from .providers.base import Provider
from .safety.judge import LLMJudge
from .safety.policy import Approver, PermissionGate, Rule, ShellPolicy
from .sandbox import LocalShellRunner, Workspace
from .tools.builtin import SearchBackend, register_builtin_tools
from .tools.registry import ToolContext, ToolRegistry


def build_agent(
    provider: Provider,
    workspace_dir: str,
    approver: Optional[Approver] = None,
    judge_provider: Optional[Provider] = None,
    judge_can_allow: bool = False,
    rules: Optional[List[Rule]] = None,
    shell_allow_prefixes: Optional[List[str]] = None,
    search: Optional[SearchBackend] = None,
    shell=None,
    agent_config: Optional[AgentConfig] = None,
    context_config: Optional[ContextConfig] = None,
    on_event: Optional[EventHandler] = None,
) -> Agent:
    workspace = Workspace(workspace_dir)
    registry = register_builtin_tools(ToolRegistry())
    gate = PermissionGate(
        rules=rules,
        analyzers=[ShellPolicy(allow_prefixes=shell_allow_prefixes)],
        judge=LLMJudge(judge_provider) if judge_provider else None,
        judge_can_allow=judge_can_allow,
        approver=approver,
    )
    return Agent(
        provider=provider,
        registry=registry,
        gate=gate,
        tool_context=ToolContext(workspace=workspace, shell=shell or LocalShellRunner(workspace), search=search),
        context=ContextManager(provider, context_config),
        config=agent_config,
        on_event=on_event,
    )
