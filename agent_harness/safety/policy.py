"""Permission gate for tool calls.

Each validated call is decided in this order; the first step that decides wins:

  1. Hard DENY rules           (never overridable, not even by the user)
  2. Tool-specific analyzers   (e.g. the shell command policy)
  3. Session "always allow"    (remembered user approvals)
  4. ALLOW / ASK rules         (configured allowlists)
  5. Default by risk level     (READ/WRITE allow, NETWORK/EXECUTE/DESTRUCTIVE ask)
  6. AI judge                  (only for ASK: may block, and may auto-approve
                                if ``judge_can_allow``; fails closed)
  7. Human approver            (for whatever is still ASK; no approver = deny)

A denied call is not an error for the loop: it becomes an error tool result,
so the model sees why and can choose another approach.
"""

from __future__ import annotations

import enum
import fnmatch
import json
import re
import shlex
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Set, Tuple

from ..tools.registry import PreparedCall, Risk


class Decision(str, enum.Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass
class Verdict:
    decision: Decision
    reason: str
    source: str


@dataclass
class PermissionRequest:
    tool: str
    risk: Risk
    args: Dict
    reason: str  # why it needs approval
    judge_note: Optional[str] = None


class Approval(str, enum.Enum):
    YES = "yes"
    NO = "no"
    ALWAYS_EXACT = "always_exact"  # this exact call, for the rest of the session
    ALWAYS_TOOL = "always_tool"  # any call to this tool, for the rest of the session


Approver = Callable[[PermissionRequest], Approval]


@dataclass
class Rule:
    """Match a tool by name pattern (fnmatch) and optionally by arguments."""

    name: str
    decision: Decision
    tools: List[str] = field(default_factory=lambda: ["*"])
    when: Optional[Callable[[Dict], bool]] = None

    def matches(self, prepared: PreparedCall) -> bool:
        if not any(fnmatch.fnmatchcase(prepared.tool.name, p) for p in self.tools):
            return False
        return self.when is None or bool(self.when(prepared.args))


# --------------------------------------------------------------- shell policy

DEFAULT_SHELL_DENY = [
    r"\bsudo\b",
    r"\bsu\s",
    r"\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+(/|~|\$HOME|\*)(\s|$)",
    r"\bmkfs",
    r"\bdd\b.*\bof=/dev/",
    r">\s*/dev/(sd|disk|nvme)",
    r":\(\)\s*\{.*\};\s*:",  # fork bomb
    r"\b(curl|wget)\b[^|]*\|\s*(ba|z)?sh\b",  # pipe-to-shell
    r"\bchmod\s+(-R\s+)?[0-7]*777\s+/",
    r"\b(shutdown|reboot|halt)\b",
    r"\bgit\s+push\b.*(--force|-f)\b",
]

DEFAULT_SHELL_ALLOW = {
    "ls", "cat", "head", "tail", "wc", "pwd", "echo", "grep", "rg", "find",
    "diff", "sort", "uniq", "stat", "file", "which", "true", "date",
}

_SEPARATORS = re.compile(r"\|\||&&|[;|\n&]")
_UNANALYZABLE = ("$(", "`", "<(", ">(", ">")


class ShellPolicy:
    """Hard-denies known-dangerous commands and auto-allows commands made only
    of allowlisted, read-only programs. Everything else falls through (ASK by
    default)."""

    def __init__(
        self,
        tool_name: str = "run_shell",
        deny_patterns: Optional[List[str]] = None,
        allow_programs: Optional[Set[str]] = None,
        allow_prefixes: Optional[List[str]] = None,
    ):
        self.tool_name = tool_name
        self.deny = [re.compile(p) for p in (deny_patterns if deny_patterns is not None else DEFAULT_SHELL_DENY)]
        self.allow_programs = allow_programs if allow_programs is not None else set(DEFAULT_SHELL_ALLOW)
        # Exact command prefixes, e.g. "git status", "python -m pytest".
        self.allow_prefixes = allow_prefixes or []

    def evaluate(self, prepared: PreparedCall) -> Optional[Verdict]:
        if prepared.tool.name != self.tool_name:
            return None
        return self.check_command(prepared.args.get("command", ""))

    def check_command(self, command: str) -> Optional[Verdict]:
        """DENY, ALLOW, or None (no opinion) for a raw command string."""
        for rx in self.deny:
            if rx.search(command):
                return Verdict(Decision.DENY, "command matches blocked pattern %r" % rx.pattern, "shell_policy")
        if any(tok in command for tok in _UNANALYZABLE):
            return None  # substitutions/redirects: can't vouch for it
        segments = [s.strip() for s in _SEPARATORS.split(command) if s.strip()]
        if segments and all(self._segment_allowed(s) for s in segments):
            return Verdict(Decision.ALLOW, "all programs allowlisted", "shell_policy")
        return None

    def _segment_allowed(self, segment: str) -> bool:
        if any(segment == p or segment.startswith(p + " ") for p in self.allow_prefixes):
            return True
        try:
            argv = shlex.split(segment)
        except ValueError:
            return False
        if not argv:
            return False
        if argv[0] == "find" and any(a in ("-exec", "-execdir", "-delete", "-ok") for a in argv):
            return False
        return argv[0] in self.allow_programs


# ---------------------------------------------------------------- AI judge


class Judge(Protocol):
    def review(self, request: PermissionRequest, task_context: str) -> Verdict: ...


# --------------------------------------------------------------------- gate

DEFAULT_RISK_POLICY = {
    Risk.READ: Decision.ALLOW,
    Risk.WRITE: Decision.ALLOW,  # writes are confined to the sandbox
    Risk.NETWORK: Decision.ASK,
    Risk.EXECUTE: Decision.ASK,
    Risk.DESTRUCTIVE: Decision.ASK,
}


@dataclass
class AuditEntry:
    timestamp: float
    tool: str
    args: Dict
    verdict: Verdict


class PermissionGate:
    def __init__(
        self,
        rules: Optional[List[Rule]] = None,
        analyzers: Optional[List[ShellPolicy]] = None,
        risk_policy: Optional[Dict[Risk, Decision]] = None,
        judge: Optional[Judge] = None,
        judge_can_allow: bool = False,
        approver: Optional[Approver] = None,
    ):
        self.rules = rules or []
        self.analyzers = analyzers if analyzers is not None else [ShellPolicy()]
        self.risk_policy = dict(DEFAULT_RISK_POLICY, **(risk_policy or {}))
        self.judge = judge
        self.judge_can_allow = judge_can_allow
        self.approver = approver
        self._always_exact: Set[Tuple[str, str]] = set()
        self._always_tool: Set[str] = set()
        self.audit: List[AuditEntry] = []

    def check(self, prepared: PreparedCall, task_context: str = "") -> Verdict:
        verdict = self._decide(prepared, task_context)
        self.audit.append(AuditEntry(time.time(), prepared.tool.name, prepared.args, verdict))
        return verdict

    @staticmethod
    def _key(prepared: PreparedCall) -> Tuple[str, str]:
        return prepared.tool.name, json.dumps(prepared.args, sort_keys=True)

    def _decide(self, prepared: PreparedCall, task_context: str) -> Verdict:
        name = prepared.tool.name

        for rule in self.rules:
            if rule.decision == Decision.DENY and rule.matches(prepared):
                return Verdict(Decision.DENY, "blocked by rule %r" % rule.name, "rule:" + rule.name)

        verdict: Optional[Verdict] = None
        for analyzer in self.analyzers:
            verdict = analyzer.evaluate(prepared)
            if verdict is not None:
                break
        if verdict is not None and verdict.decision == Decision.DENY:
            return verdict

        if name in self._always_tool or self._key(prepared) in self._always_exact:
            return Verdict(Decision.ALLOW, "approved earlier this session", "session")

        if verdict is None:
            for rule in self.rules:
                if rule.matches(prepared):
                    verdict = Verdict(rule.decision, "matched rule %r" % rule.name, "rule:" + rule.name)
                    break
        if verdict is None:
            decision = self.risk_policy[prepared.tool.risk]
            verdict = Verdict(decision, "default for %s risk" % prepared.tool.risk.name, "default")

        if verdict.decision != Decision.ASK:
            return verdict

        request = PermissionRequest(name, prepared.tool.risk, prepared.args, verdict.reason)

        if self.judge is not None:
            try:
                jv = self.judge.review(request, task_context)
            except Exception as e:  # fail closed: a broken judge never approves
                jv = Verdict(Decision.ASK, "judge unavailable (%s)" % e, "judge")
            if jv.decision == Decision.DENY:
                return Verdict(Decision.DENY, "AI judge: " + jv.reason, "judge")
            if jv.decision == Decision.ALLOW and self.judge_can_allow:
                return Verdict(Decision.ALLOW, "AI judge: " + jv.reason, "judge")
            request.judge_note = jv.reason

        if self.approver is None:
            return Verdict(Decision.DENY, "needs approval but no approver is configured", "no_approver")
        answer = self.approver(request)
        if answer == Approval.ALWAYS_TOOL:
            self._always_tool.add(name)
        elif answer == Approval.ALWAYS_EXACT:
            self._always_exact.add(self._key(prepared))
        # The approver may be a person or a policy function, so don't claim "the user".
        note = " (AI judge: %s)" % request.judge_note if request.judge_note else ""
        if answer == Approval.NO:
            return Verdict(Decision.DENY, "the approver declined this call" + note, "approver")
        return Verdict(Decision.ALLOW, "approved by the approver" + note, "approver")
