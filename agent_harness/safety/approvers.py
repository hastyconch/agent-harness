"""Approver implementations: who answers when the gate says ASK."""

from __future__ import annotations

import json
from typing import Callable, Optional

from .policy import Approval, PermissionRequest


def deny_all(request: PermissionRequest) -> Approval:
    """Non-interactive default: anything needing approval is refused."""
    return Approval.NO


def allow_all(request: PermissionRequest) -> Approval:
    """For tests and fully sandboxed (container) runs only."""
    return Approval.YES


class ConsoleApprover:
    def __init__(self, input_fn: Callable[[str], str] = input, print_fn: Callable[[str], None] = print):
        self.input_fn = input_fn
        self.print_fn = print_fn

    def __call__(self, request: PermissionRequest) -> Approval:
        self.print_fn("\n--- permission required: %s (%s risk) ---" % (request.tool, request.risk.name))
        self.print_fn(json.dumps(request.args, indent=2)[:2000])
        self.print_fn("why: " + request.reason)
        if request.judge_note:
            self.print_fn("AI judge: " + request.judge_note)
        choices = {"y": Approval.YES, "n": Approval.NO, "a": Approval.ALWAYS_EXACT, "t": Approval.ALWAYS_TOOL}
        while True:
            try:
                answer: Optional[str] = self.input_fn("[y]es / [n]o / [a]lways this call / always this [t]ool > ")
            except EOFError:
                return Approval.NO
            choice = choices.get((answer or "").strip().lower()[:1])
            if choice is not None:
                return choice
