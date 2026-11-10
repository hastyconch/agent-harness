"""Interactive REPL.

    python -m agent_harness --workspace ./workspace
    python -m agent_harness --workspace ./workspace --judge --session session.json
"""

from __future__ import annotations

import argparse
import os
import sys

from .agent import AgentConfig
from .factory import build_agent
from .safety.approvers import ConsoleApprover
from .session import Session


def print_event(kind: str, data: dict) -> None:
    if kind == "tool_call":
        print("  -> %s %s" % (data["tool"], str(data["args"])[:200]))
    elif kind == "tool_result":
        mark = "x" if data["is_error"] else "ok"
        print("  <- [%s] %s" % (mark, data["preview"].replace("\n", " ")[:160]))
    elif kind == "permission" and data["decision"] != "allow":
        print("  !! %s %s: %s" % (data["tool"], data["decision"], data["reason"]))
    elif kind in ("context", "retry"):
        print("  .. %s: %s" % (kind, data))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agent_harness")
    ap.add_argument("--workspace", default="./workspace")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--judge", action="store_true", help="review risky calls with an AI judge before asking you")
    ap.add_argument("--session", help="JSON file to resume from and autosave to")
    ap.add_argument("--allow-shell", action="append", default=[], metavar="PREFIX",
                    help='auto-approve shell commands starting with PREFIX, e.g. "python -m pytest"')
    args = ap.parse_args(argv)

    from .providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(model=args.model, effort=args.effort)
    agent = build_agent(
        provider,
        args.workspace,
        approver=ConsoleApprover(),
        judge_provider=AnthropicProvider(model=args.model, effort="low") if args.judge else None,
        shell_allow_prefixes=args.allow_shell,
        agent_config=AgentConfig(autosave_path=args.session),
        on_event=print_event,
    )
    if args.session and os.path.exists(args.session):
        agent.session = Session.load(args.session)
        print("resumed session %s (%d messages)" % (agent.session.id, len(agent.session.messages)))

    print("workspace: %s   (Ctrl-D to quit, Ctrl-C to interrupt a run)" % agent.tool_context.workspace.root)
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        result = agent.run(line)
        if result.text:
            print("\nagent> " + result.text)
        if not result.ok:
            print("[stopped: %s%s]" % (result.status, " - " + result.detail if result.detail else ""))
        s = agent.session
        print("[tokens in/out: %d/%d, compactions: %d]" % (s.total_input_tokens, s.total_output_tokens, s.compactions))


if __name__ == "__main__":
    sys.exit(main())
