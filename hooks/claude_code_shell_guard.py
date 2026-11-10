"""Claude Code PreToolUse hook: block dangerous Bash commands with the
harness's ShellPolicy.

Register it for the Bash tool:

    {"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command",
      "command": "/usr/bin/python3 /path/to/agent-harness/hooks/claude_code_shell_guard.py"}]}]}}

It only ever denies. Commands the policy would auto-allow inside the harness
sandbox are not auto-approved here, because Claude Code is not confined to a
workspace the same way; everything not blocked goes through Claude Code's
normal permission flow. On its own errors it exits 1, which Claude Code
reports without blocking, so a bug here can't wedge a session.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from agent_harness.safety.policy import DEFAULT_SHELL_DENY, Decision, ShellPolicy  # noqa: E402

# Your own additions (Python regexes), checked alongside the defaults.
EXTRA_DENY = [
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\s+-[a-zA-Z]*f",
]


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except ValueError:
        return 0
    if event.get("tool_name") != "Bash":
        return 0
    command = (event.get("tool_input") or {}).get("command") or ""
    verdict = ShellPolicy(deny_patterns=DEFAULT_SHELL_DENY + EXTRA_DENY).check_command(command)
    if verdict is not None and verdict.decision == Decision.DENY:
        json.dump(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "Blocked by the agent-harness shell guard: %s" % verdict.reason,
                }
            },
            sys.stdout,
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # report, don't block
        print("shell guard error: %s: %s" % (type(e).__name__, e), file=sys.stderr)
        sys.exit(1)
