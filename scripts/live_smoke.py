"""Live end-to-end check against Claude. Costs a small amount of API usage.

    .venv/bin/python scripts/live_smoke.py

1. Builds fizzbuzz + tests in a temp workspace and runs them (shell allowlisted
   for `python3 -m unittest` only; everything else needing approval is denied,
   after the AI judge has a look).
2. Forces a compaction, then asks a follow-up, to prove the rewritten history
   (summary + kept tail, reasoning stripped) is accepted by the API.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_harness import AgentConfig, build_agent  # noqa: E402
from agent_harness.__main__ import print_event  # noqa: E402
from agent_harness.providers.anthropic import AnthropicProvider  # noqa: E402
from agent_harness.safety import deny_all  # noqa: E402


def main() -> int:
    workspace = tempfile.mkdtemp(prefix="harness-live-")
    agent = build_agent(
        AnthropicProvider(),
        workspace,
        approver=deny_all,
        judge_provider=AnthropicProvider(effort="low"),
        shell_allow_prefixes=["python3 -m unittest"],
        agent_config=AgentConfig(max_iterations=20, max_session_tokens=400_000),
        on_event=print_event,
    )

    print("== task 1 (workspace: %s)" % workspace)
    r1 = agent.run(
        "Create fizzbuzz.py with a function fizzbuzz(n) returning the list of strings for 1..n, "
        "and test_fizzbuzz.py with unittest tests. Run them with `python3 -m unittest` and fix "
        "anything that fails. Also try to delete test_fizzbuzz.py at the end (I want to see what happens)."
    )
    print("\nstatus=%s detail=%s\n%s" % (r1.status, r1.detail, r1.text))
    if not r1.ok:
        return 1

    print("\n== forcing compaction")
    agent.context.compact(agent.session)
    print("messages after compaction: %d" % len(agent.session.messages))

    print("\n== task 2 (after compaction)")
    r2 = agent.run("What files exist now, and do the tests still pass? Run them again to check.")
    print("\nstatus=%s detail=%s\n%s" % (r2.status, r2.detail, r2.text))

    s = agent.session
    print("\n== summary")
    print("tokens in/out: %d/%d  compactions: %d" % (s.total_input_tokens, s.total_output_tokens, s.compactions))
    for e in agent.gate.audit:
        print("  gate: %-12s %-6s %-12s %s" % (e.tool, e.verdict.decision.value, e.verdict.source, e.verdict.reason[:70]))
    print("files:", sorted(os.listdir(workspace)))
    return 0 if r1.ok and r2.ok else 1


if __name__ == "__main__":
    sys.exit(main())
