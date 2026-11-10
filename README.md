# agent-harness

A provider-independent agent harness in Python. It wraps an LLM in an execution loop
with validated tool calls, a permission gate, context management and a workspace sandbox.

The core needs only the standard library (Python 3.9+). Claude support comes from an
optional adapter (`pip install anthropic`).

```
agent_harness/
  agent.py            core loop: model -> tools -> results -> model, until a final answer
  conversation.py     neutral message/block types that every provider maps to and from
  providers/
    base.py           Provider interface (one method: complete)
    anthropic.py      Claude adapter (adaptive thinking, streaming, refusal fallback)
    scripted.py       deterministic provider for tests and offline demos
  tools/
    schema.py         JSON Schema validator with type coercion and defaults
    registry.py       Tool, ToolRegistry: prepare (validate) -> execute (timeout, errors, truncation)
    builtin.py        read_file, list_dir, grep, write_file, edit_file, delete_file, run_shell, web_search
  safety/
    policy.py         PermissionGate, Rule, ShellPolicy, risk defaults, audit log
    judge.py          LLMJudge: AI reviewer for calls that need approval (fails closed)
    approvers.py      ConsoleApprover, deny_all, allow_all
  context.py          token budget tracking, pruning, compaction (LLM summary with fallback)
  session.py          Session state and JSON save/resume
  sandbox.py          Workspace (path confinement, soft delete), Local/Docker shell runners
  factory.py          build_agent(): the default stack in one call
```

## Quick start

```python
from agent_harness import build_agent, ConsoleApprover
from agent_harness.providers.anthropic import AnthropicProvider

agent = build_agent(
    AnthropicProvider(),                 # claude-opus-5-5, effort "high"
    workspace_dir="./workspace",
    approver=ConsoleApprover(),          # asks you y/n/always for risky calls
    shell_allow_prefixes=["python -m pytest"],
)
result = agent.run("Write fizzbuzz.py with tests, then run the tests.")
print(result.status, result.text)
```

Or use the REPL:

```bash
python3 -m agent_harness --workspace ./workspace --judge --session session.json
```

## How a turn works

1. `ContextManager.ensure_fits` prunes or compacts if the next request would be too large.
2. The provider is called. Retryable errors (rate limits, 5xx, network) back off and retry.
3. If there are no tool calls, the run ends with `status="completed"`.
4. Each call is checked in order: the registry looks up the tool and validates its arguments
   (coercing `"5"` to `5`, filling defaults); a guard catches repeated identical calls; then the
   `PermissionGate` decides.
5. Allowed calls run. Read-only calls may run in parallel; anything that writes runs one at a time.
   Exceptions, timeouts and oversized output all become error results the model can read.
6. All results go back in a single message, in call order, and the loop continues.

`run()` always returns an `AgentResult`. `status` is one of `completed`, `refused`, `max_tokens`,
`max_iterations`, `stuck`, `budget_exceeded`, `context_overflow`, `provider_error` or `cancelled`.
On Ctrl-C, every pending call gets a "cancelled" result, so the session stays valid and can be resumed.

## Safety gate

Each call goes through these steps in order. The first step that decides wins:

| Step | Example |
|---|---|
| Hard DENY rules | `Rule("no-secrets", Decision.DENY, ["read_file"], when=lambda a: ".env" in a["path"])` |
| Tool analyzers | `ShellPolicy` blocks `sudo`, `rm -rf /`, `curl … \| sh`, force-push and fork bombs; auto-allows pipelines made only of read-only programs (`ls`, `cat`, `grep`, …) |
| Session memory | the user previously answered "always" |
| ALLOW/ASK rules | allowlists for specific tools or arguments |
| Risk defaults | READ and WRITE (inside the sandbox) are allowed; NETWORK, EXECUTE and DESTRUCTIVE ask |
| AI judge | reviews ASK calls against the user's task. It can block. It can only approve if `judge_can_allow=True`. If it errors, the call still needs approval |
| Approver | a human or a policy function. With no approver, the call is denied |

A denied call is returned to the model as an error result, with the reason, so the model can adapt.
`gate.audit` records every decision.

## Context management

- **Tracking.** The token count starts from the provider's reported usage, which is exact. Only
  the messages added since that report are estimated.
- **Pruning** (at 60% of the budget). Large tool results that are older than the last few messages
  are cut down to their first 400 characters.
- **Compaction** (at 80% of the budget). Everything before a cut point is summarized into one
  message. The recent tail is kept verbatim. The cut always lands on an assistant turn, so a tool
  call is never separated from its result. If the summarizer fails, a mechanical outline is used
  instead.
- **Append-only otherwise.** Claude's thinking blocks are only valid with the exact history that
  produced them. Any rewrite therefore removes reasoning blocks from the first edited message onward.
  Blocks before the edit are kept.
- **Sessions.** `Session.save` and `Session.load` persist state. `AgentConfig(autosave_path=...)`
  saves after every run.

## Sandbox

`Workspace` resolves every path with `realpath` and rejects anything outside the root. That covers
`..`, absolute paths, `~` and symlinks that point out of the workspace. Writes are atomic. `.git` is
read-only. `delete_file` moves files into `.harness/trash/` so they can be restored. `.harness/` also holds the shell's `HOME` and `TMPDIR` and is hidden from listings.

`LocalShellRunner` runs commands with the workspace as the working directory and a scrubbed
environment (no API keys). It also sets CPU and file-size rlimits, and on timeout it kills the
whole process group. **It is not a security boundary:** a command can still read files outside the
workspace by absolute path. For untrusted code, pass `shell=DockerShellRunner(workspace)`. That runs
each command in a container with no network, only the workspace mounted, and memory, CPU and pid
limits.

## Adding a provider

Subclass `Provider` and implement `complete(ModelRequest) -> ModelResponse`. Map native blocks to
`TextBlock`/`ToolCall`, and put anything else (reasoning, citations) into `ProviderBlock`. Those are
replayed verbatim to the same provider and dropped for any other. Raise
`ProviderError(retryable=...)` for API failures.

## Adding a tool

```python
from agent_harness import Risk, ToolError

@registry.tool("http_get", "Fetch a URL.", {
    "type": "object",
    "properties": {"url": {"type": "string", "pattern": "^https://"}},
    "required": ["url"], "additionalProperties": False,
}, risk=Risk.NETWORK, timeout=20)
def http_get(args, ctx):
    ...
    raise ToolError("shown to the model as an error result")
```

## Claude Code shell guard

`hooks/claude_code_shell_guard.py` reuses `ShellPolicy` as a Claude Code `PreToolUse` hook. It
blocks dangerous Bash commands (`sudo`, `rm -rf /`, `curl … | sh`, force-push,
`git reset --hard`, …) and tells Claude why. It never approves anything, so every other command
still goes through Claude Code's normal permission prompts. It needs only the standard library.

To use it in every project, add this to `~/.claude/settings.json`, with the absolute path to your
clone:

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Bash",
      "hooks": [{"type": "command", "command": "python3 /path/to/agent-harness/hooks/claude_code_shell_guard.py", "timeout": 10}]
    }]
  }
}
```

To add your own blocked patterns, edit `EXTRA_DENY` at the top of the script.

## Tests

```bash
python3 -m unittest discover -s tests
```

The tests run offline with `ScriptedProvider`. A fake SDK client checks the Anthropic adapter's
request and response mapping.
