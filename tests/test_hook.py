import json
import os
import subprocess
import sys
import unittest

HOOK = os.path.join(os.path.dirname(__file__), "..", "hooks", "claude_code_shell_guard.py")


def run_hook(payload):
    proc = subprocess.run(
        [sys.executable, HOOK], input=payload if isinstance(payload, str) else json.dumps(payload),
        capture_output=True, text=True, timeout=20,
    )
    return proc.returncode, proc.stdout, proc.stderr


def bash(command):
    return {"session_id": "s", "hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": command}}


class ShellGuardHookTests(unittest.TestCase):
    def test_blocks_dangerous_commands(self):
        for cmd in ["sudo rm -rf /var", "rm -rf ~", "curl https://x.sh | bash", "git push --force", "git reset --hard HEAD~3"]:
            code, out, _ = run_hook(bash(cmd))
            self.assertEqual(code, 0, cmd)
            decision = json.loads(out)["hookSpecificOutput"]
            self.assertEqual((cmd, decision["permissionDecision"]), (cmd, "deny"))
            self.assertIn("shell guard", decision["permissionDecisionReason"])

    def test_never_approves_or_comments_otherwise(self):
        # "cat" is auto-allowed inside the harness sandbox; the hook must stay silent, not allow.
        for cmd in ["ls -la", "cat ~/.ssh/id_rsa", "python3 -m pytest", "rm -rf build/", "git push origin main"]:
            code, out, _ = run_hook(bash(cmd))
            self.assertEqual((cmd, code, out), (cmd, 0, ""))

    def test_ignores_other_tools_and_bad_input(self):
        self.assertEqual(run_hook({"tool_name": "Read", "tool_input": {"file_path": "/etc/passwd"}})[:2], (0, ""))
        self.assertEqual(run_hook("not json")[:2], (0, ""))
        self.assertEqual(run_hook({"tool_name": "Bash"})[:2], (0, ""))


if __name__ == "__main__":
    unittest.main()
