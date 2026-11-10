import os
import time
import unittest

from agent_harness import SandboxViolation, ToolCall, ToolContext, ToolRegistry
from agent_harness.sandbox import LocalShellRunner
from agent_harness.tools import register_builtin_tools

from helpers import WorkspaceTestCase


class WorkspaceTests(WorkspaceTestCase):
    def test_escapes_are_rejected(self):
        outside = os.path.dirname(self.tmp)
        os.symlink(outside, str(self.ws.root / "link"))
        for path in ["../x", "/etc/passwd", "a/../../x", "link/x", "~/.ssh/id_rsa", "a\x00b"]:
            with self.assertRaises(SandboxViolation, msg=path):
                self.ws.resolve(path)
        self.assertEqual(self.ws.resolve("a/../b"), self.ws.root / "b")
        self.assertEqual(self.ws.resolve(str(self.ws.root / "c")), self.ws.root / "c")

    def test_write_read_and_read_only_paths(self):
        self.ws.write_text("src/app.py", "print(1)\n")
        self.assertEqual(self.ws.read_text("src/app.py"), "print(1)\n")
        with self.assertRaises(SandboxViolation):
            self.ws.write_text(".git/config", "x")
        self.ws.write_text(".github/ci.yml", "x")  # prefix match must be per path segment

    def test_size_limit(self):
        self.ws.max_file_bytes = 10
        with self.assertRaises(SandboxViolation):
            self.ws.write_text("big", "x" * 11)

    def test_delete_is_soft(self):
        self.ws.write_text("old/a.txt", "x")
        moved = self.ws.delete("old")
        self.assertFalse((self.ws.root / "old").exists())
        self.assertTrue((self.ws.root / moved / "a.txt").exists())
        self.assertNotIn(self.ws.HARNESS_DIR + "/", self.ws.list_dir("."))
        with self.assertRaises(SandboxViolation):
            self.ws.delete(".")


class ShellTests(WorkspaceTestCase):
    def test_runs_in_workspace_with_scrubbed_env(self):
        os.environ["HARNESS_TEST_SECRET"] = "s3cret"
        try:
            res = LocalShellRunner(self.ws).run("pwd; echo ${HARNESS_TEST_SECRET:-unset}; echo $HOME")
        finally:
            del os.environ["HARNESS_TEST_SECRET"]
        lines = res.stdout.split()
        self.assertEqual(os.path.realpath(lines[0]), str(self.ws.root))
        self.assertEqual(lines[1], "unset")
        self.assertEqual(lines[2], str(self.ws.root / ".harness" / "home"))
        self.assertEqual(self.ws.list_dir("."), [])  # HOME/TMPDIR stay out of listings

    def test_timeout_kills_process_group(self):
        start = time.time()
        res = LocalShellRunner(self.ws, timeout=0.5).run("sleep 5 & sleep 5; echo never")
        self.assertTrue(res.timed_out)
        self.assertLess(time.time() - start, 3)
        self.assertNotIn("never", res.stdout)

    def test_output_cap_and_exit_code(self):
        res = LocalShellRunner(self.ws, max_output_bytes=100).run("yes | head -c 10000; exit 3")
        self.assertEqual(res.exit_code, 3)
        self.assertEqual(len(res.stdout), 100)


class BuiltinToolTests(WorkspaceTestCase):
    def run_tool(self, name, **args):
        reg = register_builtin_tools(ToolRegistry())
        prepared = reg.prepare(ToolCall("id", name, args))
        if not hasattr(prepared, "tool"):
            return prepared
        return reg.execute(prepared, ToolContext(workspace=self.ws, shell=LocalShellRunner(self.ws)))

    def test_edit_file(self):
        self.ws.write_text("a.py", "x = 1\ny = 1\n")
        self.assertIn("2 times", self.run_tool("edit_file", path="a.py", old_text="= 1", new_text="= 2").content)
        self.assertFalse(self.run_tool("edit_file", path="a.py", old_text="x = 1", new_text="x = 3").is_error)
        self.assertEqual(self.ws.read_text("a.py"), "x = 3\ny = 1\n")
        self.assertTrue(self.run_tool("edit_file", path="a.py", old_text="zzz", new_text="").is_error)

    def test_sandbox_errors_are_tool_errors(self):
        res = self.run_tool("read_file", path="../../etc/passwd")
        self.assertTrue(res.is_error)
        self.assertIn("outside the workspace", res.content)

    def test_grep_and_search_stub(self):
        self.ws.write_text("src/a.py", "def foo():\n    return 1\n")
        self.assertIn("src/a.py:1: def foo():", self.run_tool("grep", pattern=r"def \w+").content)
        res = self.run_tool("web_search", query="x")
        self.assertTrue(res.is_error)
        self.assertIn("not configured", res.content)


if __name__ == "__main__":
    unittest.main()
