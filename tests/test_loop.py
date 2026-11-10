import time
import unittest

from agent_harness import AgentConfig, ProviderError, Risk, StopReason, Tool, ToolRegistry, ToolResultBlock
from agent_harness.conversation import ModelResponse, ProviderBlock, TextBlock, Usage
from agent_harness.tools import register_builtin_tools

from helpers import WorkspaceTestCase, calls, text


def last_results(provider, request_index=-1):
    return provider.requests[request_index].messages[-1].tool_results


class LoopTests(WorkspaceTestCase):
    def test_plain_answer(self):
        agent, provider = self.make_agent([text("hello")])
        result = agent.run("hi")
        self.assertEqual((result.status, result.text, result.iterations), ("completed", "hello", 1))
        self.assertEqual(len(provider.requests[0].tools), 8)

    def test_tool_round_trip(self):
        (self.ws.root / "notes.txt").write_text("alpha\nbeta\n")
        step = calls(("read_file", {"path": "notes.txt"}))
        agent, provider = self.make_agent([step, text("it says alpha, beta")])
        result = agent.run("read notes")
        self.assertTrue(result.ok)
        [res] = last_results(provider)
        self.assertEqual(res.tool_call_id, step.content[0].id)
        self.assertFalse(res.is_error)
        self.assertIn("alpha", res.content)

    def test_reasoning_blocks_are_replayed_in_order(self):
        thinking = ProviderBlock("scripted", {"type": "thinking", "signature": "x"}, reasoning=True)
        agent, provider = self.make_agent([calls(("list_dir", {}), prefix=[thinking]), text("done")])
        agent.run("ls")
        assistant = provider.requests[1].messages[1]
        self.assertIs(type(assistant.content[0]), ProviderBlock)
        self.assertEqual(assistant.content[1].type, "tool_call")

    def test_invalid_args_become_error_and_model_recovers(self):
        bad = calls(("read_file", {"pth": "x"}))
        good = calls(("read_file", {"path": "x.txt", "limit": "3"}))  # "3" is coerced to 3
        (self.ws.root / "x.txt").write_text("1\n2\n3\n4\n")
        agent, provider = self.make_agent([bad, good, text("ok")])
        self.assertTrue(agent.run("read x").ok)
        [err] = last_results(provider, 1)
        self.assertTrue(err.is_error)
        self.assertIn("missing required property 'path'", err.content)
        self.assertIn("unexpected property 'pth'", err.content)
        [res] = last_results(provider, 2)
        self.assertFalse(res.is_error)
        self.assertNotIn("4", res.content.split("\t")[-1])

    def test_unknown_tool_and_unparseable_args(self):
        step = calls(("nope", {}), ("read_file", {}))
        step.content[1].parse_error = "Unterminated string"
        agent, provider = self.make_agent([step, text("ok")])
        agent.run("x")
        unknown, unparsed = last_results(provider)
        self.assertIn("Unknown tool 'nope'", unknown.content)
        self.assertIn("Could not parse arguments", unparsed.content)

    def test_tool_exception_and_timeout_do_not_crash(self):
        reg = ToolRegistry()
        params = {"type": "object", "properties": {}}
        reg.register(Tool("boom", "", params, lambda a, c: 1 / 0))
        reg.register(Tool("slow", "", params, lambda a, c: time.sleep(2), timeout=0.2))
        agent, provider = self.make_agent([calls(("boom", {}), ("slow", {})), text("ok")], registry=reg)
        self.assertTrue(agent.run("x").ok)
        boom, slow = last_results(provider)
        self.assertIn("ZeroDivisionError", boom.content)
        self.assertIn("timed out", slow.content)

    def test_output_is_truncated(self):
        reg = ToolRegistry(max_output_chars=1000)
        reg.register(Tool("big", "", {"type": "object"}, lambda a, c: "x" * 50_000))
        agent, provider = self.make_agent([calls(("big", {})), text("ok")], registry=reg)
        agent.run("x")
        [res] = last_results(provider)
        self.assertLess(len(res.content), 1100)
        self.assertIn("characters omitted", res.content)

    def test_retryable_provider_errors_are_retried(self):
        agent, provider = self.make_agent([ProviderError("429", retryable=True), ProviderError("503", retryable=True), text("ok")])
        self.assertTrue(agent.run("x").ok)
        self.assertEqual(sum(1 for k, _ in self.events if k == "retry"), 2)

    def test_fatal_provider_error_stops_cleanly(self):
        agent, _ = self.make_agent([ProviderError("400 bad request")])
        result = agent.run("x")
        self.assertEqual(result.status, "provider_error")
        agent2, _ = self.make_agent([RuntimeError("adapter bug")])
        self.assertEqual(agent2.run("x").status, "provider_error")

    def test_truncated_tool_call_is_not_executed(self):
        step = calls(("write_file", {"path": "a.txt", "content": "partial"}), stop=StopReason.MAX_TOKENS)
        agent, provider = self.make_agent([step, text("ok")])
        agent.run("write")
        self.assertFalse((self.ws.root / "a.txt").exists())
        [res] = last_results(provider)
        self.assertTrue(res.is_error)
        self.assertIn("output token limit", res.content)

    def test_refusal_is_not_added_and_next_input_merges(self):
        refusal = ModelResponse([TextBlock("partial")], StopReason.REFUSAL, Usage(10, 1), {"category": "cyber"})
        agent, provider = self.make_agent([refusal, text("sure")])
        self.assertEqual(agent.run("bad").status, "refused")
        self.assertEqual([m.role for m in agent.session.messages], ["user"])
        agent.run("something else")
        msgs = provider.requests[1].messages
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].text, "badsomething else")

    def test_identical_repeated_calls_are_blocked(self):
        steps = [calls(("list_dir", {})) for _ in range(5)] + [text("ok")]
        agent, provider = self.make_agent(steps, config=AgentConfig(max_identical_calls=3, retry_base_delay=0))
        agent.run("x")
        self.assertFalse(last_results(provider, 3)[0].is_error)
        self.assertIn("identical call", last_results(provider, 4)[0].content)

    def test_stuck_when_every_call_keeps_failing(self):
        steps = [calls(("read_file", {"path": "missing%d" % i})) for i in range(10)]
        agent, _ = self.make_agent(steps, config=AgentConfig(max_consecutive_failed_turns=3))
        result = agent.run("x")
        self.assertEqual((result.status, result.iterations), ("stuck", 3))

    def test_iteration_and_token_budgets(self):
        steps = [calls(("list_dir", {"path": "."}), ("grep", {"pattern": str(i)})) for i in range(10)]
        agent, _ = self.make_agent(list(steps), config=AgentConfig(max_iterations=4, max_identical_calls=99))
        self.assertEqual(agent.run("x").status, "max_iterations")
        agent, _ = self.make_agent(list(steps), config=AgentConfig(max_session_tokens=250, max_identical_calls=99))
        result = agent.run("x")
        self.assertEqual((result.status, result.iterations), ("budget_exceeded", 3))

    def test_parallel_read_results_keep_order(self):
        for name in "abc":
            (self.ws.root / name).write_text(name * 3)
        step = calls(*[("read_file", {"path": n}) for n in "abc"])
        agent, provider = self.make_agent([step, text("ok")])
        agent.run("x")
        results = last_results(provider)
        self.assertEqual([r.tool_call_id for r in results], [c.id for c in step.content])
        self.assertEqual([r.content.split("\t")[1] for r in results], ["aaa", "bbb", "ccc"])

    def test_interrupt_closes_dangling_calls(self):
        reg = ToolRegistry()

        def interrupt(a, c):
            raise KeyboardInterrupt

        reg.register(Tool("ctrl_c", "", {"type": "object"}, interrupt))
        agent, _ = self.make_agent([calls(("ctrl_c", {}))], registry=reg)
        # KeyboardInterrupt surfaces through future.result() in the registry
        result = agent.run("x")
        self.assertEqual(result.status, "cancelled")
        last = agent.session.messages[-1]
        self.assertEqual(last.role, "user")
        self.assertTrue(all(isinstance(b, ToolResultBlock) for b in last.content))

    def test_post_tool_hook_can_redact(self):
        (self.ws.root / ".env").write_text("API_KEY=sk-secret-123\n")

        def redact(call, result):
            return ToolResultBlock(result.tool_call_id, result.content.replace("sk-secret-123", "[REDACTED]"), result.is_error)

        agent, provider = self.make_agent([calls(("read_file", {"path": ".env"})), text("ok")], post_tool_hooks=[redact])
        agent.run("x")
        self.assertIn("[REDACTED]", last_results(provider)[0].content)


if __name__ == "__main__":
    unittest.main()
