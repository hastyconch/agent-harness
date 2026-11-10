import os
import unittest

from agent_harness import ContextConfig, ContextManager, ContextOverflow, Message, ScriptedProvider, Session
from agent_harness.context import SUMMARY_PREFIX, estimate_messages
from agent_harness.conversation import ProviderBlock, TextBlock, ToolCall, ToolResultBlock, Usage

from helpers import WorkspaceTestCase, calls, text

THINK = lambda: ProviderBlock("p", {"type": "thinking"}, reasoning=True)  # noqa: E731


def turn(i, result_size=4000):
    """One assistant tool call plus its result."""
    return [
        Message("assistant", [THINK(), ToolCall("c%d" % i, "read_file", {"path": "f%d" % i})]),
        Message("user", [ToolResultBlock("c%d" % i, "r" * result_size)]),
    ]


def session_with(n_turns, result_size=4000):
    s = Session(system="sys", messages=[Message.user("do the task")])
    for i in range(n_turns):
        s.messages.extend(turn(i, result_size))
    return s


class ContextTests(unittest.TestCase):
    def manager(self, summary="SUMMARY", **cfg):
        provider = ScriptedProvider([text(summary)], context_window=100_000)
        return ContextManager(provider, ContextConfig(**cfg)), provider

    def test_estimate_anchors_on_reported_usage(self):
        cm, _ = self.manager()
        s = Session(system="sys", messages=[Message.user("hi"), Message("assistant", [TextBlock("yo")])])
        cm.record_usage(s, Usage(5000, 100))
        self.assertEqual(cm.estimate(s, []), 5100)
        s.append(Message.user("x" * 400))
        self.assertAlmostEqual(cm.estimate(s, []), 5100 + 105, delta=10)
        self.assertEqual((s.total_input_tokens, s.total_output_tokens), (5000, 100))

    def test_prune_shortens_old_results_and_strips_later_reasoning(self):
        cm, _ = self.manager(protect_recent_messages=4, pruned_result_chars=100)
        s = session_with(5)
        s.messages[2].content[0] = ToolResultBlock("c0", "short")  # first result is small: not edited
        n = cm.prune(s)
        self.assertEqual(n, 2)  # turns 1 and 2; turns 3-4 are protected
        self.assertIn("pruned", s.messages[4].content[0].content)
        self.assertEqual(s.messages[-1].content[0].content, "r" * 4000)
        # reasoning before the first edit survives; from the first edit on it is gone
        self.assertIsInstance(s.messages[1].content[0], ProviderBlock)
        self.assertIsInstance(s.messages[3].content[0], ProviderBlock)
        for m in s.messages[4:]:
            self.assertFalse(any(isinstance(b, ProviderBlock) for b in m.content))

    def test_compaction_keeps_pairs_and_recent_tail(self):
        cm, provider = self.manager(keep_recent_tokens=2500)
        s = session_with(6)
        tail_before = [m.content[-1] for m in s.messages[-4:]]
        cm.compact(s)
        self.assertEqual(s.messages[0].role, "user")
        self.assertTrue(s.messages[0].text.startswith(SUMMARY_PREFIX + "SUMMARY"))
        self.assertEqual(s.messages[1].role, "assistant")  # cut lands on an assistant turn
        roles = [m.role for m in s.messages]
        self.assertTrue(all(a != b for a, b in zip(roles, roles[1:])), roles)
        call_ids = {c.id for m in s.messages for c in m.tool_calls}
        result_ids = {r.tool_call_id for m in s.messages for r in m.tool_results}
        self.assertEqual(call_ids, result_ids)
        self.assertEqual([m.content[-1] for m in s.messages[-4:]], tail_before)
        self.assertFalse(any(isinstance(b, ProviderBlock) for m in s.messages for b in m.content))
        self.assertIn("do the task", provider.requests[0].messages[0].text)
        self.assertEqual(s.compactions, 1)

    def test_summarizer_failure_falls_back(self):
        cm, _ = self.manager(keep_recent_tokens=2500)
        cm.summarizer = lambda msgs: 1 / 0
        s = session_with(4)
        cm.compact(s)
        self.assertIn("automatic outline", s.messages[0].text)
        self.assertIn("do the task", s.messages[0].text)

    def test_ensure_fits_thresholds(self):
        cm, _ = self.manager(max_context_tokens=20_000, reserve_tokens=1000, keep_recent_tokens=3000, protect_recent_messages=2)
        s = session_with(1, 100)
        self.assertEqual(cm.ensure_fits(s, []), [])
        s = session_with(16)  # ~16k tokens of results
        actions = cm.ensure_fits(s, [])
        self.assertTrue(any("pruned" in a for a in actions), actions)
        self.assertLess(cm.estimate(s, []), 20_000 * 0.8)

    def test_overflow_when_nothing_can_be_compacted(self):
        cm, _ = self.manager(max_context_tokens=1000, reserve_tokens=100)
        s = Session(system="", messages=[Message.user("x" * 8000)])
        with self.assertRaises(ContextOverflow):
            cm.ensure_fits(s, [])


class CompactionInLoopTests(WorkspaceTestCase):
    def test_long_run_compacts_and_finishes(self):
        for i in range(12):
            self.ws.write_text("f%d.txt" % i, ("line %d\n" % i) * 600)  # ~4.8k chars each
        steps = [calls(("read_file", {"path": "f%d.txt" % i}), usage=(0, 0)) for i in range(12)]
        steps.append(text("all read"))
        provider = ScriptedProvider([], context_window=12_000)
        summaries = []

        def summarizer(msgs):
            summaries.append(len(msgs))
            return "read files so far"

        # Pruning disabled (huge threshold) so the run has to compact.
        cfg = ContextConfig(reserve_tokens=1000, keep_recent_tokens=2000, protect_recent_messages=2, pruned_result_chars=10**6)
        cm = ContextManager(provider, cfg, summarizer)
        agent, provider = self.make_agent(steps, context=cm)
        cm.budget = 12_000
        result = agent.run("read every file")
        self.assertTrue(result.ok, result)
        self.assertGreater(agent.session.compactions, 0)
        self.assertTrue(any(k == "context" for k, _ in self.events))
        for req in provider.requests:
            self.assertLess(estimate_messages(req.messages), 12_000)


class SessionTests(WorkspaceTestCase):
    def test_round_trip(self):
        s = session_with(2)
        s.messages[1].content.append(TextBlock("note"))
        path = os.path.join(self.tmp, "s.json")
        s.save(path)
        loaded = Session.load(path)
        self.assertEqual(loaded.messages, s.messages)
        self.assertEqual(loaded.id, s.id)


if __name__ == "__main__":
    unittest.main()
