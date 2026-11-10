import sys
import types
import unittest

from agent_harness.conversation import Message, ModelRequest, ProviderBlock, StopReason, TextBlock, ToolCall, ToolResultBlock, ToolSpec
from agent_harness.tools import validate


class SchemaTests(unittest.TestCase):
    S = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "minimum": 1, "maximum": 5},
            "mode": {"type": "string", "enum": ["a", "b"], "default": "a"},
            "tags": {"type": "array", "items": {"type": "string", "pattern": "^[a-z]+$"}, "maxItems": 2},
            "flag": {"type": "boolean"},
            "opt": {"type": ["string", "null"]},
        },
        "required": ["n"],
        "additionalProperties": False,
    }

    def test_valid_with_coercion_and_defaults(self):
        value, errors = validate({"n": "3", "flag": "true", "opt": None}, self.S)
        self.assertEqual(errors, [])
        self.assertEqual(value, {"n": 3, "flag": True, "opt": None, "mode": "a"})

    def test_errors_have_paths(self):
        _, errors = validate({"n": 9, "mode": "c", "tags": ["ok", "BAD", "x"], "extra": 1}, self.S)
        joined = "\n".join(errors)
        for needle in ["$.n: must be <= 5", "$.mode: must be one of", "$.tags: allows at most 2", "$.tags[1]: does not match", "unexpected property 'extra'"]:
            self.assertIn(needle, joined)

    def test_no_coercion_and_bool_is_not_int(self):
        _, errors = validate({"n": "3"}, self.S, coerce=False)
        self.assertTrue(errors)
        _, errors = validate({"n": True}, self.S)
        self.assertTrue(errors)


# ----------------------------------------------------------- Anthropic adapter


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class _Stream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


class _FakeClient:
    def __init__(self, msg):
        self.calls = []
        outer = self

        class Messages:
            def stream(self, **params):
                outer.calls.append(params)
                return _Stream(msg)

        self.beta = _Obj(messages=Messages())


def _install_fake_sdk():
    mod = types.ModuleType("anthropic")
    for name in ["APIStatusError", "RateLimitError", "InternalServerError", "APIConnectionError"]:
        setattr(mod, name, type(name, (Exception,), {}))
    sys.modules.setdefault("anthropic", mod)


class AnthropicAdapterTests(unittest.TestCase):
    def setUp(self):
        _install_fake_sdk()
        from agent_harness.providers.anthropic import AnthropicProvider

        msg = _Obj(
            content=[
                _Obj(type="thinking", thinking="", signature="sig"),
                _Obj(type="text", text="Let me look."),
                _Obj(type="tool_use", id="tu_1", name="read_file", input={"path": "a"}),
            ],
            stop_reason="tool_use",
            usage=_Obj(input_tokens=10, output_tokens=5, cache_read_input_tokens=90, cache_creation_input_tokens=0),
        )
        self.client = _FakeClient(msg)
        self.provider = AnthropicProvider(client=self.client)

    def test_request_and_response_mapping(self):
        history = [
            Message.user("hi"),
            Message("assistant", [
                ProviderBlock("anthropic", {"type": "thinking", "thinking": "", "signature": "s0"}, reasoning=True),
                ProviderBlock("other", {"type": "whatever"}),
                ToolCall("tu_0", "list_dir", {}),
            ]),
            Message("user", [ToolResultBlock("tu_0", "a\nb", is_error=False)]),
        ]
        resp = self.provider.complete(ModelRequest("sys", history, [ToolSpec("list_dir", "d", {"type": "object"})]))
        params = self.client.calls[0]
        self.assertEqual(params["model"], "claude-opus-5-5")
        self.assertEqual(params["thinking"], {"type": "adaptive"})
        self.assertEqual(params["output_config"], {"effort": "high"})
        self.assertEqual((params["fallbacks"], params["betas"]), ("default", ["server-side-fallback-2026-07-01"]))
        self.assertTrue(params["tools"][0]["eager_input_streaming"])
        self.assertEqual(params["messages"][1]["content"], [
            {"type": "thinking", "thinking": "", "signature": "s0"},
            {"type": "tool_use", "id": "tu_0", "name": "list_dir", "input": {}},
        ])
        self.assertEqual(params["messages"][2]["content"][0]["tool_use_id"], "tu_0")

        self.assertEqual(resp.stop_reason, StopReason.TOOL_USE)
        self.assertEqual(resp.usage.input_tokens, 100)
        self.assertTrue(resp.content[0].reasoning)
        self.assertEqual(resp.content[1], TextBlock("Let me look."))
        self.assertEqual(resp.tool_calls, [ToolCall("tu_1", "read_file", {"path": "a"})])

    def test_no_tools_key_when_no_tools(self):
        self.provider.complete(ModelRequest("sys", [Message.user("hi")]))
        self.assertNotIn("tools", self.client.calls[0])


if __name__ == "__main__":
    unittest.main()
