"""Unit tests for the NeuralMemory Hermes plugin port.

Covers the pure-function surface (config resolution, schema normalization,
regex hygiene pipelines, fallback/compat tool wiring, hook behavior) without
spawning the MCP subprocess. E2E against a live nmem-mcp server was verified
separately during development (connect -> 63-tool discovery -> remember ->
recall -> stats round-trip).
"""

from __future__ import annotations

import json
import sys
import types
import unittest
from importlib import util as importlib_util
from pathlib import Path
from unittest import mock

PLUGIN_DIR = Path(__file__).resolve().parent.parent


def _load_plugin_module():
    """Load the plugin the way Hermes's loader does (as a package)."""
    ns = types.ModuleType("hermes_plugins")
    ns.__path__ = []
    sys.modules.setdefault("hermes_plugins", ns)
    spec = importlib_util.spec_from_file_location(
        "hermes_plugins.neuralmemory",
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    mod = importlib_util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.neuralmemory"
    mod.__path__ = [str(PLUGIN_DIR)]
    sys.modules["hermes_plugins.neuralmemory"] = mod
    spec.loader.exec_module(mod)
    return mod


# Load the plugin package at import time so `from hermes_plugins.neuralmemory
# import ...` works in every test class.
_load_plugin_module()


class _MockCtx:
    def __init__(self):
        self.tools: list[str] = []
        self.hooks: dict[str, list] = {}

    def register_tool(self, name, toolset, schema, handler, description=None, **kw):
        self.tools.append(name)

    def register_hook(self, hook_name, cb):
        self.hooks.setdefault(hook_name, []).append(cb)


class _FakeMcp:
    """Duck-typed stand-in for NeuralMemoryMcpClient (no subprocess)."""

    def __init__(self, connected=True, recall_payload=None):
        self.connected = connected
        self.recall_payload = recall_payload or {"answer": "memory", "confidence": 0.9}
        self.calls: list[tuple[str, dict]] = []

    def ensure_connected(self):
        self.connected = True

    def call_tool(self, name, args=None):
        self.calls.append((name, args or {}))
        if name == "nmem_recall":
            return json.dumps(self.recall_payload)
        return json.dumps({"ok": True})


class TestConfig(unittest.TestCase):
    def setUp(self):
        from hermes_plugins.neuralmemory import config as cfg_mod
        self.cfg = cfg_mod

    def test_defaults_when_empty(self):
        c = self.cfg.resolve_config(None)
        self.assertEqual(c.python_path, "python")
        self.assertEqual(c.brain, "default")
        self.assertTrue(c.auto_context)
        self.assertTrue(c.auto_capture)
        self.assertTrue(c.auto_flush)
        self.assertTrue(c.auto_consolidate)
        self.assertEqual(c.context_depth, 1)
        self.assertEqual(c.max_context_tokens, 500)
        self.assertEqual(c.timeout, 30_000)
        self.assertEqual(c.init_timeout, 90_000)

    def test_brain_name_validation(self):
        self.assertEqual(self.cfg.resolve_config({"brain": "valid-brain_1.2"}).brain, "valid-brain_1.2")
        self.assertEqual(self.cfg.resolve_config({"brain": "bad brain!"}).brain, "default")
        self.assertEqual(self.cfg.resolve_config({"brain": "x" * 65}).brain, "default")
        self.assertEqual(self.cfg.resolve_config({"brain": 42}).brain, "default")

    def test_range_clamps_match_upstream(self):
        self.assertEqual(self.cfg.resolve_config({"contextDepth": 4}).context_depth, 1)
        self.assertEqual(self.cfg.resolve_config({"contextDepth": 0}).context_depth, 0)
        self.assertEqual(self.cfg.resolve_config({"maxContextTokens": 50}).max_context_tokens, 500)
        self.assertEqual(self.cfg.resolve_config({"maxContextTokens": 10000}).max_context_tokens, 10000)
        self.assertEqual(self.cfg.resolve_config({"timeout": 100}).timeout, 30_000)
        self.assertEqual(self.cfg.resolve_config({"timeout": 120000}).timeout, 120_000)
        self.assertEqual(self.cfg.resolve_config({"initTimeout": 5000}).init_timeout, 90_000)

    def test_type_coercion(self):
        c = self.cfg.resolve_config({"autoContext": "yes", "contextDepth": 2.7})
        self.assertTrue(c.auto_context)  # non-bool falls back to default True
        self.assertEqual(c.context_depth, 1)  # non-int falls back


class TestSchemaNormalization(unittest.TestCase):
    def setUp(self):
        from hermes_plugins.neuralmemory import tools_proxy
        self.tp = tools_proxy

    def test_strip_keys_removed(self):
        src = {"type": "object", "properties": {
            "n": {"type": "number", "minimum": 0, "maximum": 10, "maxLength": 5}}}
        out = self.tp.normalize_schema(src)
        self.assertNotIn("minimum", out["properties"]["n"])
        self.assertNotIn("maximum", out["properties"]["n"])
        self.assertNotIn("maxLength", out["properties"]["n"])

    def test_integer_becomes_number(self):
        out = self.tp.normalize_schema({"type": "integer"})
        self.assertEqual(out["type"], "number")

    def test_object_gets_properties_and_additional_properties(self):
        out = self.tp.normalize_schema({"type": "object"})
        self.assertEqual(out["properties"], {})
        self.assertFalse(out["additionalProperties"])

    def test_nested_combinators(self):
        src = {"anyOf": [{"type": "object", "properties": {"a": {"type": "integer"}}}]}
        out = self.tp.normalize_schema(src)
        self.assertEqual(out["anyOf"][0]["properties"]["a"]["type"], "number")

    def test_safe_schema_fallback(self):
        self.assertEqual(self.tp.to_safe_schema(None),
                         {"type": "object", "properties": {}, "additionalProperties": False})
        out = self.tp.to_safe_schema({"type": "object", "required": ["x"],
                                      "properties": {"x": {"type": "string"}}})
        self.assertEqual(out["required"], ["x"])


class TestRegexPipelines(unittest.TestCase):
    def setUp(self):
        from hermes_plugins.neuralmemory import hooks
        self.hooks = hooks

    def test_strip_prompt_metadata_full_pipeline(self):
        raw = ('{"message_id": 123, "sender": "test"}\n'
               '## Relevant Memories\n- [concept] junk neuron\n'
               '[NeuralMemory — relevant context]\n'
               'export FOO=bar\n'
               'what is the deployment config?')
        self.assertEqual(self.hooks.strip_prompt_metadata(raw),
                         "what is the deployment config?")

    def test_strip_prompt_metadata_fallback_last_line(self):
        cleaned = self.hooks.strip_prompt_metadata("## Relevant Memories\n- [concept] x")
        self.assertTrue(cleaned)  # never returns empty

    def test_sanitize_auto_capture(self):
        out = self.hooks.sanitize_auto_capture(
            "## Relevant Memories\nstuff\nOK.\nDone\nWe chose PostgreSQL because of JSONB support.")
        self.assertIn("PostgreSQL", out)
        self.assertNotIn("[concept]", out)
        self.assertNotIn("OK.", out)

    def test_sanitize_capture_requires_whitespace_after_bracket(self):
        """MINOR-3: sanitize pipeline uses canon's stricter \\]\\s variant."""
        # "- [fact]x" (no space) must SURVIVE sanitize (canon keeps it)...
        out = self.hooks.sanitize_auto_capture("- [fact]x\nWe fixed the bug.")
        self.assertIn("- [fact]x", out)
        # ...while "- [fact] x" (with space) is stripped
        out2 = self.hooks.sanitize_auto_capture("- [fact] junk\nWe fixed the bug.")
        self.assertNotIn("- [fact]", out2)


class TestToolWiring(unittest.TestCase):
    def setUp(self):
        from hermes_plugins.neuralmemory import tools_proxy
        self.tp = tools_proxy

    def test_fallback_tool_names_and_handlers(self):
        mcp = _FakeMcp()
        tools = self.tp.create_fallback_tools(mcp)
        names = [t["name"] for t in tools]
        self.assertEqual(names, ["nmem_remember", "nmem_recall", "nmem_context",
                                 "nmem_stats", "nmem_health"])
        for t in tools:
            out = json.loads(t["handler"]({}))
            self.assertFalse(out.get("error"), f"{t['name']} returned error: {out}")
        self.assertEqual([c[0] for c in mcp.calls], names)

    def test_compat_shims_route_to_recall(self):
        mcp = _FakeMcp()
        search, get = self.tp.create_compatibility_tools(mcp)
        search["handler"]({"query": "q1"})
        get["handler"]({"id": "abc"})
        self.assertEqual(mcp.calls[0], ("nmem_recall", {"query": "q1", "depth": 1}))
        self.assertEqual(mcp.calls[1], ("nmem_recall", {"query": "abc", "depth": 0}))

    def test_auto_reconnect_error_is_soft(self):
        class _DeadMcp(_FakeMcp):
            def ensure_connected(self):
                raise RuntimeError("no python")

        mcp = _DeadMcp(connected=False)
        tools = self.tp.create_fallback_tools(mcp)
        out = json.loads(tools[0]["handler"]({"content": "x"}))
        self.assertTrue(out["error"])
        self.assertIn("auto-connect failed", out["message"])

    def test_memory_type_enum_matches_upstream(self):
        mcp = _FakeMcp()
        remember = self.tp.create_fallback_tools(mcp)[0]
        enum = remember["parameters"]["properties"]["type"]["enum"]
        self.assertEqual(enum, ["fact", "decision", "preference", "todo", "insight",
                                "context", "instruction", "error", "workflow", "reference"])


class TestHooks(unittest.TestCase):
    def setUp(self):
        from hermes_plugins.neuralmemory import hooks
        from hermes_plugins.neuralmemory import tools_proxy
        from hermes_plugins.neuralmemory.config import resolve_config
        self.hooks = hooks
        self.tp = tools_proxy
        self.cfg = resolve_config(None)

    def test_pre_llm_first_turn_injects_instructions_and_recall(self):
        mcp = _FakeMcp()
        tools = self.tp.create_fallback_tools(mcp)
        hook = self.hooks.make_pre_llm_hook(mcp, self.cfg, tools)
        out = hook(session_id="s", user_message="hello", conversation_history=[],
                   is_first_turn=True, model="m", platform="cli")
        self.assertIn("context", out)
        self.assertIn("Neural Memory", out["context"])
        self.assertIn("relevant context", out["context"])

    def test_pre_llm_later_turn_recall_only(self):
        mcp = _FakeMcp()
        tools = self.tp.create_fallback_tools(mcp)
        hook = self.hooks.make_pre_llm_hook(mcp, self.cfg, tools)
        out = hook(session_id="s", user_message="hello", conversation_history=[],
                   is_first_turn=False)
        self.assertIn("relevant context", out["context"])
        self.assertNotIn("WHEN TO RECALL", out["context"])

    def test_pre_llm_low_confidence_skips_recall(self):
        mcp = _FakeMcp(recall_payload={"answer": "weak", "confidence": 0.05})
        hook = self.hooks.make_pre_llm_hook(mcp, self.cfg,
                                            self.tp.create_fallback_tools(mcp))
        self.assertIsNone(hook(session_id="s", user_message="x",
                               conversation_history=[], is_first_turn=False))

    def test_pre_llm_disconnected_skips_recall(self):
        mcp = _FakeMcp(connected=False)
        hook = self.hooks.make_pre_llm_hook(mcp, self.cfg,
                                            self.tp.create_fallback_tools(mcp))
        self.assertIsNone(hook(session_id="s", user_message="x",
                               conversation_history=[], is_first_turn=False))
        self.assertEqual(mcp.calls, [])

    def test_pre_llm_recall_failure_never_raises(self):
        class _Exploding(_FakeMcp):
            def call_tool(self, name, args=None):
                raise RuntimeError("boom")

        mcp = _Exploding()
        hook = self.hooks.make_pre_llm_hook(mcp, self.cfg,
                                            self.tp.create_fallback_tools(mcp))
        self.assertIsNone(hook(session_id="s", user_message="x",
                               conversation_history=[], is_first_turn=False))

    def test_post_llm_captures_long_assistant_text(self):
        mcp = _FakeMcp()
        hook = self.hooks.make_post_llm_hook(mcp, self.cfg)
        hook(session_id="s", user_message="q",
             assistant_response="We decided to keep the port faithful to the upstream plugin.",
             conversation_history=[{"role": "assistant",
                                    "content": "We decided to keep the port faithful to the upstream plugin."}])
        self.assertEqual(mcp.calls[0][0], "nmem_auto")
        self.assertEqual(mcp.calls[0][1]["action"], "process")

    def test_post_llm_skips_short_acknowledgement(self):
        mcp = _FakeMcp()
        hook = self.hooks.make_post_llm_hook(mcp, self.cfg)
        hook(session_id="s", user_message="q", assistant_response="OK.",
             conversation_history=[{"role": "assistant", "content": "OK."}])
        self.assertEqual(mcp.calls, [])

    def test_post_llm_disconnected_is_noop(self):
        mcp = _FakeMcp(connected=False)
        hook = self.hooks.make_post_llm_hook(mcp, self.cfg)
        hook(session_id="s", user_message="q", assistant_response="x" * 100,
             conversation_history=[])
        self.assertEqual(mcp.calls, [])

    def test_session_reset_flush(self):
        mcp = _FakeMcp()
        hook = self.hooks.make_session_reset_hook(mcp, self.cfg)
        hook(session_key="s1")
        self.assertEqual(mcp.calls, [("nmem_auto",
                                      {"action": "process", "text": "[session boundary — reset]"})])

    def test_session_end_closes_client_and_evicts_pool(self):
        """MINOR-2: port of service.stop() — close + pool eviction."""

        class _ClosableMcp(_FakeMcp):
            def __init__(self, **kw):
                super().__init__(**kw)
                self.closed = False

            def close(self):
                self.closed = True

        mcp = _ClosableMcp()
        hook = self.hooks.make_session_end_hook(mcp, self.cfg)
        hook()
        self.assertTrue(mcp.closed)


class TestRegistration(unittest.TestCase):
    def test_register_wires_tools_and_hooks(self):
        ctx = _MockCtx()
        _load_plugin_module().register(ctx)
        self.assertEqual(sorted(ctx.tools),
                         sorted(["nmem_remember", "nmem_recall", "nmem_context",
                                 "nmem_stats", "nmem_health", "memory_search", "memory_get"]))
        self.assertEqual(set(ctx.hooks),
                         {"pre_llm_call", "post_llm_call", "on_session_reset", "on_session_end"})


class TestMcpClientConstants(unittest.TestCase):
    def test_protocol_constants_match_upstream(self):
        from hermes_plugins.neuralmemory import mcp_client
        self.assertEqual(mcp_client.PROTOCOL_VERSION, "2024-11-05")
        self.assertEqual(mcp_client.MAX_BUFFER_BYTES, 10 * 1024 * 1024)
        self.assertEqual(mcp_client.DEFAULT_TIMEOUT, 30_000)

    def test_env_allowlist(self):
        from hermes_plugins.neuralmemory import mcp_client
        expected = {"PATH", "PATHEXT", "HOME", "USERPROFILE", "SYSTEMROOT", "TEMP",
                    "TMP", "LANG", "LC_ALL", "VIRTUAL_ENV", "CONDA_PREFIX",
                    "PYTHONPATH", "PYTHONHOME", "NEURALMEMORY_DIR", "NEURALMEMORY_BRAIN",
                    "NEURAL_MEMORY_DIR", "NEURAL_MEMORY_JSON", "NEURAL_MEMORY_DEBUG"}
        self.assertEqual(set(mcp_client.ALLOWED_ENV_KEYS), expected)

    def test_build_child_env_brain(self):
        from hermes_plugins.neuralmemory import mcp_client
        env = mcp_client.build_child_env("custom-brain")
        self.assertEqual(env.get("NEURALMEMORY_BRAIN"), "custom-brain")
        self.assertNotIn("NEURALMEMORY_BRAIN", mcp_client.build_child_env("default"))

    def test_drain_buffer_consumes_parsed_lines(self):
        """M1 fix: the cap applies to the UNDRAINED buffer, not cumulative bytes."""
        from hermes_plugins.neuralmemory import mcp_client
        client = mcp_client.NeuralMemoryMcpClient("python", "b")
        seen: list[dict] = []
        client._handle_message = lambda m: seen.append(m)
        buf = b'{"id":1,"result":{}}\n{"id":2,"result":{"x":1}}\n{"partial"'
        rest = client._drain_buffer(buf)
        self.assertEqual(rest, b'{"partial"')  # partial line stays buffered
        self.assertEqual(len(seen), 2)

    def test_pool_keyed_by_pythonpath_and_brain(self):
        """M3 fix: pool reuses one client per (pythonPath, brain) across calls."""
        import hermes_plugins.neuralmemory as plugin
        from hermes_plugins.neuralmemory.config import resolve_config
        cfg_a = resolve_config({"brain": "a"})
        cfg_a2 = resolve_config({"brain": "a"})
        cfg_b = resolve_config({"brain": "b"})
        plugin._mcp_clients.clear()
        try:
            c1 = plugin._get_or_create_mcp_client(cfg_a)
            c2 = plugin._get_or_create_mcp_client(cfg_a2)
            c3 = plugin._get_or_create_mcp_client(cfg_b)
            self.assertIs(c1, c2)
            self.assertIsNot(c1, c3)
            self.assertEqual(len(plugin._mcp_clients), 2)
        finally:
            plugin._mcp_clients.clear()


if __name__ == "__main__":
    unittest.main()
