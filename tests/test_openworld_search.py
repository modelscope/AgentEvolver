"""Exercise OpenWorld's search resolver and dispatcher through the MCP SDK."""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from mcp.client.streamable_http import streamablehttp_client

from env_service.environments.openworld import mcp_utils, openworld_env
from env_service.registry import Registry


class SearchMCPFixture:
    """Serve deterministic JSON-RPC responses at the HTTP transport boundary."""

    def __init__(self):
        self.requests = []
        self.fail_once = False
        self.transport_error = None

    def respond(self, request):
        payload = json.loads(request.content) if request.content else {}
        self.requests.append((request, payload))
        if request.method == "GET":
            return httpx.Response(405)
        if "id" not in payload:
            return httpx.Response(202)
        method = payload["method"]
        if self.transport_error == method:
            raise httpx.ReadTimeout("fixture transport timeout", request=request)
        if method == "initialize":
            result = {
                "protocolVersion": payload["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "search-fixture", "version": "1"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": name,
                        "description": "Search or fetch source pages",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                    for name in ("web_search", "web_fetch")
                ]
            }
        elif method == "tools/call":
            args = payload["params"]["arguments"]
            if self.fail_once:
                self.fail_once = False
                return httpx.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": payload["id"],
                        "error": {"code": -32000, "message": "temporary failure"},
                    },
                )
            if args.get("objective") == "quota":
                text = "Free-tier rate limit reached"
            else:
                text = json.dumps(
                    {
                        "url": "https://example.org/source?query=agent#details",
                        "excerpt": "Source evidence for the agent's next turn.",
                    }
                )
            result = {
                "content": [{"type": "text", "text": text}],
                "isError": args.get("objective") == "quota",
            }
        else:
            raise AssertionError(method)
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result}
        )

    def client(self, headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            auth=auth,
            transport=httpx.MockTransport(self.respond),
        )

    def transport(self, **kwargs):
        return streamablehttp_client(**kwargs, httpx_client_factory=self.client)

    @property
    def calls(self):
        return [
            p["params"] for _, p in self.requests if p.get("method") == "tools/call"
        ]


def action(name="web_search", **arguments):
    return {
        "role": "assistant",
        "content": "```json\n"
        + json.dumps([{"tool_name": name, "tool_args": arguments}])
        + "\n```",
    }


class OpenworldSearchTest(unittest.TestCase):
    def setUp(self):
        self.fixture = SearchMCPFixture()
        self.patch = patch.object(
            mcp_utils, "streamablehttp_client", self.fixture.transport
        )
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def config(self, servers):
        path = Path(self.temp.name) / (
            str(len(list(Path(self.temp.name).iterdir()))) + ".json"
        )
        path.write_text(json.dumps({"mcpServers": servers}))
        return str(path)

    def env(self, params=None):
        env = Registry.get("openworld")(instance_id="research", params=params or {})
        self.addCleanup(env.close)
        return env

    def test_unconfigured_search_fetch_and_followup_consume_source_url(self):
        # Clear unrelated credentials; the production resolver does not read them.
        with patch.dict("os.environ", {}, clear=True):
            env = self.env()
            initial = env.get_init_state({"query": "Find agent research sources"})
            self.assertEqual(
                initial["state"][-1]["content"], "Find agent research sources"
            )
            self.assertIn("web_search", initial["state"][0]["content"])
            first = env.step(
                action(
                    objective="Agent research",
                    search_queries=["agent research sources"],
                )
            )
            self.assertFalse(first["is_terminated"])
            # A scripted agent turn reads the tool result and passes its source URL
            # into the next tool call, using the ordinary model-output parser.
            source = json.loads(first["state"][0]["content"].split(" is ", 1)[1])
            fetched = env.step(action("web_fetch", urls=[source["url"]]))
            self.assertIn("Source evidence", fetched["state"][0]["content"])
            env.step(
                action(objective="Follow-up", search_queries=["agent source followup"])
            )
            self.assertEqual(
                self.fixture.calls[1]["arguments"]["urls"], [source["url"]]
            )
            ids = {call["arguments"]["session_id"] for call in self.fixture.calls}
            self.assertEqual(ids, {env.session_id})
            for request, _ in self.fixture.requests:
                self.assertEqual(str(request.url), "https://search.parallel.ai/mcp")
                self.assertEqual(
                    request.headers["user-agent"],
                    "AgentEvolver/0.1.0 (+https://github.com/modelscope/AgentEvolver)",
                )
                self.assertNotIn("authorization", request.headers)
                self.assertNotIn("x-api-key", request.headers)
            self.assertTrue(
                env.step({"content": "Finished research."})["is_terminated"]
            )
        env.close()
        self.assertFalse(env.async_thread.thread.is_alive())

    def test_interleaved_conversations_reset_and_retry_keep_ownership(self):
        first, second = self.env(), self.env()
        first.get_init_state()
        second.get_init_state()
        first.step(action(objective="First", session_id="model-chosen-id"))
        second.step(action(objective="Second"))
        first.step(action("web_fetch", urls=["https://example.org"]))
        self.assertNotEqual(first.session_id, second.session_id)
        self.assertEqual(
            [c["arguments"]["session_id"] for c in self.fixture.calls],
            [first.session_id, second.session_id, first.session_id],
        )
        old_id = first.session_id
        self.fixture.fail_once = True
        first.step(action(objective="Retry"))
        self.assertEqual(
            [c["arguments"]["session_id"] for c in self.fixture.calls[-2:]],
            [old_id, old_id],
        )
        quota = first.step(action(objective="quota"))
        self.assertIn("rate limit", quota["state"][0]["content"])
        self.assertEqual(first.session_id, old_id)
        first.get_init_state()
        first.step(action(objective="New conversation"))
        self.assertNotEqual(first.session_id, old_id)
        self.assertEqual(
            self.fixture.calls[-1]["arguments"]["session_id"], first.session_id
        )

    def test_custom_servers_credentials_and_disabled_maps_are_preserved(self):
        custom = {
            "custom": {
                "type": "streamable_http",
                "url": "https://custom.example/mcp",
                "headers": {
                    "Authorization": "Bearer configured",
                    "User-Agent": "custom-agent",
                },
            }
        }
        env = self.env({"server_configs_path": self.config(custom)})
        env.step(action(objective="Custom", session_id="explicit-session"))
        self.assertEqual(env.server_configs["mcpServers"], custom)
        self.assertEqual(
            self.fixture.calls[-1]["arguments"]["session_id"], "explicit-session"
        )
        self.assertEqual(
            self.fixture.requests[-1][0].headers["authorization"], "Bearer configured"
        )
        self.assertEqual(
            str(self.fixture.requests[-1][0].url), "https://custom.example/mcp"
        )
        before = len(self.fixture.requests)
        disabled = self.env({"server_configs_path": self.config({})})
        self.assertNotIn("web_search", disabled.get_init_state()["state"][0]["content"])
        self.assertTrue(disabled.step(action())["is_terminated"])
        self.assertEqual(len(self.fixture.requests), before)
        with self.assertRaises(FileNotFoundError):
            self.env({"server_configs_path": "/does/not/exist.json"})
        # Edits to the old default (credentials, servers, or an empty map) are
        # user configuration and must not acquire a new external search provider.
        for servers in (custom, {}):
            with patch.object(openworld_env, "config_path", self.config(servers)):
                saved = self.env()
                self.assertEqual(saved.server_configs["mcpServers"], servers)

    def test_unknown_tool_is_not_dispatched_and_explicit_failure_does_not_fallback(
        self,
    ):
        env = self.env()
        before = len(self.fixture.calls)
        self.assertTrue(env.step(action("not_allowed"))["is_terminated"])
        self.assertEqual(len(self.fixture.calls), before)
        custom = {
            "broken": {"type": "streamable_http", "url": "https://broken.example/mcp"}
        }
        with patch.object(
            mcp_utils.MCPSessionHandler,
            "initialize",
            side_effect=ValueError("bad config"),
        ):
            with self.assertRaisesRegex(ValueError, "bad config"):
                self.env({"server_configs_path": self.config(custom)})

    def test_timeout_error_and_cancellation_through_dispatcher(self):
        env = self.env()
        handler = next(iter(env.server_handler_list.values()))
        for error in (TimeoutError("timed out"), RuntimeError("failed")):
            with patch.object(handler.session, "call_tool", side_effect=error):
                result = env.step(action(objective="Error"))
                self.assertIn("Failed to call tool", result["state"][0]["content"])
        session_id = env.session_id
        with patch.object(
            handler.session, "call_tool", side_effect=asyncio.CancelledError
        ):
            with self.assertRaises(asyncio.CancelledError):
                env.step(action(objective="Cancelled"))
        self.assertEqual(env.session_id, session_id)
        self.assertIn(
            "Source evidence",
            env.step(action(objective="Continue"))["state"][0]["content"],
        )

    def test_custom_stdio_connection_preserves_server_state(self):
        script = Path(__file__).parent / "fixtures" / "openworld_stdio.py"
        servers = {"memory": {"command": sys.executable, "args": [str(script)]}}
        env = self.env({"server_configs_path": self.config(servers)})
        env.get_init_state()
        env.step(action("remember", value="first"))
        result = env.step(action("remember", value="second"))
        self.assertIn("first,second", result["state"][0]["content"])
        self.assertEqual(len(self.fixture.requests), 0)
        env.close()
        self.assertFalse(env.async_thread.thread.is_alive())

    def test_real_sdk_transport_failure_unwinds_without_hanging(self):
        self.fixture.transport_error = "initialize"
        with self.assertRaises((Exception, asyncio.CancelledError)):
            self.env()
        initializations = [
            p for _, p in self.fixture.requests if p.get("method") == "initialize"
        ]
        self.assertEqual(len(initializations), 1)
        self.fixture.transport_error = None
        env = self.env()
        self.fixture.transport_error = "tools/call"
        failed = env.step(action(objective="Transport timeout"))
        self.assertIn("Failed to call tool", failed["state"][0]["content"])
        env.close()
        self.assertFalse(env.async_thread.thread.is_alive())

    def test_multi_server_sessions_share_one_owner_and_cleanup_in_reverse(self):
        servers = {
            name: {"type": "streamable_http", "url": f"https://{name}.example/mcp"}
            for name in ("first", "second")
        }
        owner_tasks = []
        original = mcp_utils.MCPSessionHandler.cleanup

        async def cleanup(handler):
            owner_tasks.append((handler.name, asyncio.current_task()))
            await original(handler)

        with patch.object(mcp_utils.MCPSessionHandler, "cleanup", cleanup):
            env = self.env({"server_configs_path": self.config(servers)})
            env.step(action(objective="Second server"))
            env.close()
        self.assertEqual([name for name, _ in owner_tasks], ["second", "first"])
        self.assertIs(owner_tasks[0][1], owner_tasks[1][1])
        self.assertFalse(env.async_thread.thread.is_alive())


if __name__ == "__main__":
    unittest.main()
