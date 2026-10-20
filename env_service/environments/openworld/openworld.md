# OpenWorld MCP environment

OpenWorld exposes MCP tools to AgentEvolver agents. An unconfigured installation
uses [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
for free web search and page fetching, without an account or API key. Anonymous
search uses Fast mode. The free service has usage limits; tool errors are returned
to the agent so it can stop or adjust its research.

Install the environment service dependencies in your OpenWorld environment:

```bash
pip install -r env_service/environments/openworld/requirements.txt
python -m env_service.env_service --env openworld --portal 127.0.0.1 --port 8080
```

Create an `openworld` instance through EnvService. Its initial system message
contains the discovered tools and their schemas. You can set `params.query` in
`get_init_state` to replace the built-in sample finance question. Agents request
tools using a JSON array in a fenced `json` block:

````text
```json
[
  {
    "tool_name": "web_search",
    "tool_args": {
      "objective": "Find recent research about self-evolving agents",
      "search_queries": ["self evolving agent research"]
    }
  }
]
```
````

Use source URLs from the returned state in a `web_fetch` call with a `urls` array.
OpenWorld returns tool text in the next user message. The environment owns one
opaque `session_id` across related Parallel searches, fetches and retries. Calling
`get_init_state` starts a new research conversation and creates a fresh ID.
Model-supplied IDs do not override it. Each environment instance has independent
connections and conversation state. Call `close` (or release the EnvService
instance) to close those connections.

## Existing and custom servers

`params.server_configs_path` selects a JSON file in the **environment service's
filesystem**, with the usual `{"mcpServers": {...}}` structure. Explicit selections
replace the default entirely, including headers, transport and timeout settings.
A missing or invalid file raises an error; OpenWorld does not switch providers
after an explicit configuration fails. To disable all external tools, select a
file containing `{"mcpServers": {}}`.

The bundled `mcp_econ_tool.json` remains available for finance tools. If you already
edited that file (for example, to supply its Authorization header), OpenWorld
keeps using it. Only the original finance entry with its empty Authorization
header selects the new search default. To explicitly select finance, pass
`server_configs_path` pointing to that file. The memory server in `mcp_tool.json`
is also available through the same parameter.

The search default is in `mcp_search_tool.json`, using Streamable HTTP at
`https://search.parallel.ai/mcp`. To customize it, copy it and select the copy with
`server_configs_path`. For higher limits, you can add an `Authorization: Bearer
YOUR_KEY` header from [Parallel](https://platform.parallel.ai). No environment API
key is implicitly read or added to anonymous requests. Preserve the project's
User-Agent header when using the bundled search configuration.

OpenWorld's tool parser is in `tool_call_extract.py`; its prompt can be customized
in `openworld_env.py`. Evaluation still returns `0.0`: this environment supplies
tools and sample queries, not a research-quality reward function.

Run the focused resolver, SDK transport and dispatcher tests without the training
stack:

```bash
python -m unittest discover -s tests -p test_openworld_search.py -v
```
