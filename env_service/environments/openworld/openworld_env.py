import asyncio
import json
import os
import queue
import threading
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, wait
from typing import Dict

from env_service.base import BaseEnv
from env_service.registry import Registry
from env_service.environments.openworld.mcp_utils import MCPSessionHandler
from env_service.environments.openworld.tool_call_extract import extract_tool_calls

local_path = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(local_path, "mcp_econ_tool.json")


def load_server_configs(params):
    """Resolve explicit servers before the unconfigured search default.

    Args:
        params: Environment parameters, optionally including server_configs_path.

    Returns:
        The selected MCP server configuration.
    """
    if "server_configs_path" in params:
        with open(params["server_configs_path"]) as config_file:
            return json.load(config_file)
    with open(config_path) as config_file:
        legacy = json.load(config_file)
    # Only the shipped, unconfigured finance entry is replaced. Preserve edits
    # to the legacy file, including credentials, extra servers and disabled maps.
    unconfigured_finance = {
        "mcpServers": {
            "阿里云百炼_上市公司财报数据精灵": {
                "type": "sse",
                "url": "https://dashscope.aliyuncs.com/api/v1/mcps/finance-spirit/sse",
                "headers": {"Authorization": ""},
            }
        }
    }
    if legacy != unconfigured_finance:
        return legacy
    with open(os.path.join(local_path, "mcp_search_tool.json")) as config_file:
        return json.load(config_file)


class AsyncLoopThread:
    """Run MCP initialization, calls and cleanup in one owning async task."""

    def __init__(self, cleanup):
        self.tasks = queue.Queue()
        self.cleanup = cleanup
        self.stopped = Future()
        self.thread = threading.Thread(target=self._start_loop, daemon=True)
        self.thread.start()

    def _start_loop(self):
        try:
            asyncio.run(self._worker())
        except BaseException as error:
            self.stopped.set_exception(error.with_traceback(None))
        else:
            self.stopped.set_result(None)

    async def _worker(self):
        try:
            while True:
                request = await asyncio.to_thread(self.tasks.get)
                if request is None:
                    return
                func, args, kwargs, future = request
                try:
                    # MCP transport cancel scopes must keep one owning task.
                    result = await func(*args, **kwargs)
                except asyncio.CancelledError as error:
                    if asyncio.current_task().cancelling():
                        raise  # Unwind a background transport failure first.
                    future.set_exception(error.with_traceback(None))
                except Exception as error:
                    # Do not expose this suspended worker frame to sync callers.
                    future.set_exception(error.with_traceback(None))
                else:
                    future.set_result(result)
        finally:
            # A background transport failure may cancel the worker while idle.
            # Always unwind connections here, in the task that entered them.
            try:
                await self.cleanup()
            finally:
                self.tasks.put(None)  # Release any pending queue reader thread.

    def run_async(self, func, *args, **kwargs):
        if self.stopped.done():
            raise RuntimeError("OpenWorld MCP worker is closed")
        future = Future()
        self.tasks.put((func, args, kwargs, future))
        wait((future, self.stopped), return_when=FIRST_COMPLETED)
        if future.done():
            return future.result()
        raise RuntimeError("OpenWorld MCP worker stopped") from self.stopped.exception()

    def shutdown(self):
        if self.thread.is_alive():
            self.tasks.put(None)
            self.thread.join()


@Registry.register("openworld")
class OpenworldEnv(BaseEnv):
    def __init__(self, task_id: str = None, instance_id: str = None, params={}):
        self.task_id = task_id
        self.instance_id = instance_id
        self.tool_info = None

        self.server_configs = load_server_configs(params)
        self.session_id = str(uuid.uuid4())

        self.server_handler_list = {}
        self.tool_to_server = {}

        # 每个 Env 实例维护独立异步线程
        self.async_thread = AsyncLoopThread(self._cleanup_handlers)

        self.system_prompt = """
        你具有以下工具，如果需要使用工具来辅助你回答问题，请按照下列格式，直接输出工具调用相关内容
        ```json 
        [
             {
                "tool_name": "工具名称字符串",
                "tool_args": {
                        "参数名1": "参数值1",
                        "参数名2": "参数值2"
                        // 具体参数根据工具定义填写，必须是有效的 JSON 格式，不允许注释或尾随逗号
                                }
            }
            // 如果需要调用多个工具，继续添加对象，数组中可以有任意多个工具调用
        ]
        ```
        ----------
        """

        try:
            self.tool_info = self.async_thread.run_async(
                self._load_tool_info, self.server_configs
            )
        except BaseException:
            self.close()
            raise

        for server_name in self.tool_info:
            for tool in self.tool_info[server_name]:
                self.tool_to_server[tool.name] = server_name
                self.system_prompt += f"""
                ###
                tool name： {tool.name}
                tool description: {tool.description}
                tool inputSchema：{str(tool.inputSchema)}
                ###
                """

    async def _load_tool_info(self, server_configs):
        all_tools = {}
        for server_name, config in server_configs["mcpServers"].items():
            handler = MCPSessionHandler(name=server_name, config=config)
            await handler.initialize()
            self.server_handler_list[server_name] = handler
            all_tools[server_name] = await handler.list_tools()
        return all_tools

    async def _call_tool(self, handler, tool_name, tool_args):
        arguments = dict(tool_args)
        endpoint = handler.config.get("url", "").split("?")[0]
        if endpoint == "https://search.parallel.ai/mcp" and tool_name in {
            "web_search",
            "web_fetch",
        }:
            # The environment instance owns the research conversation, even if
            # the model supplies an unrelated session_id in its tool arguments.
            arguments["session_id"] = self.session_id
        return await handler.call_tool(tool_name=tool_name, arguments=arguments)

    def get_init_state(self, params={}):
        query = (
            "帮我查询一下宁德时代的股票，在今天是否值得购买"
            if self.instance_id == "0"
            else "我想知道匠心家具最近为什么涨这么多，请帮我分析一下"
        )

        self.session_id = str(uuid.uuid4())
        query = params.get("query", query)

        return {
            "state": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query},
            ],
            "info": {"instance_id": self.instance_id, "task_id": self.task_id},
        }

    def get_info(self, messages={}, params={}):
        return self.system_prompt

    def tool_call(self, action_msg, content_msg):
        is_terminated = False

        if (
            action_msg
            and action_msg["tool_name"] in self.tool_to_server
            and self.server_handler_list
        ):
            server_name = self.tool_to_server[action_msg["tool_name"]]
            try:
                tool_result = self.async_thread.run_async(
                    self._call_tool,
                    self.server_handler_list[server_name],
                    action_msg["tool_name"],
                    action_msg["tool_args"],
                )
                tool_results = (
                    "".join([txt.text for txt in tool_result.content])
                    if tool_result
                    else "Tool returned no content."
                )
            except Exception:
                tool_results = f"Failed to call tool {action_msg['tool_name']} with params {action_msg['tool_args']}"
        else:
            if not action_msg:
                name_in_msg = any(
                    tool_name in content_msg for tool_name in self.tool_to_server
                )
                if name_in_msg:
                    tool_results = (
                        "Tool name or arguments parsing failed. Please check format."
                    )
                else:
                    tool_results = "No useful tool call identified in the message."
                    is_terminated = True
            else:
                is_terminated = True
                tool_results = (
                    f"Tool {action_msg['tool_name']} not found or handler missing."
                )

        return is_terminated, tool_results

    def step(self, action: {}, params={}):
        content_msg = action["content"]
        action_msg_list = extract_tool_calls(content_msg)

        tool_results = ""
        is_terminated = True

        if action_msg_list:
            for tool_call in action_msg_list:
                terminated, result = self.tool_call(tool_call, content_msg)
                is_terminated = is_terminated and terminated
                tool_results += (
                    f"tool call result of {tool_call['tool_name']} is {result}\n"
                )
        else:
            is_terminated, tool_results = self.tool_call(None, content_msg)

        reward = self.evaluate() if is_terminated else 0.0

        return {
            "state": [{"role": "user", "content": tool_results}],
            "reward": reward,
            "is_terminated": is_terminated,
            "info": {"instance_id": self.instance_id, "task_id": self.task_id},
        }

    async def _cleanup_handlers(self):
        for name in reversed(list(self.server_handler_list)):
            await self.server_handler_list.pop(name).cleanup()

    def close(self):
        self.async_thread.shutdown()

    def evaluate(self, messages: Dict = {}, params={}) -> float:
        return 0.0

    @staticmethod
    def get_query_list(split: str = "train", params={}):
        return [1]


def main(debug=False):
    env = OpenworldEnv(instance_id="0")

    init_response = env.get_init_state()

    print(init_response)
    return init_response


if __name__ == "__main__":
    a = main(debug=False)
    print(a)
