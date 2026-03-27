
import json

max_retry = 5
debug = True

import os

local_path = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(local_path, 'mcp_econ_tool.json')
with open(config_path, 'r') as config_file:
    server_configs = json.load(config_file)

    local_server_configs=server_configs


import asyncio

import threading
import queue

from typing import Dict

from env_service.base import BaseEnv
from env_service.registry import Registry
from env_service.environments.openworld.mcp_utils import MCPSessionHandler

from env_service.environments.openworld.tool_call_extract import extract_tool_calls

max_retry = 5

class AsyncLoopThread:
    def __init__(self):
        self.loop = None
        self.thread = threading.Thread(target=self._start_loop, daemon=True)
        self.tasks = queue.Queue()
        self.result_map = {}
        self._stop_event = threading.Event()
        self._worker_future = None
        self.thread.start()

        # 等待 loop 启动完成
        while self.loop is None:
            pass

    def _start_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._worker_future = asyncio.ensure_future(self._worker())
        try:
            self.loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            try:
                self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            self.loop.close()

    async def _worker(self):
        while not self._stop_event.is_set():
            func, request_id, args, kwargs = await self.loop.run_in_executor(None, self.tasks.get)
            try:
                future = asyncio.ensure_future(func(*args, **kwargs))
                result = await future
                self.result_map[request_id] = result
            except Exception as e:
                self.result_map[request_id] = e

    def run_async(self, func, *args, **kwargs):
        request_id = id(func) + id(args) + id(kwargs)
        self.tasks.put((func, request_id, args, kwargs))
        while request_id not in self.result_map:
            pass
        result = self.result_map.pop(request_id)
        if isinstance(result, Exception):
            raise result
        return result

    def shutdown(self):
        if self.loop is None or not self.loop.is_running():
            return
        self._stop_event.set()

        # 等待 _worker 退出
        def _stop():
            if self._worker_future:
                self._worker_future.cancel()
            self.loop.stop()

        self.loop.call_soon_threadsafe(_stop)
        if self.thread.is_alive():
            self.thread.join()



@Registry.register("openworld")
class OpenworldEnv(BaseEnv):
    def __init__(self, task_id: str = None, instance_id: str = None,
                 params={}):
        self.task_id = task_id
        self.instance_id = instance_id
        self.tool_info = None


        if 'server_configs_path' in params and os.path.exists(params['server_configs_path']):
            with open(params['server_configs_path'], 'r') as config_file:
                server_configs = json.load(config_file)

                self.server_configs =server_configs
        else:
            self.server_configs =  local_server_configs


        self.server_handler_list = {}
        self.tool_to_server = {}

        # 每个 Env 实例维护独立异步线程
        self.async_thread = AsyncLoopThread()

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

        # 同步阻塞初始化 tool_info
        if self.server_configs:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.tool_info = loop.run_until_complete(self._load_tool_info(self.server_configs))
            loop.close()


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
            for attempt in range(max_retry):
                try:
                    await handler.initialize()
                    tools = await handler.list_tools()
                    all_tools[server_name] = tools
                    await handler.cleanup()
                    break
                except Exception as e:
                    if attempt == max_retry - 1:
                        raise e
                    await asyncio.sleep(1)
        return all_tools

    async def _init_handlers(self):
        handlers = {}
        for server_name, config in self.server_configs["mcpServers"].items():
            for attempt in range(max_retry):
                try:
                    handler = MCPSessionHandler(name=server_name, config=config)
                    await handler.initialize()
                    handlers[server_name] = handler
                    break
                except Exception as e:
                    if attempt == max_retry - 1:
                        raise e
                    await asyncio.sleep(1)
        return handlers

    async def _call_tool(self, handler, tool_name, tool_args):
        return await handler.call_tool(tool_name=tool_name, arguments=tool_args)

    def get_init_state(self, params={}):
        query = '帮我查询一下宁德时代的股票，在今天是否值得购买' \
            if self.instance_id == '0' else '我想知道匠心家具最近为什么涨这么多，请帮我分析一下'

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.server_handler_list = loop.run_until_complete(self._init_handlers())
        loop.close()


        return {
            "state": [{"role": "system", "content": self.system_prompt},
                      {"role": "user", "content": query}],
            "info": {"instance_id": self.instance_id, "task_id": self.task_id},
        }

    def get_info(self, messages={}, params={}):

        return self.system_prompt

    def tool_call(self, action_msg, content_msg):
        is_terminated = False

        if action_msg and action_msg["tool_name"] in self.tool_to_server and self.server_handler_list:
            server_name = self.tool_to_server[action_msg["tool_name"]]
            try:
                tool_result = self.async_thread.run_async(
                    self._call_tool,
                    self.server_handler_list[server_name],
                    action_msg["tool_name"],
                    action_msg["tool_args"]
                )
                tool_results = "".join([txt.text for txt in tool_result.content]) if tool_result else "Tool returned no content."
            except Exception:
                tool_results = f'Failed to call tool {action_msg["tool_name"]} with params {action_msg["tool_args"]}'
        else:
            if not action_msg:
                name_in_msg = any(tool_name in content_msg for tool_name in self.tool_to_server)
                if name_in_msg:
                    tool_results = 'Tool name or arguments parsing failed. Please check format.'
                else:
                    tool_results = 'No useful tool call identified in the message.'
                    is_terminated = True
            else:
                is_terminated = True
                tool_results = f'Tool {action_msg["tool_name"]} not found or handler missing.'

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
                tool_results += f'tool call result of {tool_call["tool_name"]} is {result}\n'
        else:
            is_terminated, tool_results = self.tool_call(None, content_msg)

        reward = self.evaluate() if is_terminated else 0.0

        return {
            "state": [{"role": "user", "content": tool_results}],
            "reward": reward,
            "is_terminated": is_terminated,
            "info": {"instance_id": self.instance_id, "task_id": self.task_id},
        }

    def close(self):
        async def _cleanup_all():
            for handler in self.server_handler_list.values():
                await handler.cleanup()
        try:
            self.async_thread.run_async(_cleanup_all)
        except Exception:
            pass
        self.async_thread.shutdown()

    def evaluate(self, messages: Dict = {}, params={}) -> float:
        return 0.0

    @staticmethod
    def get_query_list(split: str = "train", params={}):
        return [1]




def main(debug=False):
    env = OpenworldEnv( instance_id='0')

    init_response = env.get_init_state()

    print(init_response)
    return init_response

if __name__ == '__main__':
    a=main(debug=False)
    print(a)

