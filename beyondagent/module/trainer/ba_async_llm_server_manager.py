import asyncio
from typing import Any, Dict, List

# from verl.workers.rollout.async_server import AsyncLLMServerManager
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, AgentLoopManager
from verl.experimental.agent_loop.tool_parser import ToolParser

class BaAsyncLLMServerManager(AgentLoopManager):

    def submit_chat_completions(
            self,
            messages: List[Dict[str, str]],
            sampling_params: Dict[str, Any],
            request_id: str = None,
    ):
        """Submit a chat completion request to chat scheduler and wait until it is done.
        To submit multiple requests in parallel, please use `generate_sequences` instead.

        Args: same as ChatCompletionScheduler.submit_chat_completions.
        """
        assert self.chat_scheduler is not None, "chat scheduler is not initialized."
        future = asyncio.run_coroutine_threadsafe(
            self.chat_scheduler._submit_chat_completions_semaphore(
                messages=messages,
                request_id=request_id,
                sampling_params=sampling_params,
            ),
            self.chat_scheduler_loop,
        )
        future.result()
