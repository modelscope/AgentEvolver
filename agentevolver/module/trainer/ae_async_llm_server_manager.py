import asyncio
from typing import Any, Dict, List, Optional

from verl.workers.rollout.async_server import AsyncLLMServerManager


class BaAsyncLLMServerManager(AsyncLLMServerManager):
    
    def chat(self, messages: list[dict[str, str]], sampling_params: dict[str, Any]) -> str:
        """todo"""
        self.submit_chat_completions(messages.copy(), sampling_params)
        return messages[-1]['content']

    def submit_chat_completions(
            self,
            messages: List[Dict[str, str]],
            sampling_params: Dict[str, Any],
            request_id: Optional[str] = None,
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
