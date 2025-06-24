import logging
from typing import Any, Dict, List

from omegaconf import DictConfig
from openai.types.chat.chat_completion import ChatCompletion
from verl import DataProto
from verl.workers.rollout.chat_scheduler import CompletionCallback

logger = logging.getLogger(__file__)


class SimpleCompletionCallback(CompletionCallback):
    def __init__(self, config: DictConfig, scheduler: "ChatCompletionScheduler"):
        super().__init__(config, scheduler)
        logger.info("=" * 10 + "SimpleCompletionCallback is inited~" + "=" * 10)
        # TODO: add reward manager to calculate reward score once a sample finish

    async def __call__(self, messages: List[Dict[str, str]], completions: ChatCompletion, info: Dict[str, Any]):
        message = completions.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
        if "content" not in message:
            message["content"] = ""
        message["request_id"] = completions.id
        messages.append(message)
        # finish_reason = completions.choices[0].finish_reason

    def postprocess(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], n: int) -> DataProto:
        """Post process batch data.

        Args:
            batch: Batch input messages from RLHFDataset.
            batch_conversations: List of messages including raw prompt, assistant response, tool response.
                Note that `len(batch_conversations) == len(batch) * n`, e.g n=2,
                batch_conversations=[messages_0_0, messages_0_1, messages_1_0, messages_1_1, ...]
            n: How many chat completion choices to generate for each input message.

        Returns:
            Batch data, should include ["prompts", "responses", "response_mask", "input_ids", "attention_mask", "position_ids"].
        """
        raise NotImplementedError
