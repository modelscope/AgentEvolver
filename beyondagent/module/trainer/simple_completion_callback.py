from typing import Any, Dict, List

from loguru import logger
from omegaconf import DictConfig
from openai.types.chat.chat_completion import ChatCompletion
from verl import DataProto
from verl.workers.rollout.chat_scheduler import CompletionCallback

class TokenAndProb:
    def __init__(self, t):
        # print(t)
        # ChatCompletionTokenLogprob(token='token_id:73594', bytes=[96, 96, 96], logprob=-1.9073468138230965e-06, top_logprobs=[])
        self.token_id = int(t.token.split('token_id:')[-1])
        self.logprob = t.logprob
        try:
            self.decoded_string = bytes(t.bytes).decode('utf-8')
        except:
            self.decoded_string = '<cannot decode>' + str(t.bytes)


class SimpleCompletionCallback(CompletionCallback):
    def __init__(self, config: DictConfig, scheduler: "ChatCompletionScheduler"):
        super().__init__(config, scheduler)
        logger.info("=" * 10 + "SimpleCompletionCallback is inited~" + "=" * 10)

    async def __call__(self, messages: List[Dict[str, str]], completions: ChatCompletion, info: Dict[str, Any]):
        message = completions.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
        if "content" not in message:
            message["content"] = "vllm failed"

        if message['content'] == '' or completions.choices[0].finish_reason != 'stop':
            logger.warning(str(completions.choices[0].finish_reason))
            logger.bind(bad_case=True).error('empty content or non-stop finish reason')
            logger.bind(bad_case=True).error(str(completions.choices[0]))
            message['content'] == 'im_end'  # 遇到致命错误，填补一个token

        t = {"role": message["role"], "request_id":completions.id, "content": message['content'], "tokens": [TokenAndProb(t) for t in completions.choices[0].logprobs.content]}
        messages.append(t)

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
