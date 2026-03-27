from typing import Any, Dict, List

from loguru import logger
from omegaconf import DictConfig
from openai.types.chat.chat_completion import ChatCompletion
from verl import DataProto
from beyondagent.utils.vscode_breakpoint import vscode_conditional_breakpoint

class TokenAndProb:
    def __init__(self, token_id, logprob, decoded_string):
        self.token_id = token_id
        self.logprob = logprob
        self.decoded_string = decoded_string


