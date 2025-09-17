import torch
import copy
from typing import List
from beyondagent.schema.trajectory import Sample
from best_logger import print_dict, print_listofdict
from beyondagent.module.context_manager.cmt_linear import ExtendedMessage, Linear_CMT
from beyondagent.module.context_manager.cmt_linear import find_sublist_indices, replace_token_ids

class LinearThinkCMT(Linear_CMT):
    """
    A linear context manager template that handles the conversation flow between LLM and environment.
    This class manages the context window, tokenization, and message history in a linear fashion.

    Attributes:
        config: Configuration object containing environment and model settings
        tokenizer: Tokenizer instance for processing text
        full_context (List[ExtendedMessage]): List of all messages in the conversation
        current_context_status (str): Current status of the context
        max_seq_length (int): Maximum sequence length for the context window
        max_env_output_length (int): Maximum length for environment outputs
        terminal_rewards_dict (dict): Dictionary storing terminal rewards

    1. prepare_next_llm_context
    2. check_context_token_num_safe
    3. prepare_world_interaction
    4. save_init_input
    5. save_llm_output
    6. save_env_output
    7. remove_last_context
    8. generate_log
    9. group_tokenize
    """

    def __init__(self, config, tokenizer, contain_phantom_hint=False, past_trajectory=None):
        """
        Initializes the LinearThinkCMT with the given configuration, tokenizer, and optional parameters.

        Args:
            config: Configuration object containing environment and model settings.
            tokenizer: Tokenizer instance for processing text.
            contain_phantom_hint (bool, optional): Flag indicating whether to include phantom hints. Defaults to False.
            past_trajectory (optional): Past trajectory data. Defaults to None.
        """
        super().__init__(config, tokenizer)  # ⭐ Initialize the base class with the provided config and tokenizer
        self.contain_phantom_hint = contain_phantom_hint
        self.past_trajectory = past_trajectory
        self.helper_llm_handle = None

    def save_init_input(self, init_input_arr:list, add_nothink: bool=False):
        if self.contain_phantom_hint:
            ...

        return super().save_init_input(init_input_arr, add_nothink)

