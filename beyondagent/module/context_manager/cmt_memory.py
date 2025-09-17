# https://arxiv.org/pdf/2505.10978

import torch
import copy
from typing import List
from best_logger import print_dict, print_listofdict
from beyondagent.module.context_manager.cmt_linear import ExtendedMessage, Linear_CMT
from beyondagent.module.context_manager.cmt_linear import find_sublist_indices, replace_token_ids
from beyondagent.schema.trajectory import Sample
from beyondagent.utils.markdown_parser import read_markdown_and_extract_sections
# next_step_prompt_init = """
# Note that you
# """
first_instruction = """Use strict markdown format
# next-step instruction code
CODE_PLACEHOLDER (use ``` ... ``` to wrap the code)"""

env_extra_instruction = """
---
1. Extract all useful information from the environment feedback above.
2. Generate code as next-step instruction.
3. Any information that is not extracted will be lost, so please extract all useful information relevant to current task such as accounts, keys, access tokens, etc.
4. Use strict markdown format, and strictly answer with the following 4 sections:
    - `# current step`
    - `# previous instruction code`
    - `# relevant environment feedback`
    - `# next-step instruction code`

---

# current step
STEP_PLACEHOLDER (step index and step summary)

# previous instruction code
PREVIOUS_CODE_PLACEHOLDER (use ``` ... ``` to wrap the code)

# relevant environment feedback
FEEDBACK_PLACEHOLDER (list of information, or traceback)

# next-step instruction code
CODE_PLACEHOLDER (use ``` ... ``` to wrap the code)
"""

from pydantic import BaseModel, Field


class GroupedSteps(BaseModel):
    num_groups: int = Field(default=0, description="Number of groups in the grouped steps")
    grouped_step_list: List[List[dict]] = Field(default_factory=list, description="List of grouped steps, each containing a list of ExtendedMessage objects")


class MemoryCoreCMT(Linear_CMT):

    def prepare_next_llm_context(self) -> List[dict]:
        """
        [instruction] -> ( [history] -> [previous instruction] -> [env feedback + env_extra_instruction] )
        """
        assert self.latest_llm_interaction_socket is None, "`prepare_next_llm_context` must be called at proper time!"
        self.current_step += 1
        self.latest_llm_interaction_socket = []
        is_first_interaction = (len(self.filter_context_via_author("llm")) == 0)


        if is_first_interaction:
            # 先解决第一次交互的特殊情况
            part_1_instruction_array = self.filter_context_via_author("initialization")
            part_1_instruction_array += [
                ExtendedMessage(
                    author="initialization",
                    role="user",
                    content=first_instruction,
                    token_generator="auto",
                    tokenizer=self.tokenizer,
                )
            ]
            self.latest_llm_interaction_socket += part_1_instruction_array
            dict_context = self.to_role_content(self.latest_llm_interaction_socket)
            return dict_context

        # ---------- part 1 ----------
        part_1_instruction_array = self.filter_context_via_author("initialization")

        # ---------- part 2 ----------
        memory_history_ext_msg = self.filter_context_via_author("memory")
        if len(memory_history_ext_msg) != 0:
            str_concat_buffer = "Previous steps:\n---\n"
            for ext_msg in memory_history_ext_msg:
                str_concat_buffer += ext_msg.content_for_future + "\n"
            part_2_memory_history_array = [ExtendedMessage(
                author="memory",
                role='assistant',
                content=str_concat_buffer,
                token_generator="auto",
                tokenizer=self.tokenizer,
            )]
        else:
            part_2_memory_history_array = []
        # ---------- part 3 ----------
        last_llm_result = self.filter_context_via_author("llm")[-1]
        last_llm_result_decompose, _, find_nothing = read_markdown_and_extract_sections(
            markdown_text=last_llm_result.content_for_future,
            expected_sections=["current step", "previous instruction code", "relevant environment feedback", "next-step instruction code"],
            default_placeholder="❌ not available."
        )
        if not find_nothing:
            last_llm_result_next_step_code = last_llm_result_decompose['next-step instruction code']
            part_3_previous_instruction = ExtendedMessage(
                author="llm",
                role='assistant',
                content=last_llm_result_next_step_code,
                token_generator="auto",
                tokenizer=self.tokenizer,
            )
        else:
            part_3_previous_instruction = ExtendedMessage(
                author="llm",
                role='assistant',
                content=last_llm_result.content_for_future,
                token_generator="auto",
                tokenizer=self.tokenizer,
            )
        # ---------- part 4 ----------
        message_arr_env_last = self.filter_context_via_author("env")[-1]
        part_4_env_and_env_extra = ExtendedMessage(
            author="env",
            role='user',
            content=message_arr_env_last.content_for_future + env_extra_instruction,
            token_generator="auto",
            tokenizer=self.tokenizer,
        )
        self.latest_llm_interaction_socket = part_1_instruction_array + \
                                             part_2_memory_history_array + \
                                             [part_3_previous_instruction] + \
                                             [part_4_env_and_env_extra]

        dict_context = self.to_role_content(self.latest_llm_interaction_socket)
        return dict_context


    def save_llm_output(self, llm_output, input_msg_ref):
        """
        Save the output from the LLM to the full context.

        Args:
            llm_output (dict): The output from the LLM containing 'role', 'content', and 'tokens'
            input_msg_ref: Reference to the input messages for token increment calculation

        Note:
            - Processes the LLM output and adds it to the conversation history
            - Handles token processing and generation prompt management
            - Ensures proper tokenization and context maintenance
        """
        def get_token_inc_from_vllm_response(input_msg_ref) -> List[int]:
            """
            Calculate the token increments from the VLLM response.

            Args:
                input_msg_ref: Reference to the input messages for token increment calculation

            Returns:
                List[int]: The final token array after processing.
            """
            generation_prompt_token, msg = self.get_inc(
                self.tokenizer.apply_chat_template(input_msg_ref, tokenize=False, add_generation_prompt=False),
                self.tokenizer.apply_chat_template(input_msg_ref, tokenize=False, add_generation_prompt=True),
            )  # ⭐ Calculate the token increment for the generation prompt
            completion_token_arr, msg2 = self.get_inc(
                self.tokenizer.apply_chat_template(input_msg_ref, tokenize=False),
                self.tokenizer.apply_chat_template(input_msg_ref + [ {"role": llm_output['role'],  "content": llm_output['content']} ], tokenize=False),
            )  # ⭐ Calculate the token increment for the completion
            vllm_output_raw_token = [t.token_id for t in llm_output['tokens']]
            final_token_arr = replace_token_ids(place_holder=completion_token_arr, replace_with=vllm_output_raw_token, begin=generation_prompt_token, end=[self.tokenizer.eos_token_id])
            return final_token_arr

        # save basic
        assert isinstance(llm_output, dict)
        assert self.latest_llm_interaction_socket is not None, "`save_llm_output` must be called at proper time!"
        ext_msg = ExtendedMessage(
            author="llm",
            role=llm_output['role'],
            content=llm_output['content'],
            token_generator="manual",
            tokenizer=self.tokenizer,
        )  # ⭐ Create an ExtendedMessage object for the LLM output
        self.full_context += [ext_msg]
        ext_msg.token_arr = get_token_inc_from_vllm_response(input_msg_ref)
        this_interaction = copy.deepcopy(self.latest_llm_interaction_socket + [ext_msg])
        self.grouped_steps += [this_interaction]
        self.latest_llm_interaction_socket = None

        # extract memory
        lm_result_decompose, find_everything, find_nothing = read_markdown_and_extract_sections(
            markdown_text=llm_output['content'],
            expected_sections=["current step", "previous instruction code", "relevant environment feedback", "next-step instruction code"],
            default_placeholder="❌ not available."
        )  # ⭐ Extract and decompose the LLM output into sections
        from textwrap import dedent
        memory_construct = dedent("""
        >> step: {current_step}
        >> instruction:
        {next_step_instruction_code}
        >> feedback from environment:
        {relevant_environment_feedback}
        ---
        """).format(
            current_step=lm_result_decompose['current step'],
            next_step_instruction_code=lm_result_decompose['next-step instruction code'],
            relevant_environment_feedback=lm_result_decompose['relevant environment feedback']
        )  # ⭐ Format the extracted sections into a memory construct
        ext_msg_memory = ExtendedMessage(
            author="memory",
            role="assistant",
            content=memory_construct,
            token_generator="auto",
            tokenizer=self.tokenizer,
        )
        if find_everything:
            self.full_context += [ext_msg_memory]
        return



class MemoryCMT(MemoryCoreCMT):
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
    """

    def __init__(self, config, tokenizer):
        """
        Initializes the MemoryCMT with the given configuration and tokenizer.

        Args:
            config: Configuration object containing environment and model settings.
            tokenizer: Tokenizer instance for processing text.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.full_context: List[ExtendedMessage] = []
        self.current_context_status = ""
        max_response_length = self.config.actor_rollout_ref.rollout.response_length
        max_model_len: int = self.config.actor_rollout_ref.rollout.max_model_len
        self.max_seq_length: int = max_model_len - max_response_length  # ⭐ Calculate the maximum sequence length for the context window
        self.max_env_output_length: int = self.config.actor_rollout_ref.rollout.max_env_len
        self.blackout_token_combo = tokenizer.encode("<|im_start|>assistant\n")

        self.terminal_rewards_dict = {}
        self.reward = None
        self.latest_llm_interaction_socket: List[ExtendedMessage] = None
        self.grouped_steps: List[List[ExtendedMessage]] = []
        self.data_id = None
        self.rollout_id = None
        self.task_id = None

        self.current_step = 0

    @property
    def steps(self):
        raise NotImplementedError("MemoryCMT does not support steps.")

    def generate_log(self, task_id):
        """
        Generates a log of grouped steps from the conversation history.

        Args:
            task_id (int): The ID of the task for which the log is being generated.

        Returns:
            GroupedSteps: An object containing the grouped steps from the conversation history.
        """
        result = GroupedSteps()
        result.num_groups = len(self.grouped_steps)
        for steps in self.grouped_steps:
            result.grouped_step_list += [self.to_role_content(steps)]  # ⭐ Convert each group of steps to role-content format and add to the result
        grouped_steps: GroupedSteps = result
        # for index, steps in enumerate(grouped_steps.grouped_step_list):
        #     print_listofdict(steps, mod='appworld_io', header=f'Task-{task_id} {index}/{grouped_steps.num_groups}')
        return



    def check_context_token_num_safe(self, messages: List[dict]):
        """
        Checks if the number of tokens in the provided messages is within a safe limit by calling the parent class's implementation.

        Args:
            messages (List[dict]): A list of message dictionaries, each containing at least a 'role' and 'content' key.

        Returns:
            bool: True if the token count is safe, False otherwise.
        """
        return super().check_context_token_num_safe(messages)  # ⭐ Calls the parent class's method to check token safety


    def prepare_previous_context(self, mod='future'):
        """
        Prepare the input context for future LLM call.

        Args:
            mod (str): The mode in which to prepare the context. Can be 'future' or 'raw'.

        Returns:
            list: Array of message dictionaries containing role and content_for_future or content,
                 formatted for LLM input.
        """
        if mod=='future':
            message_arr = [
                {"role": c.role, "content": c.content_for_future}
                for c in self.full_context
            ]
            return message_arr

        elif mod=='raw':
            message_arr = [
                {"role": c.role, "content": c.content}
                for c in self.full_context
            ]
            return message_arr

        else:
            raise ValueError(f"Unknown mod {mod} in prepare_previous_context, only support 'future' and 'raw'")


    def save_init_input(self, init_input_arr:list, add_nothink):
        """
        Save and process the initial input messages to the context.

        Args:
            init_input_arr (list): Array of initial input messages to be processed.
                                   Each message should be a dict with 'role' and 'content'.
            add_nothink: Additional parameter (not used in the provided code snippet).

        Note:
            - Initializes the context with the provided messages
            - Computes token arrays for each message
            - Validates that the context is empty before saving
        """
        # save basic
        assert self.latest_llm_interaction_socket is None, "`save_init_input` must be called at proper time!"  # ⭐ Ensures the method is called at the correct time
        super().save_init_input(init_input_arr, add_nothink)




    def save_env_output(self, env_output:dict, input_msg_ref:List[dict]=None, add_nothink=False):
        """
        Save and process environment output to the context.

        Args:
            env_output (dict): Environment output containing 'content'.
            input_msg_ref (List[dict], optional): Reference messages for token calculation.
            add_nothink (bool, optional): Flag to indicate whether to add a no-think message. Defaults to False.

        Note:
            - Clips environment output if it exceeds max_env_output_length.
            - Processes the output as a user message in the conversation.
            - Computes and stores token arrays for the environment response.
        """
        assert self.latest_llm_interaction_socket is None, "`save_env_output` must be called at proper time!"  # ⭐ Ensures the function is called at the correct time
        super().save_env_output(env_output, input_msg_ref=input_msg_ref, add_nothink=add_nothink)


    def prepare_world_interaction(self):
        """
        Process the latest model content before environment interaction.

        Returns:
            str: Processed content, with code extracted from markdown code blocks if present
                 or the raw content if no code blocks are found

        Note:
            - Extracts Python code from markdown code blocks (```python```)
            - Returns the raw content if no valid code blocks are found
        """
        ext_message_arr_memory = self.filter_context_via_author("llm")

        # extract memory
        lm_result_decompose, find_everything, find_nothing = read_markdown_and_extract_sections(
            markdown_text=ext_message_arr_memory[-1].content,
            expected_sections=["current step", "previous instruction code", "relevant environment feedback", "next-step instruction code"],
            default_placeholder="❌ not available."
        )

        return lm_result_decompose['next-step instruction code']


    def group_tokenize(self):
        """
        Tokenizes the grouped steps of a conversation and prepares them into Sample objects.

        This function processes a limited number of groups, tokenizes each group, and converts it into a Sample object.
        Each Sample object contains various attributes such as input IDs, attention masks, and position IDs.
        The function also handles truncation of output IDs to fit within specified length constraints.

        Returns:
            list[Sample]: A list of Sample objects, each representing a tokenized group of steps.
        """
        # assert self.latest_llm_interaction_socket is None, "unprocessed message buffer! forget to call `save_llm_output` after `prepare_next_llm_context`?"
        sample_arr = []
        max_num_group = 30 # self.config.actor_rollout_ref.rollout.multi_turn.max_steps
        for index, ext_steps in enumerate(self.grouped_steps):
            if index >= max_num_group:
                print(f"Warning: group_tokenize only process first {max_num_group} groups, but got {len(self.grouped_steps)} groups")
                break
            cmt_tokenized = self.tokenize_steps(ext_steps=ext_steps)  # ⭐ Tokenize the current group of steps
            sample = Sample(
                data_id=self.data_id,
                rollout_id=self.rollout_id,
                task_id=self.task_id,
                minor_index_id=index,
                messages=self.to_role_content(ext_steps),
                input_ids=cmt_tokenized["input_ids"],
                prompt_ids=cmt_tokenized["prompt_ids"],
                response_ids=cmt_tokenized["response_ids"],
                attention_mask=cmt_tokenized["attention_mask"],
                prompt_attention_mask=cmt_tokenized["prompt_attention_mask"],
                response_attention_mask=cmt_tokenized["response_attention_mask"],
                loss_mask=cmt_tokenized["loss_mask"],
                prompt_loss_mask=cmt_tokenized["prompt_loss_mask"],
                response_loss_mask=cmt_tokenized["response_loss_mask"],
                position_ids=cmt_tokenized["position_ids"],
                prompt_position_ids=cmt_tokenized["prompt_position_ids"],
                response_position_ids=cmt_tokenized["response_position_ids"],
                reward_scores=self.reward.model_dump(), # reward is duplicated in each sample
                max_prompt_len=self.config.data.max_prompt_length,
                max_response_len=self.config.data.max_response_length,
                max_model_len=self.config.data.max_response_length + self.config.data.max_prompt_length,
            )
            sample.truncate_output_ids()  # ⭐ Truncate the output IDs to fit within the specified length constraints
            sample_arr += [sample]
        return sample_arr


