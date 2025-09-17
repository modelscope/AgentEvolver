import torch
import copy
from typing import List
from beyondagent.schema.trajectory import Sample
from best_logger import print_dict, print_listofdict
from beyondagent.module.context_manager.cmt_linear import ExtendedMessage, Linear_CMT
from beyondagent.module.context_manager.cmt_linear import find_sublist_indices, replace_token_ids
from best_logger import register_logger, print_dict, print_nested, NestedJsonItem, SeqItem
from textwrap import dedent

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

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.config = config
        self.tokenizer = tokenizer
        self.full_context: List[ExtendedMessage] = []
        self.current_context_status = ""
        max_response_length = self.config.actor_rollout_ref.rollout.response_length
        max_model_len: int = self.config.actor_rollout_ref.rollout.max_model_len

        assert self.config.data.max_response_length < self.config.data.max_prompt_length, "think linear template requires a big max_prompt_length"

        self.max_seq_length: int = max_model_len - max_response_length
        assert self.max_seq_length <= self.config.data.max_prompt_length, "max_seq_length should be less than or equal to max_prompt_length"


        self.max_env_output_length: int = self.config.actor_rollout_ref.rollout.max_env_len
        self.blackout_token_combo = tokenizer.encode("<|im_start|>assistant\n")

        self.terminal_rewards_dict = {}
        self.latest_llm_interaction_socket: List[ExtendedMessage] = None
        self.grouped_steps: List[List[ExtendedMessage]] = []

        self.discarded = False
        self.is_terminated = False
        self.reward = None
        self.context_time_cost = 0
        self.force_think = config.actor_rollout_ref.rollout.force_think
        self.env_feedin_preference = config.env_service.env_feedin_preference
        if not self.force_think:
            # think_hint_for_qwen3 =
            self.think_hint: str = "\n\nThink about the next step before answering. Your thought (<think>...</think>) should be as short and concise as possible."
        else:
            if self.env_feedin_preference == "box":
                force_think_prompt = dedent("""
                    Additional requirements: Think before action! You must think step by step before your next action, and you must use <think>...</think> to wrap your thinking process before finally produce your answer with \\box{}.
                    For example:
                    <think>...your thinking process...</think>
                    \\box{...your final answer...}
                """)
            elif self.env_feedin_preference == "code":
                force_think_prompt = dedent("""
                    Additional requirements: Think before action! You must think step by step before your next action, and you must use <think>...</think> to wrap your thinking process before finally produce the next-step action.
                    For example:
                    <think>...your thinking process...</think>
                    ```python
                    # your action here
                    ```
                """)
            else:
                raise ValueError(f"Unsupported env_feedin_preference: {self.env_feedin_preference}")
            # think_hint_for_qwen2 =
            self.think_hint: str = force_think_prompt

    def _get_seq_length(self, messages: List[dict]) -> int:
        """
        Calculates the sequence length of the provided messages after applying the chat template and tokenizing them.

        Args:
            messages (List[dict]): A list of message dictionaries to be processed.

        Returns:
            int: The length of the tokenized sequence.
        """
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # ⭐ Apply the chat template to the messages
        return len(self.tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"][0])  # ⭐ Tokenize the prompt text and return the length of the tokenized sequence


    def check_context_token_num_safe(self, messages: List[dict]) -> bool:
        """
        Checks if the total number of tokens in the provided messages is within the allowed limit.

        Args:
            messages (List[dict]): A list of message dictionaries.

        Returns:
            bool: True if the total number of tokens is within the allowed limit, False otherwise.
        """
        return self._get_seq_length(messages) < self.max_seq_length  # ⭐ Compare the total token count with the max allowed length


    @property
    def steps(self):
        """
        Property that returns the result of the prepare_previous_context method with the mode set to 'future'.

        Returns:
            Any: The result of the prepare_previous_context method.
        """
        return self.prepare_previous_context(mod='future')  # ⭐ Call the prepare_previous_context method with 'future' mode


    def prepare_next_llm_context(self):
        """
        Prepares the next context for the LLM by filtering and modifying the latest interaction messages.
        It removes the `think` tags from previous LLM messages, adds `/no_think` tags to non-final messages,
        and updates the context accordingly.

        Returns:
            dict: The updated context in a dictionary format.
        """
        self.latest_llm_interaction_socket = []
        # 筛选出 `初始message-user-llm-user-llm`` 或者 `初始message-llm-user-llm-user``
        self.latest_llm_interaction_socket = self.filter_context_via_authors(["initialization", "llm", "env"])  # ⭐ Filter the context based on authors

        for index, ext_msg in enumerate(list(self.latest_llm_interaction_socket)):
            # is_last 是最后一条信息
            # remove history llm author's think (and add /no_think tag to every but last message)
            is_last = (index == len(self.latest_llm_interaction_socket) - 1)
            # 根据消息类型进行处理
            if ext_msg.author == "llm":
                # 如果是以往的llm消息，去掉think标签
                import re
                new_ext_msg_content = re.sub(r'<think>.*?</think>', '', ext_msg.content, flags=re.DOTALL).strip()  # ⭐ Remove <think> tags from the content
                new_ext_msg_content = new_ext_msg_content.replace("<think>", "")
                new_ext_msg_content = new_ext_msg_content.replace("</think>", "")
                # new_ext_msg_content = re.sub(r'<think>.*?</think>', '<think>\n\n</think>', ext_msg.content, flags=re.DOTALL)
                if self.config.actor_rollout_ref.rollout.train_history_infer_token:
                    assert ext_msg.author == "llm"
                    self.latest_llm_interaction_socket[index] = ExtendedMessage(
                        author=ext_msg.author,
                        role=ext_msg.role,
                        content=new_ext_msg_content,
                        token_generator='auto',
                        tokenizer=self.tokenizer,
                    )
                else:
                    assert ext_msg.author == "llm"
                    author_override = "llm(do_not_train)"
                    self.latest_llm_interaction_socket[index] = ExtendedMessage(
                        author=author_override,
                        role=ext_msg.role,
                        content=new_ext_msg_content,
                        token_generator='auto',
                        tokenizer=self.tokenizer,
                    )
            elif ext_msg.author in ["env", "initialization"]:
                if self.config.actor_rollout_ref.rollout.train_history_infer_token:
                    # 如果是初始化或者环境反馈，都加上 /no_think 标签
                    if not is_last:
                        self.latest_llm_interaction_socket[index] = ExtendedMessage(
                            author=ext_msg.author,
                            role=ext_msg.role,
                            content=ext_msg.content_for_future + "\n/no_think",
                            token_generator='auto',
                            tokenizer=self.tokenizer,
                        )
                    else:
                        self.latest_llm_interaction_socket[index] = ExtendedMessage(
                            author=ext_msg.author,
                            role=ext_msg.role,
                            content=ext_msg.content_for_future + self.think_hint,
                            token_generator='auto',
                            tokenizer=self.tokenizer,
                        )
                else:
                    # 如果是初始化或者环境反馈
                    if not is_last:
                        self.latest_llm_interaction_socket[index] = ExtendedMessage(
                            author=ext_msg.author,
                            role=ext_msg.role,
                            content=ext_msg.content_for_future,
                            token_generator='auto',
                            tokenizer=self.tokenizer,
                        )
                    else:
                        self.latest_llm_interaction_socket[index] = ExtendedMessage(
                            author=ext_msg.author,
                            role=ext_msg.role,
                            content=ext_msg.content_for_future + self.think_hint,
                            token_generator='auto',
                            tokenizer=self.tokenizer,
                        )
            else:
                raise RuntimeError(f"Unknown author {ext_msg.author} in latest_llm_interaction_socket")

        dict_context = self.to_role_content(self.latest_llm_interaction_socket)  # ⭐ Convert the filtered and modified messages to a dictionary context
        return dict_context



    def generate_log(self, task_id):
        """
        Generates a log for the given task ID, which includes tokenized steps, decoded text, and other relevant details.

        Args:
            task_id (str): The ID of the task for which the log is being generated.

        Returns:
            None
        """
        nested_items_print_buffer = {}
        for index, ext_steps in enumerate(self.grouped_steps):
            cmt_tokenized = self.tokenize_steps(ext_steps=ext_steps, debug=True)  # ⭐ Tokenize the extended steps
            text_arr = [self.tokenizer.decode(t) for t in cmt_tokenized["input_ids"]]
            input_id_arr = [str(t) for t in cmt_tokenized["input_ids"]]
            loss_mask_color_arr = ["#09ABCF" if mask==1 else "#D98510" for mask in cmt_tokenized["loss_mask"]]
            buffer = {
                "text_arr": text_arr,
                "input_id_arr": input_id_arr,
                "loss_mask_color_arr": loss_mask_color_arr,
            }
            reward = self.reward.outcome
            task_outcome = str(self.reward.success_rate)
            selectors = [task_id, task_outcome, str(index)]
            len_prompt_ids = len(cmt_tokenized["prompt_ids"])
            len_response_ids = len(cmt_tokenized["response_ids"])
            len_input_ids = len(cmt_tokenized["input_ids"])
            assert len_prompt_ids + len_response_ids == len_input_ids, "len_prompt_ids + len_response_ids should equal to len_input_ids"
            nested_items_print_buffer[f".".join(selectors)] = NestedJsonItem(
                item_id=f"item",
                outcome=task_outcome,
                len_prompt_ids=len_prompt_ids,
                len_response_ids=len_response_ids,
                len_input_ids=len_input_ids,
                reward=f"{float(reward):.3f}",
                content=SeqItem(
                    text=buffer['text_arr'],  # 文本
                    title=buffer['text_arr'],  # 鼠标悬浮文本
                    count=buffer['input_id_arr'],  # 高亮文本
                    color=buffer['loss_mask_color_arr']  # 颜色
                )
            )
        print_nested(nested_items_print_buffer,
            main_content="This is the main content of the nested JSON",
            header=f"Training, task {task_id}",
            mod="rollout",
            narrow=False,
            attach="copy this"
        )


    def save_init_input(self, init_input_arr:list, add_nothink):
        super().save_init_input(init_input_arr, add_nothink)
        return


    def save_llm_output(self, llm_output, input_msg_ref):
        """
        Saves the LLM output and updates the internal state by appending the latest interaction to the grouped steps and resetting the latest LLM interaction socket.

        Args:
            llm_output: The output generated by the LLM.
            input_msg_ref: A reference to the input message that triggered the LLM output.

        Returns:
            None
        """
        ext_msg = super().save_llm_output(llm_output, input_msg_ref)  # ⭐ Save the LLM output and get the extended message
        this_interaction = copy.deepcopy(self.latest_llm_interaction_socket + [ext_msg])  # ⭐ Create a deep copy of the latest interaction with the new message
        self.grouped_steps += [this_interaction]  # ⭐ Append the latest interaction to the grouped steps
        self.latest_llm_interaction_socket = []  # ⭐ Reset the latest LLM interaction socket for the next interaction
        return


    def save_env_output(self, env_output:dict, input_msg_ref:List[dict]=None, add_nothink=False):
        super().save_env_output(env_output, input_msg_ref, add_nothink)
        return


    def prepare_world_interaction(self) -> str:
        """
        Retrieves the latest content from the full context of the conversation.

        Returns:
            str: The latest content from the full context.
        """
        latest_content = self.full_context[-1].content  # ⭐ Retrieve the latest content from the full context
        return latest_content


    def group_tokenize(self):
        """
        Tokenizes grouped steps of a conversation and creates Sample objects for each group.

        Args:
            None

        Returns:
            list: A list of Sample objects, each representing a tokenized group of steps.
        """
        sample_arr = []
        max_num_group = self.config.actor_rollout_ref.rollout.multi_turn.max_sample_per_task
        for index, ext_steps in enumerate(self.grouped_steps):
            if index >= max_num_group:
                print(f"Warning: group_tokenize only process first {max_num_group} groups, but got {len(self.grouped_steps)} groups")
                break
            cmt_tokenized = self.tokenize_steps(ext_steps=ext_steps, debug=True)  # ⭐ Tokenize the extended steps
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
            sample.truncate_output_ids()
            assert len(sample.response_ids) != 0, "response_ids should not be empty"
            sample_arr += [sample]
        return sample_arr


