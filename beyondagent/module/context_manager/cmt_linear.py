import copy
import uuid
import json
import re
import torch
from typing import List, Union
from beyondagent.schema.trajectory import Sample, Reward
from beyondagent.schema.trajectory import Sample, Trajectory
from beyondagent.utils.compute_madness import repetition_penalty_reward_scalar
from beyondagent.module.context_manager.cmt_base import ExtendedMessage, ContextManagerBase
from beyondagent.module.context_manager.cmt_base import find_sublist_indices, replace_token_ids
from best_logger import register_logger, print_listofdict, print_dict, print_nested, NestedJsonItem, SeqItem




class Linear_CMT(Trajectory, ContextManagerBase):
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
        self.config = config
        self.tokenizer = tokenizer
        self.full_context: List[ExtendedMessage] = []
        self.current_context_status = ""
        max_response_length = self.config.actor_rollout_ref.rollout.response_length
        max_model_len: int = self.config.actor_rollout_ref.rollout.max_model_len
        self.max_seq_length: int = max_model_len - max_response_length
        self.max_env_output_length: int = self.config.actor_rollout_ref.rollout.max_env_len
        self.blackout_token_combo = tokenizer.encode("<|im_start|>assistant\n")
        self.generated_token_cnt = 0

        self.terminal_rewards_dict = {}
        self.discarded = False
        self.is_terminated = False
        self.reward: Union[Reward, None] = None
        self.context_time_cost = 0
        self.tag: str = ""
        self.task_id: str = ""
        self.task_train_exp_mode: str = ""
        self.current_batch_success_rate:float = -1.0
        self.llm_output_mistakes = {}
        self.experiences = []

        # log_prob_max_token_len_per_gpu: int = self.config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu
        # ref_log_prob_max_token_len_per_gpu: int = self.config.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu
        # actor_ppo_max_token_len_per_gpu: int = self.config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu
        # critic_ppo_max_token_len_per_gpu: int = self.config.critic.ppo_max_token_len_per_gpu
        # assert log_prob_max_token_len_per_gpu >= max_model_len
        # assert critic_ppo_max_token_len_per_gpu >= max_model_len
        # assert actor_ppo_max_token_len_per_gpu >= max_model_len
        # assert ref_log_prob_max_token_len_per_gpu >= max_model_len
        assert self.config.data.max_prompt_length + self.config.data.max_response_length <= max_model_len


    def prepare_previous_context(self, mod='future'):
        """
        Prepare the input context for future LLM call.

        Returns:
            list: Array of message dictionaries containing role and content_for_future,
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


    def check_context_token_num_safe(self, messages: List[dict]):
        def get_seq_length(messages):
            prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return len(self.tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"][0])
        messages = self.prepare_previous_context(mod="raw")
        return get_seq_length(messages) < self.max_seq_length   # self.config.env_engine.max_seq_length = 20480


    def get_inc(self, text_frag_from, text_frag_to):
        """
        Get the incremental token array from text_frag_from to text_frag_to.
        """
        tokenizer_output = self.tokenizer(text_frag_from, return_tensors="pt", padding=False)
        tokenizer_input_ids = tokenizer_output["input_ids"][0].tolist()
        token_ids_acc = tokenizer_input_ids

        tokenizer_output = self.tokenizer(text_frag_to, return_tensors="pt", padding=False)
        input_ids = tokenizer_output["input_ids"][0].tolist()
        input_id_increment = input_ids[len(token_ids_acc):]  # get the new tokens added in this step
        overlap_length = 0
        for i in range(len(token_ids_acc)):
            if i < len(token_ids_acc) and input_ids[i] == token_ids_acc[i]: overlap_length += 1
            else: break
        msg = f"previous token length: {len(token_ids_acc)}, overlap token length: {(overlap_length)}, increment token length: {len(input_id_increment)}"
        # print(msg)
        return input_id_increment, msg

    def remove_last_context(self):
        if len(self.full_context) > 0:
            if self.full_context[-1].author != "llm":
                self.full_context.pop(-1)

    def remove_last_non_llm_msg(self, ext_msg_list:List[ExtendedMessage]):
        if len(ext_msg_list) > 0:
            if ext_msg_list[-1].author != "llm":
                ext_msg_list.pop(-1)
        return ext_msg_list



    @property
    def steps(self):
        return self.prepare_previous_context(mod='future')

    def json(self):
        return json.dumps(self.prepare_previous_context(mod='future'), ensure_ascii=False, indent=2)

    def prepare_next_llm_context(self):
        return self.prepare_previous_context(mod='future')


    def save_init_input(self, init_input_arr:list, add_nothink: bool=False):
        """
        Save and process the initial input messages to the context.

        Args:
            init_input_arr (list): Array of initial input messages to be processed
                                  Each message should be a dict with 'role' and 'content'

        Note:
            - Initializes the context with the provided messages
            - Computes token arrays for each message
            - Validates that the context is empty before saving
        """
        # save basic
        assert len(self.full_context) == 0, "full_context should be empty when saving init input"
        for index, llm_msg in enumerate(init_input_arr):
            if (index == len(init_input_arr) - 1) and add_nothink:
                llm_msg['content'] += "\n/no_think"
            ext_msg = ExtendedMessage(
                author="initialization",
                role=llm_msg['role'],
                content=llm_msg['content'],
                token_generator="manual",
                tokenizer=self.tokenizer,
            )
            self.full_context += [ext_msg]

        # compute token array for each message
        token_ids_acc = []
        for llm_msg, ext_msg, index in zip(init_input_arr, self.full_context, range(len(init_input_arr))):
            text_with_chat_template = self.tokenizer.apply_chat_template(init_input_arr[:(index+1)], tokenize=False)
            tokenizer_output = self.tokenizer(text_with_chat_template, return_tensors="pt", padding=False)
            input_ids = tokenizer_output["input_ids"][0].tolist()
            # attention_mask = outputs["attention_mask"][0].tolist()
            input_id_increment = input_ids[len(token_ids_acc):]  # get the new tokens added in this step
            overlap_length = 0
            for i in range(len(token_ids_acc)):
                if (i < len(token_ids_acc)) and (input_ids[i] == token_ids_acc[i]): overlap_length += 1
                else: break
            ext_msg._info = f"previous token length: {len(token_ids_acc)}, overlap token length: {(overlap_length)}, increment token length: {len(input_id_increment)}"
            ext_msg.token_arr = input_id_increment
            token_ids_acc += input_ids
        return

    def influence_extra_reward(self, llm_output):
        this_msg_repetition_penalty_reward = repetition_penalty_reward_scalar(completion=llm_output['content'])
        if this_msg_repetition_penalty_reward != 0:
            print_dict({
                "reason": "repetition_penalty_reward",
                "content": llm_output['content'],
                "score": this_msg_repetition_penalty_reward,
            })
        if 'repetition_penalty_reward' not in self.llm_output_mistakes:
            self.llm_output_mistakes['repetition_penalty_reward'] = 0
        self.llm_output_mistakes['repetition_penalty_reward'] = min(this_msg_repetition_penalty_reward, self.llm_output_mistakes['repetition_penalty_reward'])

    def save_llm_output(self, llm_output, input_msg_ref, auto_register_full_context=True):
        """
        Save the output from the LLM to the full context.

        Args:
            llm_output (dict): The output from the LLM containing 'role', 'content', and 'tokens'
            input_msg_ref: Reference to the input messages for token increment calculation
            out_of_full_context: Register in full_context or not

        Note:
            - Processes the LLM output and adds it to the conversation history
            - Handles token processing and generation prompt management
            - Ensures proper tokenization and context maintenance
        """
        # save basic
        assert isinstance(llm_output, dict)
        token_generator = "manual" if 'tokens' in llm_output else "auto"
        ext_msg = ExtendedMessage(
            author="llm",
            role=llm_output['role'],
            content=llm_output['content'],
            token_generator=token_generator,
            tokenizer=self.tokenizer,
        )
        if auto_register_full_context:
            self.full_context += [ext_msg]

        # check mistakes
        if auto_register_full_context:
            self.influence_extra_reward(llm_output)

        # generate token
        def get_token_inc_from_vllm_response(input_msg_ref) -> List[int]:
            generation_prompt_token, msg = self.get_inc(
                self.tokenizer.apply_chat_template(input_msg_ref, tokenize=False, add_generation_prompt=False),
                self.tokenizer.apply_chat_template(input_msg_ref, tokenize=False, add_generation_prompt=True),
            )
            # completion_token_arr will contain generation_prompt header
            completion_token_arr, msg2 = self.get_inc(
                # ... <|im_end|>
                self.tokenizer.apply_chat_template(input_msg_ref, tokenize=False),
                # ... <|im_end|><|im_start|>...<|im_end|>
                self.tokenizer.apply_chat_template(input_msg_ref + [ {"role": llm_output['role'],  "content": llm_output['content']} ], tokenize=False),
            )
            vllm_output_raw_token = [t.token_id for t in llm_output['tokens']]
            self.generated_token_cnt += len(vllm_output_raw_token)
            final_token_arr = replace_token_ids(place_holder=completion_token_arr, replace_with=vllm_output_raw_token, begin=generation_prompt_token, end=[self.tokenizer.eos_token_id])
            return final_token_arr

        if token_generator == "manual":
            token_arr_method2 = get_token_inc_from_vllm_response(input_msg_ref)
            ext_msg.token_arr = token_arr_method2
        return ext_msg


    def save_llm_output_do_not_register_full_context(self, llm_output, input_msg_ref):
        return Linear_CMT.save_llm_output(self, llm_output, input_msg_ref, auto_register_full_context=False)


    def save_env_output(self, env_output:dict, input_msg_ref:List[dict]=None, add_nothink=False):
        """
        Save and process environment output to the context.

        Args:
            env_output (dict): Environment output containing 'content'
            input_msg_ref (List[dict], optional): Reference messages for token calculation

        Note:
            - Clips environment output if it exceeds max_env_output_length
            - Processes the output as a user message in the conversation
            - Computes and stores token arrays for the environment response
        """
        assert isinstance(env_output, dict)
        if ('content' not in env_output) and ('error' in env_output):
            env_output['content'] = f"[Error from environment: {env_output['error']}]"
        elif ('content' not in env_output) or (not env_output['content']):
            env_output['content'] = '[No content provided by the environment]'
        if add_nothink:
            env_output['content'] += " /no_think"
        ext_msg = ExtendedMessage(
            author="env",
            role="user",
            content=env_output['content'],
            clip=True,
            clip_token_limit=self.max_env_output_length,
            token_generator="auto",
            tokenizer=self.tokenizer,
        )
        self.full_context += [ext_msg]
        return

    def to_role_content(self, ext_msg_array: List[ExtendedMessage]) -> List[dict]:
        return [{"role": ext_msg.role, "content": ext_msg.content_for_future} for ext_msg in ext_msg_array]

    def prepare_world_interaction(self) -> str:
        """
        Process the latest model content before environment interaction.

        Returns:
            str: Processed content, with code extracted from markdown code blocks if present
                 or the raw content if no code blocks are found

        Note:
            - Extracts Python code from markdown code blocks (```python```)
            - Returns the raw content if no valid code blocks are found
        """
        latest_content = self.full_context[-1].content
        return latest_content

    def filter_context_via_author(self, author: str) -> List[ExtendedMessage]:
        return copy.deepcopy([ c for c in self.full_context if c.author == author ])

    def filter_context_via_authors(self, authors: str) -> List[ExtendedMessage]:
        return copy.deepcopy([ c for c in self.full_context if c.author in authors ])

    def group_tokenize(self):
        # assert self.latest_llm_interaction_socket is None, "unprocessed message buffer! forget to call `save_llm_output` after `prepare_next_llm_context`?"
        sample_arr = []
        ext_steps=self.full_context
        cmt_tokenized = self.tokenize_steps(ext_steps=ext_steps)
        sample = Sample(
            data_id=self.data_id,
            rollout_id=self.rollout_id,
            task_id=self.task_id,
            minor_index_id=0,
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
        sample_arr += [sample]
        return sample_arr


    def group_render_token_log(self):
        ext_steps=self.full_context
        cmt_tokenized = self.tokenize_steps(ext_steps=ext_steps)
        text_arr = [self.tokenizer.decode(t) for t in cmt_tokenized["input_ids"]]
        input_id_arr = [str(t) for t in cmt_tokenized["input_ids"]]
        loss_mask_color_arr = ["#09ABCF" if mask==1 else "#D98510" for mask in cmt_tokenized["loss_mask"]]
        return {
            "text_arr": text_arr,
            "input_id_arr": input_id_arr,
            "loss_mask_color_arr": loss_mask_color_arr,
        }


    def generate_log(self, task_id):
        nested_items_print_buffer = {}
        ext_steps=self.full_context
        cmt_tokenized = self.tokenize_steps(ext_steps=ext_steps)
        text_arr = [self.tokenizer.decode(t) for t in cmt_tokenized["input_ids"]]
        input_id_arr = [str(t) for t in cmt_tokenized["input_ids"]]
        loss_mask_color_arr = ["#09ABCF" if mask==1 else "#D98510" for mask in cmt_tokenized["loss_mask"]]
        buffer = {
            "text_arr": text_arr,
            "input_id_arr": input_id_arr,
            "loss_mask_color_arr": loss_mask_color_arr,
        }
        len_prompt_ids = len(cmt_tokenized["prompt_ids"])
        len_response_ids = len(cmt_tokenized["response_ids"])
        len_input_ids = len(cmt_tokenized["input_ids"])
        reward = self.reward.outcome
        task_outcome = str(self.reward.success_rate)
        final_reward = self.reward_patch(self.reward).outcome
        selectors = [task_id, task_outcome]
        nested_items_print_buffer[f".".join(selectors)] = NestedJsonItem(
            item_id=f"item",
            outcome=task_outcome,
            len_prompt_ids=len_prompt_ids,
            len_response_ids=len_response_ids,
            len_input_ids=len_input_ids,
            reward=f"{float(reward):.3f}",
            final_reward=final_reward,
            content=SeqItem(
                text = buffer['text_arr'],  # 文本
                title = buffer['text_arr'], # 鼠标悬浮文本
                count = buffer['input_id_arr'], # 高亮文本
                color = buffer['loss_mask_color_arr']   # 颜色
            )
        )
        print_nested(nested_items_print_buffer,
            main_content="This is the main content of the nested JSON",
            header=f"Training task {task_id} (Final Reward {final_reward})",
            mod="rollout",
            narrow=False,
            attach="Copy Sample Message"
        )
        print_listofdict(
            self.steps,
            header=f"Training task {task_id} (Final Reward {final_reward})",
            mod="conversation",
            narrow=False,
        )

    def reward_patch(self, reward):
        _reward = copy.deepcopy(reward)
        # if self.compute_madness() < 0: _reward.outcome = -1.0
        return _reward


    def compute_madness(self) -> float:
        """
        根据样本输出的乱码占比（异常中文日文俄文，special token，重复字符），评估模型的疯狂程度。
        注意：我们假定模型仅输出ascii字符。
        """
        threshold = -0.01
        for k, v in self.llm_output_mistakes.items():
            if v < threshold: return -1.0
        return 0.0


    def tokenize_steps(self, ext_steps: List[ExtendedMessage], debug=False) -> dict:
        """
        Create an Experience object from the current conversation context.

        Returns:
            Experience: An object containing processed conversation data for model training

        Note:
            - Removes the last user message as it's not required in casual model training
            - Processes input IDs, attention masks, and loss masks
            - Separates prompt and response components
            - Handles position IDs and reward scores
            - Truncates output IDs as needed
        """
        from verl.utils.model import compute_position_id_with_mask
        ext_steps = self.remove_last_non_llm_msg(copy.deepcopy(ext_steps))

        # ANNI experience extraction and discard
        def extract_and_discard_experience(input_string, experience_template):  # <EXP>{}</EXP>
            pattern = re.escape(experience_template).replace(r'\{\}', '(.*?)')
            match = re.search(pattern, input_string)
            if match:
                experience = match.group(1)
                prompt = re.sub(pattern, '', input_string)
                return experience, prompt
            else:
                return "", input_string

        # ANNI experience extraction and discard
        # from vsdb import bp
        # bp('X1')
        if self.task_train_exp_mode == "discard":
            # "\n\nSome Related Experience to help you to complete the task:<EXP>{}</EXP>"
            self.experience_template = self.config.hybrid_experience_training.experience_template
            for i, ext_msg in enumerate(ext_steps):
                experience, new_content = extract_and_discard_experience(ext_msg.content_for_future, self.experience_template)
                self.experiences += [experience]
                if experience:
                    ext_steps[i] = ExtendedMessage(
                        author=ext_msg.author,
                        role=ext_msg.role,
                        content=new_content,
                        token_generator='auto',
                        tokenizer=self.tokenizer,
                        uuid=ext_msg.uuid,
                    )

        # mapping
        input_ids = []
        attention_mask = []
        loss_mask = []
        split_prompt_reponse_index = -1
        for ext_msg in ext_steps:
            # find split index, this have to be done before input_ids += ext_msg.token_arr
            if (split_prompt_reponse_index == -1) and (ext_msg.need_training):
                split_prompt_reponse_index = len(input_ids)
                assert ext_msg.author == 'llm', "The first message after initialization should be from LLM, not from env or user"
            input_ids += ext_msg.token_arr
            attention_mask += [1] * len(ext_msg.token_arr)
            loss_mask += ext_msg.get_loss_mask(blackout_token_combo=self.blackout_token_combo)

        assert split_prompt_reponse_index != -1, "split_prompt_reponse_index should not be -1, at least one message should be in the context"
        position_ids = compute_position_id_with_mask(torch.tensor(attention_mask)).tolist()

        # sperate prompt and response
        prompt_ids =            input_ids[:split_prompt_reponse_index]
        prompt_attention_mask = attention_mask[:split_prompt_reponse_index]
        prompt_position_ids =   position_ids[:split_prompt_reponse_index]
        prompt_loss_mask =      loss_mask[:split_prompt_reponse_index]

        response_ids =              input_ids[split_prompt_reponse_index:]
        response_attention_mask =   attention_mask[split_prompt_reponse_index:]
        response_position_ids =     position_ids[split_prompt_reponse_index:]
        response_loss_mask =        loss_mask[split_prompt_reponse_index:]

        cmt_tokenized = {}
        cmt_tokenized["input_ids"] = input_ids
        cmt_tokenized["prompt_ids"] = prompt_ids
        cmt_tokenized["response_ids"] = response_ids
        cmt_tokenized["attention_mask"] = attention_mask
        cmt_tokenized["prompt_attention_mask"] = prompt_attention_mask
        cmt_tokenized["response_attention_mask"] = response_attention_mask
        cmt_tokenized["loss_mask"] = loss_mask
        cmt_tokenized["prompt_loss_mask"] = prompt_loss_mask
        cmt_tokenized["response_loss_mask"] = response_loss_mask
        cmt_tokenized["position_ids"] = position_ids
        cmt_tokenized["prompt_position_ids"] = prompt_position_ids
        cmt_tokenized["response_position_ids"] = response_position_ids

        return cmt_tokenized
