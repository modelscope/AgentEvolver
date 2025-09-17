import torch
import copy
import re
import json
import random
import time
from typing import List, Callable
from beyondagent.schema.trajectory import Sample
from best_logger import print_dict, print_listofdict
from beyondagent.module.context_manager.cmt_linear_think import ExtendedMessage, Linear_CMT, LinearThinkCMT
from beyondagent.module.context_manager.cmt_linear import find_sublist_indices, replace_token_ids
from best_logger import register_logger, print_dict, print_nested, NestedJsonItem, SeqItem
from textwrap import dedent
from openai import OpenAI
from loguru import logger


def construct_alien_llm_chat_fn(config, rollout_config):
    def alien_llm_chat_fn(messages, request_id=""):
        max_try = 4
        alien_model_name = config.actor_rollout_ref.rollout.context_template_alien_llm_model
        alien_model_response_length = config.actor_rollout_ref.rollout.context_template_alien_model_response_length
        regular_key_list = ["sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"]
        backup_key_list = ["sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"]
        for n_try in range(max_try):
            try:
                if n_try < max_try // 2:
                    api_key=random.choice(regular_key_list)
                elif n_try == max_try // 2:
                    api_key=random.choice(backup_key_list)
                else:
                    api_key=random.choice(regular_key_list + backup_key_list)
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
                sampling_params = dict(
                    n=1,
                    max_completion_tokens=alien_model_response_length,
                )
                sampling_params["temperature"] = 0
                completion = client.chat.completions.create(
                    model=alien_model_name,
                    messages=messages,
                    extra_body=sampling_params
                )
                message = completion.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
                if "content" not in message: message["content"] = ""
                return {"role": message["role"], "content": message['content']}
            except Exception as e:
                logger.bind(exception=True).exception(f"Error calling alien llm: {e}")
                time.sleep(5)
                print(f"Error calling alien llm: {e}, retrying...")
        raise RuntimeError(f"Failed to get response from alien llm after {max_try} attempts")
    return alien_llm_chat_fn


class SelfContextClipCMT(LinearThinkCMT):
    """
    A non-linear context manager template that handles the conversation flow between LLM and environment.
    """

    def __init__(self, config, tokenizer, llm_chat_fn):
        self.llm_chat_fn = llm_chat_fn
        self.alien_llm_chat_fn: Callable = construct_alien_llm_chat_fn(config, config.actor_rollout_ref.rollout)
        self.latest_env_response_id = ""
        self.latest_env_response_content = ""
        self.console_debug_mode = False
        self.force_think = config.actor_rollout_ref.rollout.force_think
        self.env_feedin_preference = config.env_service.env_feedin_preference
        self.train_sp_action = config.actor_rollout_ref.rollout.context_template_train_sp_action
        self.clipped_before = False
        if self.env_feedin_preference == "box":
            self.force_think_prompt = dedent("""
                Additional requirements: Think before action! You must think step by step before your next action, and you must use <think>...</think> to wrap your thinking process before finally produce your answer with \\box{}.
                Your thought (<think>...</think>) should be as short and concise as possible.
                For example:
                <think>...your thinking process...</think>
                \\box{...your final answer...}
            """)
        elif self.env_feedin_preference == "code":
            self.force_think_prompt = dedent("""
                Additional requirements: Think before action! You must think step by step before your next action, and you must use <think>...</think> to wrap your thinking process before finally produce the next-step action.
                Your thought (<think>...</think>) should be as short and concise as possible.
                For example:
                <think>...your thinking process...</think>
                ```python
                # your action here
                ```
            """)
        super().__init__(config, tokenizer)

    def post_tag_init_message_context(self, content, is_last) -> str:
        """
        Processes the content of a message, ensuring it is properly formatted and adding additional prompts if necessary.

        Args:
            content (str): The content of the message to be processed.
            is_last (bool): A flag indicating whether this is the last message in the sequence.

        Returns:
            str: The processed content of the message.
        """
        if is_last:
            content = content.strip()  # ⭐ Ensure the content is stripped of leading/trailing whitespace
        if is_last and self.force_think:
            content += self.force_think_prompt  # ⭐ Append the force_think_prompt if this is the last message and force_think is enabled
        return content.strip()  # ⭐ Return the final processed content

    def post_tag_env_message_context(self, content, turn, is_last) -> str:
        """
        Formats and tags a message from the environment, appending an identifier based on the turn number.
        Updates the internal state with the latest environment response and its ID.
        If the message is the last and `force_think` is enabled, appends a prompt for further thinking.

        Args:
            content (str): The content of the message from the environment.
            turn (int): The turn number of the message, must be in the range [0, 99).
            is_last (bool): A flag indicating if this is the last message in the conversation.

        Returns:
            str: The formatted and tagged message content.
        """
        from textwrap import dedent
        assert 0 <= turn < 99, "turn 必须在 [0, 99) 范围内"
        turn_id = f"{turn:03d}"
        self.latest_env_response_id = f"ER{turn_id}"  # ⭐ Update the latest environment response ID
        self.latest_env_response_content = content.strip()  # ⭐ Update the latest environment response content
        content = dedent(f"""
            [Environment Response, id=ER{turn_id}]
            ---
        """).strip() + '\n' + content.strip()  # ⭐ Format and tag the message content
        if is_last and self.force_think:
            content += self.force_think_prompt  # ⭐ Append the force think prompt if necessary
        return content

    def post_tag_llm_message_context(self, content, turn, is_last) -> str:
        """
        Formats the LLM's message by adding a specific tag and turn identifier.

        Args:
            content (str): The content of the LLM's message.
            turn (int): The turn number of the message in the conversation.
            is_last (bool): A flag indicating if this is the last message in the conversation.

        Returns:
            str: The formatted message with the added tag and turn identifier.
        """
        from textwrap import dedent
        assert not is_last, "llm message should never be last"  # ⭐ Ensures the LLM's message is not the last in the conversation
        assert 0 <= turn < 99, "turn 必须在 [0, 99) 范围内"  # ⭐ Validates the turn number is within the valid range
        turn_id = f"{turn:03d}"
        content = dedent(f"""
            [Assistant Response, id=AR{turn_id}]
            ---
        """).strip() + '\n' + content.strip()  # ⭐ Adds the tag and turn identifier to the message
        return content

    def strip_think_tags(self, text: str) -> str:
        """
        Removes <think> and </think> tags, along with their enclosed content, from the provided text.

        Args:
            text (str): The input text containing <think> and </think> tags.

        Returns:
            str: The text with all <think> and </think> tags and their enclosed content removed.
        """
        new_ext_msg_content = re.sub(r'\<think\>.*?\<\/think\>', '', text, flags=re.DOTALL).strip()  # ⭐ Remove content within <think> and </think> tags
        new_ext_msg_content = new_ext_msg_content.replace("<think>", "")  # Remove any remaining <think> tags
        new_ext_msg_content = new_ext_msg_content.replace("</think>", "")  # Remove any remaining </think> tags
        return new_ext_msg_content

    def prepare_next_llm_context(self):
        """
        Prepares the next context for the LLM by filtering and processing the previous messages.

        This function filters out non-deprecated context, processes each message based on its author,
        and formats the context appropriately for the next LLM interaction.

        Returns:
            list: A list of dictionaries representing the updated context.
        """
        self.latest_llm_interaction_socket = []

        # first we get all previous context (non-deprecated context)
        # get `init_message -> user -> llm -> user -> llm`` or `init_message -> llm -> user -> llm -> user``
        self.latest_llm_interaction_socket = self.filter_context_via_authors(["initialization", "llm", "env"])  # ⭐ Filter the context to include only relevant authors

        env_turn = 1
        llm_turn = 1
        for index, ext_msg in enumerate(list(self.latest_llm_interaction_socket)):
            is_last = (index == len(self.latest_llm_interaction_socket) - 1)
            # 根据消息类型进行处理
            if ext_msg.author == "llm":
                # 如果是以往的llm消息，去掉think标签
                new_ext_msg_content = self.strip_think_tags(ext_msg.content)
                author_override = "llm(do_not_train)"
                self.latest_llm_interaction_socket[index] = ExtendedMessage(
                    author=author_override,
                    role=ext_msg.role,
                    content=self.post_tag_llm_message_context(new_ext_msg_content, turn=llm_turn, is_last=is_last),
                    token_generator='auto',
                    tokenizer=self.tokenizer,
                    build_from_uuid=ext_msg.uuid,
                )
                llm_turn += 1

            # process env message
            elif ext_msg.author == "env":
                self.latest_llm_interaction_socket[index] = ExtendedMessage(
                    author=ext_msg.author,
                    role=ext_msg.role,
                    content=self.post_tag_env_message_context(content=ext_msg.content_for_future, turn=env_turn, is_last=is_last),
                    token_generator='auto',
                    tokenizer=self.tokenizer,
                    build_from_uuid=ext_msg.uuid,
                )
                env_turn += 1

            elif ext_msg.author in ["initialization"]:
                self.latest_llm_interaction_socket[index] = ExtendedMessage(
                    author=ext_msg.author,
                    role=ext_msg.role,
                    content=self.post_tag_init_message_context(content=ext_msg.content_for_future, is_last=is_last),
                    token_generator='auto',
                    tokenizer=self.tokenizer,
                    build_from_uuid=ext_msg.uuid,
                )

            else:
                raise RuntimeError(f"Unknown author {ext_msg.author} in latest_llm_interaction_socket")

        listofdict_context = self.to_role_content(self.latest_llm_interaction_socket)  # ⭐ Convert the processed context to a list of dictionaries
        return listofdict_context


    def save_init_input(self, init_input_arr:list, add_nothink):
        """
        Saves the initial input array by calling the parent class's method.

        Args:
            init_input_arr (list): The initial input array to be saved.
            add_nothink: Additional parameter passed to the parent method.
        """
        super().save_init_input(init_input_arr, add_nothink)  # ⭐ Calls the parent class's method to save the initial input
        return


    def impl_new_request_from_previous_interaction(self, new_message,  this_interaction, strip_think=False):
        """
        Processes a new request in the context of previous interactions, optionally stripping 'think' tags,
        and generates a new LLM response based on the updated context.

        Args:
            new_message: The new message to be added to the interaction.
            this_interaction: The list of previous interactions.
            strip_think (bool, optional): Whether to strip 'think' tags from the messages. Defaults to False.

        Returns:
            tuple: A tuple containing the updated interaction list and the LLM's output content.
        """
        latest_llm_interaction_socket_additional = copy.deepcopy(this_interaction)
        if strip_think:
            for index, ext_msg in enumerate(latest_llm_interaction_socket_additional):
                if ext_msg.author == "llm(do_not_train)" or ext_msg.author == "llm":
                    latest_llm_interaction_socket_additional[index] = ExtendedMessage(
                        author=ext_msg.author,
                        role=ext_msg.role,
                        content=self.strip_think_tags(ext_msg.content),
                        token_generator='auto',
                        tokenizer=self.tokenizer,
                        build_from_uuid=ext_msg.build_from_uuid if ext_msg.build_from_uuid else ext_msg.uuid,
                    )  # ⭐ Strips 'think' tags from the message content
                else:
                    continue
        latest_llm_interaction_socket_additional += [new_message]
        dict_context = self.to_role_content(latest_llm_interaction_socket_additional)
        if self.train_sp_action:
            llm_output = self.llm_chat_fn(dict_context, request_id="")  # ⭐ Generates LLM output using the current model
        else:
            llm_output = self.alien_llm_chat_fn(dict_context, request_id="")  # ⭐ Generates LLM output using an external model
        latest_llm_interaction_socket_additional += [self.save_llm_output_do_not_register_full_context(llm_output, dict_context)]  # ⭐ Adds the LLM output to the interaction history
        if self.train_sp_action:
            this_interaction = copy.deepcopy(latest_llm_interaction_socket_additional)
            self.grouped_steps += [this_interaction]  # ⭐ Updates the grouped steps with the latest interaction
        if self.console_debug_mode:
            print_listofdict(
                dict_context + [{'role': 'llm_latest', 'content': llm_output['content']}], mod='c'
            )  # ⭐ Prints the updated context and LLM output to the console
        else:
            print_listofdict(
                dict_context + [{'role': 'llm_latest', 'content': llm_output['content']}], mod='env_clip'
            )  # ⭐ Logs the updated context and LLM output to a file
        output_llm_content = llm_output['content'].strip()
        return latest_llm_interaction_socket_additional, output_llm_content


    def after_save_llm_output(self, this_interaction):
        """
        this_interaction = [
            init msg,
            ...,
            init msg,
            ...
            previous env msg,
            latest llm msg,
        ]
        """
        from textwrap import dedent
        if not self.latest_env_response_id:
            return

        clip_token_cnt = self.config.actor_rollout_ref.rollout.context_template_clip_trigger_token_num
        this_interaction = copy.deepcopy(this_interaction)
        if self._get_seq_length(this_interaction) < clip_token_cnt:
            return

        if self.clipped_before:
            return
        self.clipped_before = True

        _, generated_content = self.impl_new_request_from_previous_interaction(
            new_message=ExtendedMessage(
                author='user',
                role='user',
                content=dedent("""
                    Your new task is to inspect each `Environment Response` and `Assistant Response` messages,
                    and determine whether each message is useful for the next-step decision-making.
                    Generate a json structure following the format below:
                    ```json
                    [
                        {"id":"ARXXX or ERXXX", "useful":true or false, "action": "keep or remove or compress"},
                        ...,
                        {"id":"ARXXX or ERXXX", "useful":true or false, "action": "keep or remove or compress"},
                    ]
                    ```

                    For example:
                    ```json
                    [
                        {"id":"ER001", "useful":true, "action": "keep"},
                        {"id":"AR001", "useful":false, "action": "remove"},
                        ...
                    ]
                    ```

                    Rules:
                    - If the message contains useful information for future decisions, set "useful":true and "action":"keep".
                    - If the message records important previous action or environment feedback, set "useful":true and "action":"keep".
                    - If the message is very long and very redundant, set "useful":true and "action":"compress".
                    - If the message is completely irrelevant, set "useful":false and "action":"remove". Note that important failures should be preserved, because learning from past is vital.
                    - Ignore messages without id=XXX tags, where XXX is a 3-digit number.
                    - Ensure the JSON is properly formatted and valid.
                    - Remove or compress at least one message, because token limit is already reached.
                    - At least remove (or compress) one message.
                    - There must be no more than 2 "compress" actions in total, because "compress" action will cost considerable amount of time.

                """),
                token_generator='auto',
                tokenizer=self.tokenizer,
            ),
            this_interaction=this_interaction,
            strip_think=True,
        )

        try:
            llm_output_content = generated_content = generated_content.strip()
            if llm_output_content.count("```") == 2:
                extracted_content: str = llm_output_content.split("```")[1].strip()
            else:
                raise RuntimeError(f"Cannot find ``` in llm_output content: {llm_output_content}")
            if extracted_content.startswith('json'):
                extracted_content = extracted_content[len('json'):].strip()
            extracted_json = json.loads(extracted_content)
            for item in extracted_json:
                if 'id' not in item or 'useful' not in item or 'action' not in item:
                    raise RuntimeError(f"Each item must contain 'id', 'useful', and 'action' fields. Error in item: {item}")
                message_id = item['id']
                message_action = item['action']
                # find message from self.full_context
                ## match latest_llm_interaction_socket_additional and match self.full_context
                from_uuid = None
                for ext_msg in this_interaction:
                    if message_id in ext_msg.content_for_future:
                        from_uuid = ext_msg.build_from_uuid
                        break
                if from_uuid is None:
                    raise ValueError(f"Cannot find message_id {message_id} in `this_interaction`")
                target_msg = None
                target_index = -1
                for index, msg in enumerate(self.full_context):
                    if msg.uuid == from_uuid:
                        target_msg = msg
                        target_index = index
                        break
                if target_msg is None or target_index == -1:
                    raise ValueError(f"Cannot find message_id {message_id} in full_context")

                ## take actions
                if message_action == 'remove':
                    self.full_context[target_index] = ExtendedMessage(
                        author=target_msg.author+"(discard)",
                        role=target_msg.role,
                        content=target_msg.content,  # keep original content
                        token_generator='auto',
                        tokenizer=self.tokenizer,
                    )
                elif message_action == 'compress':
                    target_id = message_id
                    _, generated_compressed_content = self.impl_new_request_from_previous_interaction(
                        new_message=ExtendedMessage(
                            author='user',
                            role='user',
                            content=dedent(f"""
                                Your new task is to inspect {target_id}, and filter out all redundant information, and only keep the most important information that is useful for future decision-making.
                                For example, if the content is a long text with multiple paragraphs, you should only preseve the key paragraphs and use ... to replace the rest.
                                If the content is a long list of data / dict / json, you should only preseve the key items and use ... to replace the rest.
                                Be careful to preserve all information that might be useful in the future. You should at least reduce 50% of {target_id}.
                                Remember： wrap your answer with ```

                                Your response should be like:
                                ```
                                ...content after filtering...
                                ```
                            """),
                            token_generator='auto',
                            tokenizer=self.tokenizer,
                        ),
                        this_interaction=this_interaction[:-1], # exclude the latest llm message
                        strip_think=True,
                    )
                    if generated_compressed_content.count("```") != 2:
                        raise RuntimeError(f"Cannot find ``` in llm_output content: {generated_compressed_content}")
                    compressed_content = generated_compressed_content.split("```")[1].strip()
                    self.full_context[target_index] = ExtendedMessage(
                        author=target_msg.author,
                        role=target_msg.role,
                        content=compressed_content,
                        token_generator='auto',
                        tokenizer=self.tokenizer,
                    )
                elif message_action == 'keep':
                    continue
                else:
                    raise RuntimeError(f"Unknown action {message_action}, must be one of ['remove', 'keep', 'compress']")

        except Exception as e:
            logger.bind(exception=True).exception(f"Error processing llm_output: {e}")
            print(f"Error processing llm_output")
            return


    def replace_full_context_item(self, match_content: str, new_content: str):
        """
        Replaces the content of a message in the full context with new content if the match content is found.

        Args:
            match_content (str): The content to search for in the full context.
            new_content (str): The new content to replace the matched content with.

        Returns:
            bool: True if the replacement was successful, False otherwise.
        """
        success = False
        for index in range(len(self.full_context)):
            ext_msg = self.full_context[index]
            if match_content in ext_msg.content_for_future:
                success = True
                self.full_context[index] = ExtendedMessage(
                    author=ext_msg.author,
                    role=ext_msg.role,
                    content=new_content,
                    token_generator='auto',
                    tokenizer=self.tokenizer,
                )  # ⭐ Replace the matched content with the new content
                # print_dict({match_content: new_content})
                return success
        return success


    def save_llm_output(self, llm_output, input_msg_ref):
        """
        Saves the LLM's output, updates the grouped steps with the latest interaction, and resets the latest LLM interaction socket.

        Args:
            llm_output (str): The output generated by the LLM.
            input_msg_ref (str): The reference to the input message that triggered the LLM's response.

        Returns:
            None
        """
        ext_msg = Linear_CMT.save_llm_output(self, llm_output, input_msg_ref)  # ⭐ Save the LLM output and get the extended message
        this_interaction = copy.deepcopy(self.latest_llm_interaction_socket + [ext_msg])  # ⭐ Create a deep copy of the latest interaction including the new message
        self.grouped_steps += [this_interaction]  # ⭐ Append the new interaction to the grouped steps
        self.after_save_llm_output(this_interaction)  # ⭐ Call the after-save hook for any additional processing
        self.latest_llm_interaction_socket = []  # ⭐ Reset the latest LLM interaction socket
        return
