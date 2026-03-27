import torch
import copy
from typing import List
from beyondagent.schema.trajectory import Sample
from best_logger import print_dict, print_listofdict
from beyondagent.utils.vscode_breakpoint import vscode_conditional_breakpoint
from context_manager_templates.cmt_linear_think import ExtendedMessage, CMTLinear, LinearThinkCMT
from context_manager_templates.cmt_linear import find_sublist_indices, replace_token_ids
from best_logger import register_logger, print_dict, print_nested, NestedJsonItem, SeqItem


class SelfContextAwareCMT(LinearThinkCMT):
    """
    A non-linear context manager template that handles the conversation flow between LLM and environment.
    """

    def __init__(self, config, tokenizer, llm_chat_fn):
        self.llm_chat_fn = llm_chat_fn
        self.latest_env_response_id = ""
        self.latest_env_response_content = ""
        self.console_debug_mode = False
        self.force_think = config.actor_rollout_ref.rollout.force_think
        super().__init__(config, tokenizer)


    def post_tag_env_message_context(self, content, turn, is_last) -> str:
        from textwrap import dedent
        assert 0 <= turn < 99, "turn 必须在 [0, 99) 范围内"
        turn_id = f"{turn:03d}"  # 等效：str(turn).zfill(3)
        self.latest_env_response_id = f"ER{turn_id}"
        self.latest_env_response_content = content.strip()
        content = dedent(f"""
            [Environment Response, id="ER{turn_id}"]
            ---
        """).strip() + content.strip()
        if is_last and self.force_think:
            content += "\n\nAdditional requirements: \n- You must think step by step before your next action, and you must use <think>...</think> to wrap your thinking process before finally produce your answer with \\box{}. (Put \\box{} outside <think>...</think>)."

        return content

    def post_tag_init_message_context(self, content, is_last) -> str:
        if is_last:
            content = content.strip() # + "\nSome additional requirements for last msg \n"
        if is_last and self.force_think:
            content += "\n\nAdditional requirements: \n- You must think step by step before your next action, and you must use <think>...</think> to wrap your thinking process before finally produce your answer with \\box{}. (Put \\box{} outside <think>...</think>)."
        return content.strip()

    def prepare_next_llm_context(self):
        self.latest_llm_interaction_socket = []

        # first we get all previous context (non-deprecated context)
        # get `init_message -> user -> llm -> user -> llm`` or `init_message -> llm -> user -> llm -> user``
        self.latest_llm_interaction_socket = self.filter_context_via_authors(["initialization", "llm", "env"])


        env_turn = 1
        for index, ext_msg in enumerate(list(self.latest_llm_interaction_socket)):

            is_last = (index == len(self.latest_llm_interaction_socket) - 1)
            # 根据消息类型进行处理
            if ext_msg.author == "llm":
                # 如果是以往的llm消息，去掉think标签
                import re
                new_ext_msg_content = re.sub(r'<think>.*?</think>', '', ext_msg.content, flags=re.DOTALL).strip()
                new_ext_msg_content = new_ext_msg_content.replace("<think>", "")
                new_ext_msg_content = new_ext_msg_content.replace("</think>", "")

                assert ext_msg.author == "llm"
                author_override = "llm(do_not_train)"
                self.latest_llm_interaction_socket[index] = ExtendedMessage(
                    author=author_override,
                    role=ext_msg.role,
                    content=new_ext_msg_content,
                    token_generator='auto',
                    tokenizer=self.tokenizer,
                )

            # process env message
            elif ext_msg.author == "env":
                self.latest_llm_interaction_socket[index] = ExtendedMessage(
                    author=ext_msg.author,
                    role=ext_msg.role,
                    content=self.post_tag_env_message_context(content=ext_msg.content_for_future, turn=env_turn, is_last=is_last),
                    token_generator='auto',
                    tokenizer=self.tokenizer,
                )
                env_turn += 1

            elif ext_msg.author in ["initialization"]:
                self.latest_llm_interaction_socket[index] = ExtendedMessage(
                    author=ext_msg.author,
                    role=ext_msg.role,
                    content=self.post_tag_init_message_context(content=ext_msg.content_for_future, is_last=is_last),
                    token_generator='auto',
                    tokenizer=self.tokenizer,
                )

            else:
                raise RuntimeError(f"Unknown author {ext_msg.author} in latest_llm_interaction_socket")

        dict_context = self.to_role_content(self.latest_llm_interaction_socket)
        return dict_context


    def save_init_input(self, init_input_arr:list, add_nothink):
        super().save_init_input(init_input_arr, add_nothink)
        return


    def after_save_llm_output(self, llm_output, this_interaction):
        if not self.latest_env_response_id:
            return
        self.latest_llm_interaction_socket_additional = copy.deepcopy(this_interaction)
        self.latest_llm_interaction_socket_additional += [ExtendedMessage(
            author='user',
            role='user',
            content=f"""Now your new task is to inspect `Environment Response` {self.latest_env_response_id} and then extract paragraphs that may be useful information in last action or in the future."""
                     """For example, if the original Response contain paragraph ABCDEF and only paragraph ABCF maybe useful, you should answer me by copying paragraph ABCF (wrapped them between ```)."""
                     """Do not give up details easily, try your best to find useful information. When necessary, you can preserve everything.""",
            token_generator='auto',
            tokenizer=self.tokenizer,
        )]
        dict_context = self.to_role_content(self.latest_llm_interaction_socket_additional)
        llm_output = self.llm_chat_fn(dict_context, request_id="")
        self.latest_llm_interaction_socket_additional += [self.save_llm_output_do_not_register_full_context(llm_output, dict_context)]
        this_interaction = copy.deepcopy(self.latest_llm_interaction_socket_additional)
        self.grouped_steps += [this_interaction]


        if self.console_debug_mode:
            print_listofdict(
                dict_context +
                [{'role': 'llm_latest', 'content': llm_output['content']}]
            , mod='c')
        try:
            llm_output_content = llm_output['content'] = llm_output['content'].strip()
            if llm_output_content.count("```") == 2:
                extracted_content: str = llm_output_content.split("```")[1].strip()
            else:
                raise RuntimeError(f"Cannot find ``` in llm_output content: {llm_output_content}")

            # override future full_context
            assert self.latest_env_response_content != ''
            replace_success = self.replace_full_context_item(match_content=self.latest_env_response_content, new_content=extracted_content)
            if not replace_success:
                raise RuntimeError(f"Cannot find {self.latest_env_response_id} in full_context")

        except Exception as e:
            print(f"Error processing llm_output")
            return

    def replace_full_context_item(self, match_content: str, new_content: str):
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
                )
                # print_dict({match_content: new_content})
                return success
        return success

    def save_llm_output(self, llm_output, input_msg_ref):
        if llm_output['content'].strip() == "":
            try:
                from vsdb import bp
            except ImportError:
                def bp(*args, **kwargs):
                    pass
            bp('empty')
        ext_msg = CMTLinear.save_llm_output(self, llm_output, input_msg_ref)
        this_interaction = copy.deepcopy(self.latest_llm_interaction_socket + [ext_msg])
        self.grouped_steps += [this_interaction]
        self.after_save_llm_output(llm_output, this_interaction)
        self.latest_llm_interaction_socket = []
        return
