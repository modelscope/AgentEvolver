"""
Sliding-window context manager (SeeUPO).

Compression policy mixes:
- Token budget: same as LinearThinkCMT (`max_model_len - response_length`), compress until the
  post-processed chat template fits.
- Tail retention (AgentEvolver games/agents/memory/SlidingWindowMemory.py style): optionally cap
  how many logical messages we keep before summarizing; each compress keeps initialization,
  a single placeholder ``memory`` user message, and the last K LLM + (K+1) env turns.

See ``cmt_linear_think.LinearThinkCMT`` for think stripping, loss masking, and ``group_tokenize``.
"""

from __future__ import annotations

import copy
import re
from typing import List

from context_manager_templates.cmt_linear_think import ExtendedMessage, LinearThinkCMT


_TRAINING_AUTHORS = frozenset(
    {"initialization", "memory", "llm", "llm(do_not_train)", "env"}
)


class SlidingWindowCMT(LinearThinkCMT):
    """
    LinearThinkCMT + sliding-window compression when the prompt would exceed ``max_seq_length``.
    """

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        rollout = self.config.actor_rollout_ref.rollout
        self.recall_x_action: int = int(getattr(rollout, "sliding_window_keep_llm_rounds", 2))
        self.sliding_window_max_messages: int = int(getattr(rollout, "sliding_window_max_messages", 0))
        self.omitted_msg_so_far: int = 0
        self.env_cnt: int = 0
        self.llm_cnt: int = 0

    def _ordered_training_messages(self) -> List[ExtendedMessage]:
        return [m for m in self.full_context if m.author in _TRAINING_AUTHORS]

    def _compress_full_context_once(self) -> bool:
        """
        Drop middle context: keep all ``initialization``, last K LLM + (K+1) env messages (chronological),
        replace everything else with one new ``memory`` placeholder. All surviving ``llm`` become
        ``llm(do_not_train)``. Returns False if nothing can be removed.
        """
        ordered = self._ordered_training_messages()
        if not ordered:
            return False

        recall_x = max(self.recall_x_action, 1)
        llm_indices = [i for i, m in enumerate(ordered) if m.author in ("llm", "llm(do_not_train)")]
        env_indices = [i for i, m in enumerate(ordered) if m.author == "env"]
        init_indices = [i for i, m in enumerate(ordered) if m.author == "initialization"]

        keep_idx: set[int] = set(init_indices)
        for i in llm_indices[-recall_x:]:
            keep_idx.add(i)
        for i in env_indices[-(recall_x + 1) :]:
            keep_idx.add(i)

        omitted_indices = [i for i in range(len(ordered)) if i not in keep_idx]
        if not omitted_indices:
            return False

        self.omitted_msg_so_far += len(omitted_indices)

        preserved = [ordered[i] for i in sorted(keep_idx)]
        init_msgs = [m for m in preserved if m.author == "initialization"]
        tail = [m for m in preserved if m.author != "initialization"]

        memory_msg = self._create_sliding_memory_message(len(omitted_indices))
        new_context: List[ExtendedMessage] = []
        new_context.extend(init_msgs)
        new_context.append(memory_msg)
        for ext_msg in tail:
            if ext_msg.author == "llm":
                new_context.append(
                    ExtendedMessage(
                        author="llm(do_not_train)",
                        role=ext_msg.role,
                        content=ext_msg.content_for_future,
                        token_generator="auto",
                        tokenizer=self.tokenizer,
                    )
                )
            elif ext_msg.author == "llm(do_not_train)":
                new_context.append(
                    ExtendedMessage(
                        author="llm(do_not_train)",
                        role=ext_msg.role,
                        content=ext_msg.content_for_future,
                        token_generator="auto",
                        tokenizer=self.tokenizer,
                    )
                )
            else:
                new_context.append(ext_msg)

        self.full_context = new_context
        return True

    def _create_sliding_memory_message(self, dropped_message_count: int) -> ExtendedMessage:
        rounds = max(dropped_message_count // 2, 1)
        return ExtendedMessage(
            author="memory",
            role="user",
            content=f"[Previous {rounds} round(s) of conversation were omitted for brevity.]",
            token_generator="auto",
            tokenizer=self.tokenizer,
        )

    def _socket_dict_under_budget(self, socket: List[ExtendedMessage]) -> bool:
        dict_context = self.to_role_content(socket)
        return self._get_seq_length(dict_context) < self.max_seq_length

    def _compress_until_under_budget(self, max_iters: int = 64) -> None:
        """AgentEvolver-style tail cap (optional), then token-driven compress."""
        for _ in range(max_iters):
            if (
                self.sliding_window_max_messages > 0
                and len(self._ordered_training_messages()) > self.sliding_window_max_messages
            ):
                if not self._compress_full_context_once():
                    break
                continue
            socket = self.filter_context_via_authors(
                ["initialization", "memory", "llm", "llm(do_not_train)", "env"]
            )
            if self._socket_dict_under_budget(socket):
                return
            if not self._compress_full_context_once():
                return

    def prepare_next_llm_context(self):
        self._compress_until_under_budget()

        self.latest_llm_interaction_socket = []
        self.latest_llm_interaction_socket = self.filter_context_via_authors(
            ["initialization", "memory", "llm", "llm(do_not_train)", "env"]
        )

        for index, ext_msg in enumerate(list(self.latest_llm_interaction_socket)):
            is_last = index == len(self.latest_llm_interaction_socket) - 1
            if ext_msg.author in ("llm", "llm(do_not_train)"):
                new_ext_msg_content = re.sub(r"<think>.*?</think>", "", ext_msg.content, flags=re.DOTALL).strip()
                new_ext_msg_content = new_ext_msg_content.replace("<think>", "")
                new_ext_msg_content = new_ext_msg_content.replace("</think>", "")
                if self.config.actor_rollout_ref.rollout.train_history_infer_token:
                    self.latest_llm_interaction_socket[index] = ExtendedMessage(
                        author="llm",
                        role=ext_msg.role,
                        content=new_ext_msg_content,
                        token_generator="auto",
                        tokenizer=self.tokenizer,
                    )
                else:
                    self.latest_llm_interaction_socket[index] = ExtendedMessage(
                        author="llm(do_not_train)",
                        role=ext_msg.role,
                        content=new_ext_msg_content,
                        token_generator="auto",
                        tokenizer=self.tokenizer,
                    )
            elif ext_msg.author in ("env", "initialization", "memory"):
                if self.config.actor_rollout_ref.rollout.train_history_infer_token:
                    if not is_last:
                        self.latest_llm_interaction_socket[index] = ExtendedMessage(
                            author=ext_msg.author,
                            role=ext_msg.role,
                            content=ext_msg.content_for_future + "\n/no_think",
                            token_generator="auto",
                            tokenizer=self.tokenizer,
                        )
                    else:
                        self.latest_llm_interaction_socket[index] = ExtendedMessage(
                            author=ext_msg.author,
                            role=ext_msg.role,
                            content=ext_msg.content_for_future + self.think_hint,
                            token_generator="auto",
                            tokenizer=self.tokenizer,
                        )
                else:
                    if not is_last:
                        self.latest_llm_interaction_socket[index] = ExtendedMessage(
                            author=ext_msg.author,
                            role=ext_msg.role,
                            content=ext_msg.content_for_future,
                            token_generator="auto",
                            tokenizer=self.tokenizer,
                        )
                    else:
                        self.latest_llm_interaction_socket[index] = ExtendedMessage(
                            author=ext_msg.author,
                            role=ext_msg.role,
                            content=ext_msg.content_for_future + self.think_hint,
                            token_generator="auto",
                            tokenizer=self.tokenizer,
                        )
            else:
                raise RuntimeError(f"Unknown author {ext_msg.author} in latest_llm_interaction_socket")

        dict_context = self.to_role_content(self.latest_llm_interaction_socket)
        return dict_context

    def check_context_token_num_safe(self, messages: List[dict]) -> bool:
        return self._get_seq_length(messages) < self.max_seq_length

    def save_env_output(self, env_output, input_msg_ref=None, add_nothink=False):
        self.env_cnt += 1
        env_output["content"] = f"[Current Env Step {self.env_cnt}]\n\n" + env_output["content"]
        return super().save_env_output(env_output, input_msg_ref, add_nothink)

    def save_llm_output(self, llm_output, input_msg_ref):
        self.llm_cnt += 1
        super().save_llm_output(llm_output, input_msg_ref)

    def ensure_terminate_rollout_stage(self):
        socket = self.filter_context_via_authors(
            ["initialization", "memory", "llm", "llm(do_not_train)", "env"]
        )
        if any(ext_msg.need_training for ext_msg in socket):
            self.grouped_steps += [copy.deepcopy(socket)]
