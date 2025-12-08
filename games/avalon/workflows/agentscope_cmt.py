# -*- coding: utf-8 -*-
"""CMT class for converting model_call_history from AgentScope workflow to CMT object."""
import json
import copy
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from agentevolver.module.context_manager.cmt_linear import Linear_CMT, ExtendedMessage
from agentevolver.schema.trajectory import Reward

if TYPE_CHECKING:
    from agentevolver.schema.trajectory import Sample


class AgentscopeCMT(Linear_CMT):
    """
    CMT class for converting model_call_history from AgentScope workflow to CMT object.
    
    This class takes model_call_history (list of dicts with 'prompt' and 'response')
    and converts them into ExtendedMessage objects in full_context, which can then
    be tokenized using group_tokenize() to generate Sample objects.
    
    Each prompt-response pair in model_call_history will generate one Sample.
    """
    
    def __init__(self, config, tokenizer, model_call_history: List[Dict[str, Any]], 
                 reward: Optional[Reward] = None, data_id: str = "", 
                 rollout_id: str = "", task_id: str = ""):
        """
        Initialize AgentscopeCMT with model_call_history.
        
        Args:
            config: Configuration object containing environment and model settings.
            tokenizer: Tokenizer instance for processing text.
            model_call_history: List of dicts, each containing 'prompt' (list of messages) and 'response'.
            reward: Optional Reward object for the trajectory.
            data_id: Data ID for the trajectory.
            rollout_id: Rollout ID for the trajectory.
            task_id: Task ID for the trajectory.
        """
        super().__init__(config, tokenizer)
        
        # Set trajectory attributes
        self.data_id = data_id
        self.rollout_id = rollout_id
        self.task_id = task_id
        self.reward = reward if reward is not None else Reward()
        self.is_terminated = True
        
        # Store model_call_history for processing in group_tokenize
        self._build_full_context_from_history(model_call_history)
    
    def _build_sample_from_call_record(self, call_record: Dict[str, Any], minor_index_id: int):
        """
        Build a Sample from a single call record (prompt-response pair).
        
        Args:
            call_record: Dict with 'prompt' (list of messages) and 'response'.
            minor_index_id: Minor index ID for the sample.
        
        Returns:
            Sample: A Sample object for this prompt-response pair.
        """
        from agentevolver.schema.trajectory import Sample
        
        prompt_messages = call_record.get("prompt", [])
        response = call_record.get("response", "")
        
        # Ensure prompt_messages is a list
        if not isinstance(prompt_messages, list):
            # Fallback: try to parse if it's a string
            if isinstance(prompt_messages, str):
                try:
                    prompt_messages = json.loads(prompt_messages)
                except (json.JSONDecodeError, TypeError):
                    prompt_messages = [{"role": "user", "content": prompt_messages}]
            else:
                prompt_messages = [prompt_messages]
        
        # Build full_context for this single prompt-response pair
        # Use token_generator="manual" since we need to compute incremental tokens
        # based on the full conversation context, not individual messages
        full_context = []
        
        # Create ExtendedMessage objects for prompt messages
        # author="initialization" ensures prompt messages do NOT participate in training
        # (need_training=False, loss_mask will be all zeros)
        for msg in prompt_messages:
            if isinstance(msg, dict):
                ext_msg = ExtendedMessage(
                    author="initialization",  # Non-trainable: need_training=False
                    role=msg.get("role", "user"),
                    content=str(msg.get("content", "")),
                    token_generator="manual",  # Use manual to compute context-based incremental tokens
                    tokenizer=self.tokenizer,
                )
                full_context.append(ext_msg)
        
        # Create ExtendedMessage for response
        # author="llm" ensures response message DOES participate in training
        # (need_training=True, loss_mask will be computed based on response tokens)
        ext_msg_response = ExtendedMessage(
            author="llm",  # Trainable: need_training=True
            role="assistant",
            content=str(response),
            token_generator="manual",  # Use manual to compute context-based incremental tokens
            tokenizer=self.tokenizer,
        )
        full_context.append(ext_msg_response)
        
        # Compute token arrays for all messages (similar to save_init_input)
        # This computes incremental tokens based on the full conversation context
        if len(full_context) > 0:
            token_ids_acc = []
            messages_so_far = []
            
            for ext_msg in full_context:
                # Build messages up to current point
                messages_so_far.append({
                    "role": ext_msg.role,
                    "content": ext_msg.content_for_future
                })
                
                # Apply chat template and tokenize
                text_with_chat_template = self.tokenizer.apply_chat_template(
                    messages_so_far, tokenize=False
                )
                tokenizer_output = self.tokenizer(
                    text_with_chat_template, return_tensors="pt", padding=False
                )
                input_ids = tokenizer_output["input_ids"][0].tolist()
                
                # Calculate incremental tokens (new tokens added by this message)
                input_id_increment = input_ids[len(token_ids_acc):]
                ext_msg.token_arr = input_id_increment
                token_ids_acc = input_ids
        
        # Tokenize the steps
        cmt_tokenized = self.tokenize_steps(ext_steps=full_context)
        
        # Create Sample
        sample = Sample(
            data_id=self.data_id,
            rollout_id=self.rollout_id,
            task_id=self.task_id,
            minor_index_id=minor_index_id,
            messages=self.to_role_content(full_context),
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
            reward_scores=self.reward.model_dump(),
            max_prompt_len=self.config.data.max_prompt_length,
            max_response_len=self.config.data.max_response_length,
            max_model_len=self.config.data.max_response_length + self.config.data.max_prompt_length,
        )
        sample.truncate_output_ids()
        
        return sample
    
    def group_tokenize(self):
        """
        Tokenize each prompt-response pair in model_call_history into a Sample.
        
        Returns:
            List[Sample]: A list of Sample objects, one for each prompt-response pair.
        """
        sample_arr = []
        
        for minor_index_id, call_record in enumerate(self.model_call_history):
            sample = self._build_sample_from_call_record(call_record, minor_index_id)
            sample_arr.append(sample)
        
        return sample_arr
    
    def _build_full_context_from_history(self, model_call_history: List[Dict[str, Any]]):
        """
        Build full_context from model_call_history.
        Store each prompt-response pair separately for group_tokenize to process.
        
        Args:
            model_call_history: List of dicts with 'prompt' (list of messages) and 'response'.
        """
        # Store model_call_history for later use in group_tokenize
        self.model_call_history = model_call_history
        # full_context will be built per sample in group_tokenize
        self.full_context = []

