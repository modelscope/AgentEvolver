import copy
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Literal

import numpy as np
import torch
import random
import re
from loguru import logger
from omegaconf import DictConfig
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from beyondagent.module.task_manager.reward import LlmAsJudgeRewardCalculator
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import (pad_sequence_to_length)

from beyondagent.module.agent_flow.agent_flow import AgentFlow
from beyondagent.module.agent_flow.base_agent_flow import BaseAgentFlow
from beyondagent.module.env_manager.env_worker import EnvWorker
from beyondagent.module.trainer.ba_async_llm_server_manager import BaAsyncLLMServerManager
from beyondagent.schema.task import Task
from beyondagent.schema.trajectory import Trajectory, Sample


class ParallelEnvManager(object):
    def __init__(self, config: DictConfig, async_rollout_manager: BaAsyncLLMServerManager, max_parallel: int,
                 max_llm_retries: int = 3, **kwargs):
        super().__init__(**kwargs)

        self.config: DictConfig = config
        self.async_rollout_manager: BaAsyncLLMServerManager = async_rollout_manager
        self.max_parallel: int = max_parallel
        self.max_llm_retries: int = max_llm_retries

        self.rollout_n = config.actor_rollout_ref.rollout.n
        self.model_name = self.async_rollout_manager.chat_scheduler.model_name
        self.tokenizer = self.async_rollout_manager.chat_scheduler.completion_callback.tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.rollout_config = config.actor_rollout_ref.rollout

        self.experience_template = config.hybrid_experience_training.experience_template

    def get_llm_chat_fn(self, sampling_params: dict = None) -> callable:
        def llm_chat(messages: List[Dict[str, str]],
                     custom_sampling_params: dict = None,
                     request_id: str = None) -> dict:
            """
            input messages: [{"role": "system", "value": "..."}, {"role": "user", "value": "..."}]
            output messages: [{"role": "assistant", "value": "..."}]
            """
            # TODO: sending sampling_params to rollout server
            updated_sampling_params = {}
            if sampling_params:
                updated_sampling_params.update(sampling_params)
            if custom_sampling_params:
                updated_sampling_params.update(custom_sampling_params)

            # output_messages = []
            input_messages = copy.deepcopy(messages)
            weighted_addresses = self.async_rollout_manager.chat_scheduler.weighted_addresses
            logger.info(f"weighted_addresses={weighted_addresses}")
            for i in range(self.max_llm_retries):
                try:
                    self.async_rollout_manager.submit_chat_completions(messages=input_messages,
                                                                       sampling_params=updated_sampling_params,
                                                                       request_id=request_id)
                    break

                except Exception as e:
                    logger.exception(f"rollout_server.{i} error: {e.args}")
                    time.sleep(i + 1)

            return input_messages[-1]

        return llm_chat

    def rollout_env_worker(self, task: Task, data_id: str, rollout_id: str, mode: Literal["sample", "validate"],
                           thread_index: int, add_exp: bool, task_train_exp_mode: str, **kwargs) -> Trajectory: # add add_exp & task_train_exp_mode by ANNI
        """
        Process a single prompt in a thread-safe way.
        """

        # TODO add try exception
        sampling_params = dict(
            n=1,
            max_completion_tokens=self.rollout_config.response_length,
            temperature=self.rollout_config.temperature,
            top_p=self.rollout_config.top_p)

        if mode == "validate":
            sampling_params["temperature"] = self.rollout_config.val_kwargs.temperature
            sampling_params["top_k"] = self.rollout_config.val_kwargs.top_k
            sampling_params["top_p"] = self.rollout_config.val_kwargs.top_p

        llm_chat_fn = self.get_llm_chat_fn(sampling_params)
        agent_flow: BaseAgentFlow = AgentFlow(
            reward_calculator=LlmAsJudgeRewardCalculator() if task.evaluator=='synthetic' else None, # TODO: better calculator injection
            llm_chat_fn=llm_chat_fn, 
            tokenizer=self.tokenizer, 
            config=self.config,
            **kwargs
        )

        # FIXME pass env_type & task_id
        env_worker = EnvWorker(task=task, thread_index=thread_index,
                               config=self.config)
        trajectory: Trajectory = env_worker.execute(data_id=data_id, rollout_id=rollout_id, add_exp=add_exp, task_train_exp_mode=task_train_exp_mode, agent_flow=agent_flow) # add add_exp & task_train_exp_mode by ANNI

        return trajectory

    def rollout(self, tasks: List[Task], mode: Literal["sample", "validate"], epoch: str) -> List[Trajectory]:
        trajectory_list: List[Trajectory] = []
        #############
        # ANNI 0814
        if mode == "validate":
            rollout_n = self.rollout_config.val_kwargs.n
            exp_mode = self.config.hybrid_experience_training.val_rollout_expmode
        else:
            rollout_n = self.rollout_n
            exp_mode = self.config.hybrid_experience_training.train_rollout_expmode
        add_exp_choices = {
            "woexp": [False] * rollout_n,
            "mixed": sorted([i < round(rollout_n*self.config.hybrid_experience_training.rollout_expratio) for i in range(rollout_n)], key=lambda _: random.random()),
            "all": [True] * rollout_n
        }[exp_mode]

        task_train_exp_modes = [
            task.metadata.get("task_train_exp_mode", "keep")
            for task in tasks
        ]   # len(tasks)个: task_train_exp_mode是query/task-level的

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = []
            for data_id, (task, task_train_exp_mode) in enumerate(zip(tasks, task_train_exp_modes)):
                for rollout_id in range(rollout_n):
                    thread_index = data_id * rollout_n + rollout_id
                    add_exp = add_exp_choices[rollout_id]
                    future = executor.submit(self.rollout_env_worker, task=task, data_id=str(data_id),
                                            rollout_id=str(rollout_id), mode=mode, thread_index=thread_index, add_exp=add_exp, task_train_exp_mode=task_train_exp_mode)  # add add_exp & task_train_exp_mode by ANNI
                    futures.append(future)
        #############
            for future in tqdm(futures, desc=f"epoch{epoch}.collect_rollout"):
                # do not fail silently
                result = future.result()
                trajectory_list.append(result)

            trajectory_list = sorted(trajectory_list, key=lambda x: (int(x.data_id), x.rollout_id))
            return trajectory_list

    # TODO: define an extra class for trajectory-dataproto converting.
    def to_dataproto(self, trajectories: List[Trajectory]) -> DataProto:
        """Convert trajectories to DataProto"""
        # Step 1: Convert trajectories to samples: tokenizing
        samples = self.trajectories_to_samples(trajectories)

        # Step 2: Convert samples to DataProto: padding
        dataproto = self.samples_to_dataproto(samples)
        
        return dataproto
    
    #############
    # ANNI 0825
    @staticmethod
    def extract_and_discard_experience(input_string, experience_template):
        pattern = re.escape(experience_template).replace(r'\{\}', '(.*?)')
        match = re.search(pattern, input_string)
        if match:
            experience = match.group(1)
            prompt = re.sub(pattern, '', input_string)
            return experience, prompt
        else:
            return "", input_string
    #############

    def trajectories_to_samples(self, trajectories: List[Trajectory]) -> List[Sample]:
        """Convert trajectories to samples"""
        samples = []
        for trajectory in trajectories:
            messages = trajectory.steps
            if len(messages) == 0:
                # Fixme: empty trajectory yunpeng
                sample = Sample(
                    data_id=trajectory.data_id,
                    rollout_id=trajectory.rollout_id,
                    messages=trajectory.steps,
                    reward_scores=trajectory.reward.model_dump()
                )
                sample.discard()
                samples.append(sample)
                continue
                # messages = [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]
            
            #############
            # ANNI 0825
            train_sample_exp_mode = trajectory.metadata.get("task_train_exp_mode", "keep")  # "keep" or "discard"
            # print(train_sample_exp_mode)

            if train_sample_exp_mode == "keep":
                experiences = [self.extract_and_discard_experience(msg["content"], self.experience_template)[0] for msg in messages]
            elif train_sample_exp_mode == "discard":
                experiences, prompts = zip(*[self.extract_and_discard_experience(msg["content"], self.experience_template) for msg in messages])
                for msg, prompt in zip(messages, prompts):
                    msg["content"] = prompt
            #############

            full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            outputs = self.tokenizer(full_text, return_tensors="pt", padding=False)
            input_ids = outputs["input_ids"][0].tolist()  # 移除batch维度
            attention_mask = outputs["attention_mask"][0].tolist()
            
            assert len(messages)>=2 and messages[0]["role"] == "system" and messages[1]["role"] == "user", "#message must >=2 and consists of system prompt + query prompt"
            prompt_text = self.tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
            prompt_outputs = self.tokenizer(prompt_text, return_tensors="pt", padding=False)
            prompt_ids = prompt_outputs["input_ids"][0].tolist()
            prompt_attention_mask = prompt_outputs["attention_mask"][0].tolist()

            response_ids = input_ids[len(prompt_ids):]
            response_attention_mask = attention_mask[len(prompt_attention_mask):]

            position_ids = compute_position_id_with_mask(torch.tensor(attention_mask)).tolist()
            prompt_position_ids = position_ids[:len(prompt_ids)]
            response_position_ids = position_ids[len(prompt_ids):]
            
            # 生成loss mask (仅在response部分计算loss，但需要在response部分mask env的输出)
            prompt_loss_mask = [0] * len(prompt_ids)
            response_loss_mask = [1] * len(response_ids)

            response_token_ids_idxs = []
            human_token_ids_idxs = []

            response_ids_np = np.array(response_ids)

            self.instruction_template_ids = self.tokenizer.encode("<|im_start|>user\n")
            self.response_template_ids = self.tokenizer.encode("<|im_start|>assistant\n")

            for assistant_idx in np.where(response_ids_np == self.response_template_ids[0])[0]:
                if self.response_template_ids == response_ids_np[assistant_idx: assistant_idx + len(
                        self.response_template_ids)].tolist():
                    response_token_ids_idxs.append(assistant_idx + len(self.response_template_ids))

            for human_idx in np.where(response_ids_np == self.instruction_template_ids[0])[0]:
                if self.instruction_template_ids == response_ids_np[human_idx: human_idx + len(self.instruction_template_ids)].tolist():
                    human_token_ids_idxs.append(human_idx)

            if (
                len(human_token_ids_idxs) > 0
                and len(response_token_ids_idxs) > 0
                and human_token_ids_idxs[0] > response_token_ids_idxs[0]
            ):
                human_token_ids_idxs = [0] + human_token_ids_idxs

            for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                response_loss_mask[start:end] = [0] * (end-start)

            if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                response_loss_mask[human_token_ids_idxs[-1]:] = [0] * (len(response_loss_mask)-human_token_ids_idxs[-1])

            loss_mask = prompt_loss_mask + response_loss_mask

            sample = Sample(
                data_id=trajectory.data_id,
                rollout_id=trajectory.rollout_id,
                messages=trajectory.steps,
                input_ids=input_ids,
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                attention_mask=attention_mask,
                prompt_attention_mask=prompt_attention_mask,
                response_attention_mask=response_attention_mask,
                loss_mask=loss_mask,
                prompt_loss_mask=prompt_loss_mask,
                response_loss_mask=response_loss_mask,
                position_ids=position_ids,
                prompt_position_ids=prompt_position_ids,
                response_position_ids=response_position_ids,
                reward_scores=trajectory.reward.model_dump(),
                max_prompt_len=self.config.data.max_prompt_length,
                max_response_len=self.config.data.max_response_length,
                extras={
                    "add_exp": trajectory.metadata.get("add_exp", None),
                    # Flag for experience incorporation

                    "task_train_expmode": trajectory.metadata.get("task_train_exp_mode", None), 
                    # Mode for handling experience during trajectory-to-sample conversion:
                    # Specifies whether to retain or discard experience data
                    
                    "experience": experiences   
                    # List of experience values associated with the trajectory
                },
            )
            sample.truncate_output_ids()
            samples.append(sample)
        
        return samples

    def samples_to_dataproto(self, samples: list[Sample]) -> DataProto:
        # Initialize lists to store batched data
        prompt_ids, response_ids = [], []
        prompt_attention_mask, response_attention_mask = [], []
        prompt_position_ids, response_position_ids = [], []
        prompt_loss_mask, response_loss_mask = [], []
        messages = []
        reward_scores = []
        extras = [] # List of dictionaries containing supplementary data for each trajectory, including "add_exp", "task_train_expmode", "experience"
        exp_mask_list = []  # List of binary masks indicating whether to consider off_clip_high for each sample in the batch
        
        for sample in samples:
            # Validate that all fields have the same length
            assert len(sample.input_ids) == len(sample.attention_mask) == len(sample.position_ids) == len(
                sample.loss_mask), f"Sample {sample.request_id} has mismatched lengths: " \
                                f"{len(sample.input_ids)=}, {len(sample.attention_mask)=}, " \
                                f"{len(sample.position_ids)=}, {len(sample.loss_mask)=}"

            # Discard samples with prompt length exceeding limit
            if len(sample.prompt_ids) > self.config.data.max_prompt_length:
                logger.warning(
                    f"Sample {sample.request_id} has prompt_ids length {len(sample.prompt_ids)} "
                    f"greater than max_prompt_length {self.config.data.max_prompt_length}, discarding."
                )
                sample.discard()
                continue

            # Warn if response is longer than expected (but still include it)
            if len(sample.response_ids) > self.config.data.max_response_length:
                logger.warning(
                    f"Sample {sample.request_id} has response_ids length {len(sample.response_ids)} "
                    f"greater than max_response_length {self.config.data.max_response_length}."
                )

            # Append tensors to respective lists
            prompt_ids.append(torch.tensor(sample.prompt_ids, dtype=torch.int))
            response_ids.append(torch.tensor(sample.response_ids, dtype=torch.int))

            prompt_attention_mask.append(torch.tensor(sample.prompt_attention_mask, dtype=torch.int))
            response_attention_mask.append(torch.tensor(sample.response_attention_mask, dtype=torch.int))

            prompt_position_ids.append(torch.tensor(sample.prompt_position_ids, dtype=torch.int))
            response_position_ids.append(torch.tensor(sample.response_position_ids, dtype=torch.int))

            prompt_loss_mask.append(torch.tensor(sample.prompt_loss_mask, dtype=torch.int))
            response_loss_mask.append(torch.tensor(sample.response_loss_mask, dtype=torch.int))

            messages.append({"messages": sample.messages})
            reward_scores.append(sample.reward_scores)
            extras.append(sample.extras)

            # Create experience mask: 1 if off_clip_high conditions met (add_exp=True, task_train_expmode="discard"), else 0
            if sample.extras.get("add_exp", False) and sample.extras.get("task_train_expmode", None)=="discard":
                exp_mask_list.append(torch.ones(len(sample.loss_mask), dtype=torch.int))
            else:
                exp_mask_list.append(torch.zeros(len(sample.loss_mask), dtype=torch.int))



        # Batch and pad sequences
        prompt_ids = pad_sequence(prompt_ids, batch_first=True, padding_value=self.pad_token_id, padding_side="left")
        prompt_ids = pad_sequence_to_length(prompt_ids, self.config.data.max_prompt_length, self.pad_token_id, left_pad=True)

        response_ids = pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
        response_ids = pad_sequence_to_length(response_ids, self.config.data.max_response_length, self.pad_token_id)

        prompt_attention_mask = pad_sequence(prompt_attention_mask, batch_first=True, padding_value=0, padding_side="left")
        prompt_attention_mask = pad_sequence_to_length(prompt_attention_mask, self.config.data.max_prompt_length, 0,
                                                    left_pad=True)

        response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
        response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.data.max_response_length, 0)

        prompt_position_ids = pad_sequence(prompt_position_ids, batch_first=True, padding_value=0, padding_side="left")
        prompt_position_ids = pad_sequence_to_length(prompt_position_ids, self.config.data.max_prompt_length, 0,
                                                    left_pad=True)

        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=response_ids.device).unsqueeze(0).repeat(
            len(samples), 1)
        response_position_ids = prompt_position_ids[:, -1:] + delta_position_id

        prompt_loss_mask = pad_sequence(prompt_loss_mask, batch_first=True, padding_value=0, padding_side="left")
        prompt_loss_mask = pad_sequence_to_length(prompt_loss_mask, self.config.data.max_prompt_length, 0, left_pad=True)

        response_loss_mask = pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
        response_loss_mask = pad_sequence_to_length(response_loss_mask, self.config.data.max_response_length, 0)

        exp_mask = pad_sequence(exp_mask_list, batch_first=True, padding_value=0)
        exp_mask = pad_sequence_to_length(exp_mask, self.config.data.max_prompt_length + self.config.data.max_response_length, 0)

        # Concatenate prompt and response tensors
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)
        loss_mask = torch.cat((prompt_loss_mask, response_loss_mask), dim=-1)

				# Validate masks have same shape
        assert exp_mask.shape == loss_mask.shape, f"Shape mismatch: {exp_mask.shape} vs {loss_mask.shape}"

        # Construct the batch using TensorDict
        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
                "exp_mask": exp_mask        # add exp_mask by ANNI
            },
            batch_size=len(samples),
        )

        return DataProto(batch=batch, non_tensor_batch={"messages": np.array(messages), "reward_scores": np.array(reward_scores), "extras": np.array(extras)})  # add extras by ANNI