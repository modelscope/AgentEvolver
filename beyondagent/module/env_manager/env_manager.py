import copy
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Dict, List, Literal, Tuple

import numpy as np
import torch
import random
import re
import os
from loguru import logger
from omegaconf import DictConfig
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import (pad_sequence_to_length)

from beyondagent.module.agent_flow.agent_flow import AgentFlow
from beyondagent.module.agent_flow.base_agent_flow import BaseAgentFlow
from beyondagent.module.env_manager.env_worker import EnvWorker
from beyondagent.module.trainer.ba_async_llm_server_manager import BaAsyncLLMServerManager
from beyondagent.module.task_manager.rewards import grader_manager
from beyondagent.schema.task import Task
from beyondagent.schema.trajectory import Trajectory, Sample
# do not delete this line
from beyondagent.module.task_manager.rewards import LlmAsJudgeRewardCalculator,LlmAsJudgeRewardCalculatorWithGT,LlmAsJudgeBinaryRewardCalculator,LlmAsJudgeBinaryRewardCalculatorWithGT,EnvGrader, AvgBinaryGTJudge, AvgLlmJudge
from beast_logger import register_logger

def init_logger(experiment_name):
    """Initialize the logger with the given configuration."""
    if 'BEST_LOGGER_INIT' in os.environ: return # prevent re-initialization in ray environment
    os.environ['BEST_LOGGER_INIT'] = '1'
    os.environ['BEST_LOGGER_WEB_SERVICE_URL'] = "http://127.0.0.1:8181/"
    from datetime import datetime
    final_log_path = os.path.join( "./logs", datetime.now().strftime("%Y_%m_%d_%H_%M") + '_' + experiment_name )
    non_console_mods = ["conversation", "rollout", "token_clip", "bad_case", "env_clip"]
    register_logger(mods=["evaluation", "exception"], non_console_mods=non_console_mods, auto_clean_mods=[], base_log_path=final_log_path, debug=False)
    print('Run `beast_logger_install` and click the url to inspect rollout logs. Continue in 5 seconds')
    time.sleep(5)


class ParallelEnvManager(object):
    def __init__(self, config: DictConfig, async_rollout_manager: BaAsyncLLMServerManager, max_parallel: int,
                 max_llm_retries: int = 3, **kwargs):
        init_logger(experiment_name=config.trainer.experiment_name)
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
        self.llm_mode = "local" # use fsdp worker ("local") or use foreign server ("remote")
        self.current_token = 0
        self.current_token_count_time = time.time()


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
            updated_sampling_params.update({"logprobs": 1, "return_tokens_as_token_ids": True})

            # output_messages = []
            input_messages = copy.deepcopy(messages)
            weighted_addresses = self.async_rollout_manager.chat_scheduler.weighted_addresses
            # logger.info(f"weighted_addresses={weighted_addresses}")
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

        def llm_chat_remote(messages: List[Dict[str, str]],
                     custom_sampling_params: dict = None,
                     request_id: str = None) -> dict:
            """
            input messages: [{"role": "system", "value": "..."}, {"role": "user", "value": "..."}]
            output messages: [{"role": "assistant", "value": "..."}]
            """
            updated_sampling_params = {}
            if sampling_params:
                updated_sampling_params.update(sampling_params)
            if custom_sampling_params:
                updated_sampling_params.update(custom_sampling_params)
            updated_sampling_params.update({"logprobs": 1, "return_tokens_as_token_ids": True})
            input_messages = copy.deepcopy(messages)
            for i in range(self.max_llm_retries):
                try:
                    output_message = self.async_rollout_manager.submit_chat_completions(messages=input_messages,
                                                                       sampling_params=updated_sampling_params,
                                                                       request_id=request_id)
                    break
                except Exception as e:
                    logger.exception(f"rollout_server.{i} error: {e.args}")
                    time.sleep(2**i)
            return output_message[-1]

        if self.llm_mode == "remote":
            return llm_chat_remote
        else:
            return llm_chat

    def step_status_printer(self, tmux):
        # 直方数据，tmux 0~10 数量 10~20 数量 20~30 数量 30~40 数量 ……
        step_counter = {}

        current_token = sum(tmux['token'])
        current_time = time.time()
        delta_token = current_token - self.current_token
        delta_time = current_time - self.current_token_count_time
        self.current_token = current_token
        self.current_token_count_time = current_time
        token_gen_per_sec_str = f"{delta_token/delta_time:.2f} tokens/s" if delta_time > 0 else "N/A"


        for step in tmux['step']:
            if step == -1:
                step_counter[(-1, 'terminated')] = step_counter.get((-1, 'terminated'), 0) + 1
                continue
            else:
                start = (step // 5) * 5
                end = start + 5
                step_counter[(start, end)] = step_counter.get((start, end), 0) + 1

        # sort by start value (small to large)
        step_counter = dict(sorted(step_counter.items(), key=lambda x: x[0][0]))

        print_buf = []
        for (start, end), count in step_counter.items():
            if start != -1:
                print_buf += [f"[{start}-{end}]:{count} threads"]
        for (start, end), count in step_counter.items():
            if start == -1:
                print_buf += [f"[finished]:{count} threads"]
        print(f"Rollout progress ({token_gen_per_sec_str}): " + "  //  ".join(print_buf))



    def rollout_env_worker(self, task: Task, data_id: str, rollout_id: str, mode: Literal["sample", "validate"],
                           thread_index: int, add_exp: bool, task_train_exp_mode: str, tmux: dict, stop:list, **kwargs) -> Trajectory: # add add_exp & task_train_exp_mode by ANNI
        """
        Process a single prompt in a thread-safe way.
        """
        max_retry = 4
        for retry in range(max_retry):
            try:

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
                reward_caculator=grader_manager.get_calculator(task.evaluator, task=task)
                agent_flow: BaseAgentFlow = AgentFlow(
                    reward_calculator=reward_caculator,
                    llm_chat_fn=llm_chat_fn, 
                    tokenizer=self.tokenizer, 
                    config=self.config,
                    **kwargs
                )

                env_worker = EnvWorker(task=task, thread_index=thread_index, config=self.config, tokenizer=self.tokenizer)
                trajectory: Trajectory = env_worker.execute(data_id=data_id, rollout_id=rollout_id, add_exp=add_exp, task_train_exp_mode=task_train_exp_mode, agent_flow=agent_flow, tmux=tmux, stop=stop) # add add_exp & task_train_exp_mode by ANNI
                return trajectory

            except Exception as e:
                if retry < max_retry - 1:
                    logger.bind(exception=True).exception(f"rollout_env_worker error: {e.args}, retrying {retry + 1}/{max_retry}")
                    time.sleep(2 ** retry)
                else:
                    logger.bind(exception=True).exception(f"rollout_env_worker failed after {max_retry} retries: {e.args}")
                    raise e

    def rollout(self, tasks: List[Task], mode: Literal["sample", "validate"], epoch: str) -> List[Trajectory]:
        """
        使用线程池执行rollout任务，并自动重试失败的任务，直到所有任务成功。

        Args:
            tasks: 待处理的任务列表。
            mode: 模式，'sample' 或 'validate'。
            epoch: 当前的周期标识，用于日志和进度条。

        Returns:
            一个包含所有成功任务结果的Trajectory列表，并已排序。
        """
        traj_cmt_array = []
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
        
        
        # 1. 核心数据结构：使用一个字典来追踪所有“飞行中”的任务
        #    键是 Future 对象，值是提交该任务所需的完整参数
        future_to_params: Dict[Future, Tuple[Task, str, str, str, int, bool, str, dict, list[bool]]] = {}
        
        tmux = {
            'step': [0 for _ in range(len(tasks) * rollout_n)],
            'token': [0 for _ in range(len(tasks) * rollout_n)],
        }
        stop = [False for _ in range(len(tasks) * rollout_n)]

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            # 2. 初始提交：将所有任务第一次提交到线程池
            for data_id, (task, task_train_exp_mode) in enumerate(zip(tasks, task_train_exp_modes)):
                for rollout_id in range(rollout_n):
                    thread_index = data_id * rollout_n + rollout_id
                    add_exp = add_exp_choices[rollout_id]
                    params = (task, str(data_id), str(rollout_id), mode, thread_index,add_exp,task_train_exp_mode,tmux,stop)
                    future = executor.submit(self.rollout_env_worker, *params)
                    future_to_params[future] = params

            total_rollouts = len(future_to_params)
            pbar = tqdm(total=total_rollouts, desc=f"Epoch {epoch}: Collecting rollouts")

            # 3. 动态处理循环：只要还有任务在执行，就持续循环
            while future_to_params:
                # as_completed 会在任何一个 future 完成时立即返回它
                for future in as_completed(future_to_params):
                    # 获取与这个完成的 future 相关的原始参数,从字典中移除
                    params = future_to_params.pop(future)
                    self.step_status_printer(tmux) # cc: i dont know what this is

                    try:
                        # 4. 健壮的结果获取与错误处理
                        result = future.result()

                        # 处理“软失败”（worker内部捕获错误并返回特殊标记）
                        if 'error' in result.metadata:
                            error_msg = result.metadata['error']
                            logger.warning(f"Task {params[1]}-{params[2]} failed with metadata error: {error_msg}. Retrying... \n Task: {params[0]}")
                            # 由于大部分错误来自于网络和quota，此处强行等待
                            time.sleep(30)
                            # 将任务重新提交,同时重制 tmux 和 stop
                            thread_index=params[4]
                            for k in tmux: tmux[k][thread_index] = 0
                            stop[thread_index]=False
                            new_future = executor.submit(self.rollout_env_worker, *params) # type: ignore
                            future_to_params[new_future] = params
                            continue # 继续处理下一个完成的任务

                        # 5. 成功处理
                        traj_cmt_array.append(result)
                        pbar.update(1) # 只有在真正成功时才更新进度条

                    except Exception as e:
                        # 处理“硬失败”（worker中未捕获的异常）
                        logger.error(f"Task {params[1]}-{params[2]} raised an exception: {e}. Retrying... \n Task: {params[0]}")
                        # 将任务重新提交,同时重制 tmux 和 stop
                        thread_index=params[4]
                        for k in tmux: tmux[k][thread_index] = 0
                        stop[thread_index]=False
                        new_future = executor.submit(self.rollout_env_worker, *params) # type: ignore
                        future_to_params[new_future] = params
            pbar.close()

        task_success_rate = np.mean([cmt.reward.success_rate for cmt in traj_cmt_array])
        for cmt in traj_cmt_array:
            cmt.current_batch_success_rate = np.mean(task_success_rate)
        
        # keep trajectory sorted
        traj_cmt_array = sorted(traj_cmt_array, key=lambda x: (int(x.data_id), int(x.rollout_id)))
        return traj_cmt_array


    #############
    # ANNI 0825
    @staticmethod
    def extract_and_discard_experience(input_string, experience_template):  # <EXP>{}</EXP>
        pattern = re.escape(experience_template).replace(r'\{\}', '(.*?)')
        match = re.search(pattern, input_string)
        if match:
            experience = match.group(1)
            prompt = re.sub(pattern, '', input_string)
            return experience, prompt
        else:
            return "", input_string

    # TODO: define an extra class for trajectory-dataproto converting.
    def to_dataproto(self, cmt_array) -> DataProto:
        """Convert trajectories to DataProto"""
        # Step 1: Convert trajectories to samples: tokenizing
        samples = self.trajectories_to_samples(cmt_array)

        # Step 2: Convert samples to DataProto: padding
        dataproto = self.samples_to_dataproto(samples)

        return dataproto


    def get_extra(self, cmt):
        extras = {
            "add_exp": cmt.metadata.get("add_exp", None),
            "task_train_expmode": cmt.metadata.get("task_train_exp_mode", None),
            "experience": cmt.metadata.get("experience", [])
        }
        return extras


    def trajectories_to_samples(self, cmt_array: List) -> List[Sample]:
        """Convert trajectories to samples"""
        # step 1: convertion
        sample_arr_final = []
        for cmt in cmt_array:
            extras = self.get_extra(cmt)
            # cc: 新 env 返回两条数据会被标记为 initializatio，然后被排除在训练外
            sample_arr = cmt.group_tokenize()
            for sample in sample_arr:
                sample.extras = extras
            sample_arr_final += sample_arr

        # Step 2: Calculate how many samples need to be removed
        world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        remainder = len(sample_arr_final) % world_size
        if remainder != 0:
            import random
            remove_indices = random.sample(range(len(sample_arr_final)), remainder)
            # Sort in reverse order to avoid index shifting during removal
            remove_indices.sort(reverse=True)
            for idx in remove_indices:
                sample_arr_final.pop(idx)

        # random remove some samples, so that the number of samples is divisible by 8
        return sample_arr_final

    def samples_to_dataproto(self, samples: list[Sample]) -> DataProto:
        # Initialize lists to store batched data
        prompt_ids, response_ids = [], []
        prompt_attention_mask, response_attention_mask = [], []
        prompt_position_ids, response_position_ids = [], []
        prompt_loss_mask, response_loss_mask = [], []
        prompt_exp_mask_list, response_exp_mask_list = [], []  # List of binary masks indicating whether to consider off_clip_high for each sample in the batch
        messages = []
        reward_scores = []
        task_ids = []
        rollout_ids = []
        extras = [] # List of dictionaries containing supplementary data for each trajectory, including "add_exp", "task_train_expmode", "experience"

        for sample in samples:
            # Validate that all fields have the same length
            assert len(sample.input_ids) == len(sample.attention_mask) == len(sample.position_ids) == len(
                sample.loss_mask), f"Sample {sample.request_id} has mismatched lengths: " \
                                f"{len(sample.input_ids)=}, {len(sample.attention_mask)=}, " \
                                f"{len(sample.position_ids)=}, {len(sample.loss_mask)=}"

            task_ids.append(sample.task_id)
            rollout_ids.append(sample.rollout_id)
            # Discard samples with prompt length exceeding limit
            if len(sample.prompt_ids) > self.config.data.max_prompt_length:
                raise RuntimeError(f"Sample has prompt_ids length {len(sample.prompt_ids)} ")

            # Warn if response is longer than expected (but still include it)
            if len(sample.response_ids) > self.config.data.max_response_length:
                raise RuntimeError(f"Sample has prompt_ids length {len(sample.prompt_ids)} ")

            # Append tensors to respective lists
            assert len(sample.prompt_ids) != 0
            assert len(sample.response_ids) != 0
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
                prompt_exp_mask_list.append(torch.ones(len(sample.prompt_loss_mask), dtype=torch.int))
                response_exp_mask_list.append(torch.ones(len(sample.response_loss_mask), dtype=torch.int))
            else:
                prompt_exp_mask_list.append(torch.zeros(len(sample.prompt_loss_mask), dtype=torch.int))
                response_exp_mask_list.append(torch.zeros(len(sample.response_loss_mask), dtype=torch.int))



        max_prompt_length_this_batch = max([p.shape[-1] for p in prompt_ids])
        assert max_prompt_length_this_batch <= self.config.data.max_prompt_length
        max_response_length_this_batch = max([p.shape[-1] for p in response_ids])
        assert max_response_length_this_batch <= self.config.data.max_response_length

        # Batch and pad sequences
        prompt_ids =            pad_sequence(prompt_ids, batch_first=True, padding_value=self.pad_token_id, padding_side="left")
        prompt_attention_mask = pad_sequence(prompt_attention_mask, batch_first=True, padding_value=0, padding_side="left")
        prompt_position_ids =   pad_sequence(prompt_position_ids, batch_first=True, padding_value=0, padding_side="left")
        prompt_loss_mask =      pad_sequence(prompt_loss_mask, batch_first=True, padding_value=0, padding_side="left")
        prompt_exp_mask_list =  pad_sequence(prompt_exp_mask_list, batch_first=True, padding_value=0, padding_side="left")

        prompt_ids =            pad_sequence_to_length(prompt_ids, max_prompt_length_this_batch, self.pad_token_id, left_pad=True)
        prompt_attention_mask = pad_sequence_to_length(prompt_attention_mask, max_prompt_length_this_batch, 0, left_pad=True)
        prompt_position_ids =   pad_sequence_to_length(prompt_position_ids, max_prompt_length_this_batch, 0, left_pad=True)
        prompt_loss_mask =      pad_sequence_to_length(prompt_loss_mask, max_prompt_length_this_batch, 0, left_pad=True)
        prompt_exp_mask_list =  pad_sequence_to_length(prompt_exp_mask_list, max_prompt_length_this_batch, 0, left_pad=True)

        response_ids =            pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
        response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
        response_loss_mask =      pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
        response_exp_mask_list =  pad_sequence(response_exp_mask_list, batch_first=True, padding_value=0, padding_side="left")

        response_ids =            pad_sequence_to_length(response_ids, max_response_length_this_batch, self.pad_token_id)
        response_attention_mask = pad_sequence_to_length(response_attention_mask, max_response_length_this_batch, 0)
        response_loss_mask =      pad_sequence_to_length(response_loss_mask, max_response_length_this_batch, 0)
        response_exp_mask_list =  pad_sequence_to_length(response_exp_mask_list, max_prompt_length_this_batch, 0, left_pad=True)

        delta_position_id = torch.arange(1, response_ids.size(1) + 1, device=response_ids.device).unsqueeze(0).repeat(len(samples), 1)
        response_position_ids = prompt_position_ids[:, -1:] + delta_position_id

        # Concatenate prompt and response tensors
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)
        loss_mask = torch.cat((prompt_loss_mask, response_loss_mask), dim=-1)
        exp_mask = torch.cat((prompt_exp_mask_list, response_exp_mask_list), dim=-1)

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

        return DataProto(
            batch=batch,
            non_tensor_batch={
                "task_ids": np.array(task_ids),
                "rollout_ids": np.array(rollout_ids),
                "messages": np.array(messages),
                "reward_scores": np.array(reward_scores),
                "extras": np.array(extras)
            }
        )