import os
import copy
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Literal

import numpy as np
import torch
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
from beyondagent.schema.task import Task
from beyondagent.schema.trajectory import Trajectory, Sample
from context_manager_templates.cmt_linear import CMTLinear
from best_logger import register_logger, print_dict, print_listofdict
try:
    from vsdb import bp
except ImportError:
    def bp(*args, **kwargs): pass  # vsdb is optional for debugging

def init_logger(experiment_name):
    """Initialize the logger with the given configuration."""
    if 'BEST_LOGGER_INIT' in os.environ: return # prevent re-initialization in ray environment
    os.environ['BEST_LOGGER_INIT'] = '1'
    from datetime import datetime
    final_log_path = os.path.join( "./logs", datetime.now().strftime("%Y_%m_%d_%H_%M") + '_' + experiment_name )
    non_console_mods = ["appworld_io", "rollout", "token_clip", "bad_case", "env_clip"]
    register_logger(mods=["evaluation", "exception"], non_console_mods=non_console_mods, auto_clean_mods=[], base_log_path=final_log_path, debug=False)


class ParallelEnvManager(object):
    def __init__(self, config: DictConfig, async_rollout_manager: BaAsyncLLMServerManager, max_parallel: int,
                 max_llm_retries: int = 3, tokenizer=None, **kwargs):
        init_logger(experiment_name=config.trainer.experiment_name)
        self.llm_mode = kwargs.pop("llm_mode", "local")
        super().__init__(**kwargs)

        self.config: DictConfig = config
        self.async_rollout_manager: BaAsyncLLMServerManager = async_rollout_manager
        self.max_parallel: int = max_parallel
        self.max_llm_retries: int = max_llm_retries

        self.rollout_n = config.actor_rollout_ref.rollout.n
        # self.model_name = self.async_rollout_manager.chat_scheduler.model_name
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.rollout_config = config.actor_rollout_ref.rollout
        self.current_token = 0

    def get_llm_chat_fn(self, sampling_params: dict = None) -> callable:
        import asyncio, uuid
        from beyondagent.module.trainer.simple_completion_callback import TokenAndProb
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

            # updated_sampling_params.update({"logprobs": 1, "prompt_logprobs": 1})
            input_messages = copy.deepcopy(messages)
            request_id = uuid.uuid4().hex
            prompt_ids = self.tokenizer.apply_chat_template(input_messages, add_generation_prompt=True, tokenize=True)
            final_res = asyncio.run(self.async_rollout_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=updated_sampling_params,
            ))
            # bp(tag='receivellm', once=True)
            if self.rollout_config.name == 'vllm':
                token_array = final_res.outputs[0].token_ids
            elif self.rollout_config.name == 'sglang':
                token_array = final_res

            decoded_text = self.tokenizer.decode(token_array)
            # decoded_text = "Let's start by finding which API we need to use to interact with Simple Note.\n\nCode:\n```python\nprint(apis.api_docs.show_api_descriptions(app_name='simple_note'))\n```<|im_end|>"
            if decoded_text.endswith('<|im_end|>'):
                decoded_text = decoded_text[:-len('<|im_end|>')]
            # assert prompt_ids == final_res.prompt_token_ids
            # assert final_res.outputs[0].text == decoded_text
            # a = self.tokenizer.apply_chat_template(input_messages + [{"role": "assistant", "content": decoded_text}], add_generation_prompt=False, tokenize=True)
            # b = prompt_ids + token_array
            # assert all([aa==bb for aa,bb in zip(a,b)])
            return {
                "role": "assistant",
                "request_id": request_id,
                "content": decoded_text,
                "tokens": [
                    TokenAndProb(
                        token_id=token,
                        logprob=-1,
                        decoded_string=self.tokenizer.decode(token)
                    )
                    for token in token_array
                ]
            }

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
                    logger.bind(exception=True).exception(f"rollout_server.{i} error: {e.args}")
                    time.sleep(i + 1)

            return output_message[-1]


        if self.llm_mode == "remote":
            return llm_chat_remote
        else:
            return llm_chat


    def step_status_printer(self, tmux):
        # Histogram rollout step counts per tmux bucket (e.g. 0-10, 10-20, …)
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
                           thread_index: int, tmux: list, stop:list=None, **kwargs) -> Trajectory:
        """
        Process a single prompt in a thread-safe way.
        """
        max_retry = 3
        for retry in range(max_retry):
            try:
                # TODO add try exception
                response_length_eps = 6  # reserve tokens for lm_start and other special-token handling
                if self.rollout_config.name == 'vllm':
                    sampling_params = dict(
                        n=1,
                        max_tokens=self.rollout_config.response_length - response_length_eps,
                        temperature=self.rollout_config.temperature,
                        top_p=self.rollout_config.top_p
                    )
                else:
                    sampling_params = dict(
                        n=1,
                        max_new_tokens=self.rollout_config.response_length,
                        temperature=self.rollout_config.temperature,
                        top_p=self.rollout_config.top_p
                    )

                if mode == "validate":
                    sampling_params["temperature"] = self.rollout_config.val_kwargs.temperature
                    sampling_params["top_k"] = self.rollout_config.val_kwargs.top_k
                    sampling_params["top_p"] = self.rollout_config.val_kwargs.top_p

                llm_chat_fn = self.get_llm_chat_fn(sampling_params)
                agent_flow: BaseAgentFlow = AgentFlow(llm_chat_fn=llm_chat_fn, tokenizer=self.tokenizer, config=self.config, **kwargs)

                # FIXME pass env_type & task_id
                env_worker = EnvWorker(env_type=task.env_type, task_id=task.task_id, thread_index=thread_index, config=self.config)
                trajectory: Trajectory = env_worker.execute(data_id=data_id, rollout_id=rollout_id, agent_flow=agent_flow, tmux=tmux, stop=stop)
                break
            except Exception as e:
                if retry < max_retry - 1:
                    logger.bind(exception=True).exception(f"rollout_env_worker error: {e.args}, retrying {retry + 1}/{max_retry}")
                    time.sleep(2 ** retry)
                else:
                    logger.bind(exception=True).exception(f"rollout_env_worker failed after {max_retry} retries: {e.args}")
                    raise e

        return trajectory


    def rollout(self, tasks: List[Task], mode: Literal["sample", "validate"], epoch: str) -> List[CMTLinear]:
        # 1. if enable_oversample
        self.current_token_count_time = time.time()
        if mode=="sample" and (self.rollout_n!=1) and self.config.actor_rollout_ref.rollout.enable_oversample:
            return self.rollout_dynamic(tasks, mode, epoch)
        # 2. otherwise, use legacy rollout method
        cmt_array: List[CMTLinear] = []
        rollout_n = 1 if mode=="validate" else self.rollout_n
        tmux = {
            'step': [0 for _ in range(len(tasks) * rollout_n)],
            'token': [0 for _ in range(len(tasks) * rollout_n)],
        }
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = []
            for data_id, task in enumerate(tasks):
                for rollout_id in range(rollout_n):
                    thread_index = data_id * rollout_n + rollout_id
                    future = executor.submit(self.rollout_env_worker, task=task, data_id=str(data_id), rollout_id=str(rollout_id), mode=mode, thread_index=thread_index, tmux=tmux)
                    futures.append(future)

            while any(future.running() for future in futures):
                self.step_status_printer(tmux)
                time.sleep(10)

            for future in tqdm(futures, desc=f"epoch{epoch}.collect_rollout"):
                # do not fail silently
                result = future.result()
                cmt_array.append(result)


            task_success_rate = np.mean([cmt.reward.success_rate for cmt in cmt_array])
            for cmt in cmt_array:
                cmt.current_batch_success_rate = np.mean(task_success_rate)

            return cmt_array


    def greedy_max_std_selection(self, samples: List[CMTLinear], n):
        # Greedy selection of n samples from m to spread outcome values (high std heuristic).
        # Sort by |outcome| and iteratively pick distinct values.
        if len(samples) < n:
            additional_n = n - len(samples)
            n = len(samples)
        else:
            additional_n = 0

        sorted_samples = sorted(samples, key=lambda cmt: abs(cmt.reward.outcome))
        value_array = [cmt.reward.outcome for cmt in sorted_samples]
        macro_selected_value = []
        macro_selected_index = []
        while len(macro_selected_index) != n:
            selected_value = []
            selected_index = []
            for index, value in enumerate(value_array):
                if (value not in selected_value) and (index not in macro_selected_index):
                    selected_value.append(value)
                    selected_index.append(index)

            if len(selected_value) + len(macro_selected_value) <= n:
                macro_selected_value += selected_value
                macro_selected_index += selected_index

            elif len(selected_value) + len(macro_selected_value) > n:
                preserve_n = n - len(macro_selected_value)
                # Take preserve_n samples from both ends of selected_value / selected_index
                pick_left = preserve_n // 2
                pick_right = preserve_n - pick_left
                macro_selected_value += selected_value[:pick_left] + selected_value[-pick_right:]
                macro_selected_index += selected_index[:pick_left] + selected_index[-pick_right:]
            # print(macro_selected_value, macro_selected_index)

        if additional_n > 0:
            # randomly select `additional_n` samples from `macro_selected_index`, then concat to `macro_selected_index`
            additional_indices = np.random.choice(macro_selected_index, additional_n, replace=True)
            macro_selected_index += additional_indices.tolist()

        selected_samples = [sorted_samples[i] for i in macro_selected_index]
        sorted_selected_samples = sorted(selected_samples, key=lambda cmt: abs(cmt.reward.outcome))
        return sorted_selected_samples


    def rollout_dynamic(self, tasks: List[Task], mode: Literal["sample", "validate"], epoch: str, allow_sample_num_change=True, allow_force_stop=True) -> List[CMTLinear]:
        """
        Rollout more
        """
        cmt_array: List[CMTLinear] = []
        assert mode != "validate"
        rollout_n = self.rollout_n

        submit_oversample_multiplier = self.config.actor_rollout_ref.rollout.submit_oversample_multiplier
        rollout_n_oversample = int(rollout_n * submit_oversample_multiplier)
        rollout_n_confirm = int(rollout_n * (1 + submit_oversample_multiplier) / 2)
        assert rollout_n < rollout_n_confirm < rollout_n_oversample, \
            f"submit_oversample_multiplier is too small, rollout_n={rollout_n}, rollout_n_confirm={rollout_n_confirm}, rollout_n_oversample={rollout_n_oversample}"

        tmux = {
            'step': [0 for _ in range(len(tasks) * rollout_n_oversample)],
            'token': [0 for _ in range(len(tasks) * rollout_n_oversample)],
        }
        stop = [False for _ in range(len(tasks) * rollout_n_oversample)]

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            # submit worker threads
            futures = []
            for data_id, task in enumerate(tasks):
                task_future_array = []
                for rollout_id in range(rollout_n_oversample):
                    thread_index = data_id * rollout_n_oversample + rollout_id
                    future = executor.submit(self.rollout_env_worker, task=task, data_id=str(data_id), rollout_id=str(rollout_id),
                                             mode=mode, thread_index=thread_index, tmux=tmux, stop=stop)
                    task_future_array.append(future)
                futures += [task_future_array]

            tic = -1
            # poll completed futures and termination conditions
            while True:
                tic += 1
                can_terminate = [False for _ in futures]
                terminate_status = ['running' for _ in futures]
                for j, task_future_array in enumerate(futures):
                    completed_task_futures = [f for f in task_future_array if f.done()]
                    completed_results = [f.result() for f in completed_task_futures]
                    completed_results = [cmt for cmt in completed_results if not cmt.discarded]
                    reward = [cmt.reward.outcome for cmt in completed_results]
                    reward_std = np.std(reward) if reward else 0.0
                    all_finished = (len(completed_task_futures) == len(task_future_array))
                    # finish condition 1: all oversample tasks are finished
                    if all_finished:
                        can_terminate[j] = True
                        terminate_status[j] = f'all_fin({len(completed_results)}/{reward_std:.2f})'
                    num_finished = len(completed_task_futures)
                    task_cmd_reward_array = [cmt.reward.outcome for cmt in completed_results]
                    all_equal = all(x == task_cmd_reward_array[0] for x in task_cmd_reward_array)
                    # all_reward_greater_than_one = all(x >= 1 for x in task_cmd_reward_array)
                    if not all_equal:
                        if (num_finished >= rollout_n):
                            # finish condition 2: more than rollout_n tasks are finished, and, reward are not all equal
                            can_terminate[j] = True
                            terminate_status[j] = f'early_end({len(completed_results)}/{reward_std:.2f})'
                        else:
                            pass # keep waiting
                    else:
                        if num_finished >= rollout_n_confirm:
                            # finish condition 3: if more than rollout_n_confirm tasks are finished, we can confirm this task is hopeless (or successful for certainty)
                            can_terminate[j] = True
                            terminate_status[j] = f'confirm_dummy({len(completed_results)}/{reward_std:.2f})'
                            # take actions to stop future rollout
                            if allow_force_stop:
                                for k in range(j*rollout_n_oversample, j*rollout_n_oversample + rollout_n_oversample):
                                    stop[k] = True
                        else:
                            pass # keep waiting
                # check global status
                terminate_status = '/'.join(terminate_status)
                if all(can_terminate):
                    logger.info(f"epoch{epoch}.collect_rollout: all tasks finished, exiting loop")
                    for i, stop_flag in enumerate(stop): stop[i] = True # all must stop now
                    break
                else:
                    if tic % 10 == 0:
                        self.step_status_printer(tmux) # print status every 10*5=50 seconds
                        # logger.info(stop)
                        logger.info(f"task complete {sum(can_terminate)}/{len(can_terminate)} tasks: {terminate_status}")
                    time.sleep(5)
            # wait for all threads to finish or be stopped
            tic = -1
            while any(f.running() for task_future_array in futures for f in task_future_array):
                tic += 1
                if tic % 10 == 0: logger.info('waiting final sync, this will not take long')
                time.sleep(5)

            # count how many threads finished useful work per task
            task_ineffective_thread_cnt = []
            task_completed_thread_cnt = []
            task_extra_thread_cnt = []
            task_need_amend = 0
            for j, task_future_array in enumerate(futures):
                # get number of completed tasks
                completed_task_futures = [f for f in task_future_array if f.done()]
                completed_results = [f.result() for f in completed_task_futures]
                completed_results = [cmt for cmt in completed_results if not cmt.discarded]
                task_cmd_reward_array = [cmt.reward.outcome for cmt in completed_results]
                all_equal = all(x == task_cmd_reward_array[0] for x in task_cmd_reward_array)
                # tally completed vs dummy runs
                completed_task_cnt = len(completed_results)
                if all_equal:
                    task_need_amend += 1
                    task_completed_thread_cnt += [0]
                    task_extra_thread_cnt += [0]
                    task_ineffective_thread_cnt += [completed_task_cnt]
                else:
                    task_need_amend += 0
                    task_completed_thread_cnt += [completed_task_cnt]
                    task_extra_thread_cnt += [completed_task_cnt - rollout_n]
                    task_ineffective_thread_cnt += [0]

            logger.info(f"task_completed_thread_cnt: {task_completed_thread_cnt}")
            logger.info(f"task_extra_thread_cnt: {task_extra_thread_cnt}")

            world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
            total_sample = sum(task_completed_thread_cnt)
            if allow_sample_num_change and (total_sample > world_size*2):
                # Allow varying sample count; only require divisibility by world_size when padding
                # add_count = (world_size - total_sample % world_size)   # samples to add if using pad policy
                add_count = 0  # samples to add under pad policy (currently disabled)
                num_task_to_amend = len(futures)    # num_task
                logger.info(f"allow_sample_num_change policy: world_size: {world_size}, total_sample {total_sample}, add_count: {add_count}, ")
                # pad from tasks with the smallest completed count first
                while add_count != 0:
                    _task_completed_thread_cnt_find_nonzero_min = [float('inf') if x <=0 else x for x in task_completed_thread_cnt]
                    min_extra_index = _task_completed_thread_cnt_find_nonzero_min.index(min(_task_completed_thread_cnt_find_nonzero_min))
                    task_extra_thread_cnt[min_extra_index] += 1
                    task_completed_thread_cnt[min_extra_index] += 1
                    add_count -= 1
                # logger.info(f"_task_completed_thread_cnt_find_nonzero_min: {_task_completed_thread_cnt_find_nonzero_min}")
                logger.info(f"task_completed_thread_cnt (after remove): {task_completed_thread_cnt}")
                logger.info(f"task_extra_thread_cnt (after remove): {task_extra_thread_cnt}")
            else:
                # Fixed sample count: trim extras to match rollout_n budget
                num_task_max_to_amend = sum(task_extra_thread_cnt) // rollout_n
                num_task_to_amend = min(num_task_max_to_amend, task_need_amend)
                extra_num_thread_required = num_task_to_amend * rollout_n
                remove_count = sum(task_extra_thread_cnt) - extra_num_thread_required
                logger.info(f"forbid_sample_num_change policy: num_task_max_to_amend: {num_task_max_to_amend}, num_task_to_amend: {num_task_to_amend}, remove_count: {remove_count}, ")

                # drop extras from tasks with the largest surplus first
                while remove_count != 0:
                    max_extra_index = task_extra_thread_cnt.index(max(task_extra_thread_cnt))
                    assert task_extra_thread_cnt[max_extra_index] > 0, "task_extra_thread_cnt should be greater than 0"
                    task_extra_thread_cnt[max_extra_index] -= 1
                    task_completed_thread_cnt[max_extra_index] -= 1
                    remove_count -= 1
                logger.info(f"task_completed_thread_cnt (after remove): {task_completed_thread_cnt}")
                logger.info(f"task_extra_thread_cnt (after remove): {task_extra_thread_cnt}")

            # keep highest-variance subset per task via greedy_max_std_selection
            cmt_array = []
            print_buffer = ""
            task_success_rate = []
            for j, task_future_array, avail_extra_cnt in zip(range(len(futures)), futures, task_extra_thread_cnt):
                # get number of completed tasks
                completed_task_futures = [f for f in task_future_array if f.done()]
                completed_results = [f.result() for f in completed_task_futures]
                completed_results: List[CMTLinear] = [cmt for cmt in completed_results if not cmt.discarded]
                task_cmd_reward_array = [cmt.reward.outcome for cmt in completed_results]
                success_rate_array = [cmt.reward.success_rate for cmt in completed_results]
                task_success_rate += [np.mean(success_rate_array)]
                need_amend = all(x == task_cmd_reward_array[0] for x in task_cmd_reward_array)
                if need_amend and (num_task_to_amend > 0):
                    num_task_to_amend -= 1
                    print_buffer += f"/(amend)"
                    continue
                else:
                    if need_amend:
                        num_completed = len(completed_results)
                        num_to_be_selected = rollout_n
                    else:
                        num_completed = len(completed_results)
                        num_to_be_selected = rollout_n + avail_extra_cnt
                    # assert num_completed >= num_to_be_selected, f"num_completed={num_completed}, num_to_be_selected={num_to_be_selected}"
                    selected_cmt_array = self.greedy_max_std_selection(completed_results, num_to_be_selected)
                    cmt_array += selected_cmt_array
                    print_buffer += f"/({len(selected_cmt_array)})"
                    if need_amend: print_buffer += "(no-amend)"
            logger.info(print_buffer)

            for cmt in cmt_array:
                cmt.current_batch_success_rate = np.mean(task_success_rate)
            return cmt_array



    # TODO: define an extra class for trajectory-dataproto converting.
    def to_dataproto(self, cmt_array) -> DataProto:
        """Convert trajectories to DataProto"""
        # Step 1: Convert trajectories to samples: tokenizing
        samples = self.trajectories_to_samples(cmt_array)

        # Step 2: Convert samples to DataProto: padding
        dataproto = self.samples_to_dataproto(samples)

        return dataproto
    # TODO 
    def to_dataproto_list(self, cmt_array) -> List[DataProto]:
        """Convert trajectories to DataProto"""
        # Step 1: Convert trajectories to samples: tokenizing
        samples_list = self.trajectories_to_samples_list(cmt_array)

        # Step 2: Convert samples to DataProto: padding
        dataproto_list = [self.samples_to_dataproto(samples) for samples in samples_list]

        return dataproto_list

    def trajectories_to_samples(self, cmt_array: List[CMTLinear]) -> List[Sample]:
        """Convert trajectories to samples"""
        sample_arr_final = []
        for cmt in cmt_array:
            sample_arr = cmt.group_tokenize()
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

    def trajectories_to_samples_list(self, cmt_array: List[CMTLinear]):
        """Convert trajectories to samples"""
        max_steps = self.rollout_config.multi_turn.max_steps
        sample_arr_final = [[]for i in range(max_steps)]
        for cmt in cmt_array:
            sample_arr = cmt.group_tokenize_turns()
            for i in range(max_steps):
                sample_arr_final[i] += [sample_arr[i]]

        world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        for sample_turn in sample_arr_final:
            assert len(sample_turn) % world_size == 0, "sample_turn length{} must be divisible by world size{}.".format(len(sample_turn), world_size)


        return sample_arr_final

    def samples_to_dataproto(self, samples: list[Sample]) -> DataProto:
        # Initialize lists to store batched data
        prompt_ids, response_ids = [], []
        prompt_attention_mask, response_attention_mask = [], []
        prompt_position_ids, response_position_ids = [], []
        prompt_loss_mask, response_loss_mask = [], []
        messages = []
        reward_scores = []
        task_ids = []

        for sample in samples:
            # Validate that all fields have the same length
            assert len(sample.input_ids) == len(sample.attention_mask) == len(sample.position_ids) == len(
                sample.loss_mask), f"Sample {sample.request_id} has mismatched lengths: " \
                                f"{len(sample.input_ids)=}, {len(sample.attention_mask)=}, " \
                                f"{len(sample.position_ids)=}, {len(sample.loss_mask)=}"

            task_ids.append(sample.task_id)
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

        max_prompt_length_this_batch = max([p.shape[-1] for p in prompt_ids])
        assert max_prompt_length_this_batch <= self.config.data.max_prompt_length
        max_response_length_this_batch = max([p.shape[-1] for p in response_ids])
        assert max_response_length_this_batch <= self.config.data.max_response_length

        # Batch and pad sequences
        prompt_ids =            pad_sequence(prompt_ids, batch_first=True, padding_value=self.pad_token_id, padding_side="left")
        prompt_attention_mask = pad_sequence(prompt_attention_mask, batch_first=True, padding_value=0, padding_side="left")
        prompt_position_ids =   pad_sequence(prompt_position_ids, batch_first=True, padding_value=0, padding_side="left")
        prompt_loss_mask =      pad_sequence(prompt_loss_mask, batch_first=True, padding_value=0, padding_side="left")

        prompt_ids =            pad_sequence_to_length(prompt_ids, max_prompt_length_this_batch, self.pad_token_id, left_pad=True)
        prompt_attention_mask = pad_sequence_to_length(prompt_attention_mask, max_prompt_length_this_batch, 0, left_pad=True)
        prompt_position_ids =   pad_sequence_to_length(prompt_position_ids, max_prompt_length_this_batch, 0, left_pad=True)
        prompt_loss_mask =      pad_sequence_to_length(prompt_loss_mask, max_prompt_length_this_batch, 0, left_pad=True)

        response_ids =            pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
        response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
        response_loss_mask =      pad_sequence(response_loss_mask, batch_first=True, padding_value=0)

        response_ids =            pad_sequence_to_length(response_ids, max_response_length_this_batch, self.pad_token_id)
        response_attention_mask = pad_sequence_to_length(response_attention_mask, max_response_length_this_batch, 0)
        response_loss_mask =      pad_sequence_to_length(response_loss_mask, max_response_length_this_batch, 0)

        delta_position_id = torch.arange(1, response_ids.size(1) + 1, device=response_ids.device).unsqueeze(0).repeat(len(samples), 1)
        response_position_ids = prompt_position_ids[:, -1:] + delta_position_id

        # Concatenate prompt and response tensors
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)
        loss_mask = torch.cat((prompt_loss_mask, response_loss_mask), dim=-1)

        # Construct the batch using TensorDict
        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            },
            batch_size=len(samples),
        )

        return DataProto(batch=batch, non_tensor_batch={"task_ids": np.array(task_ids), "messages": np.array(messages), "reward_scores": np.array(reward_scores)})