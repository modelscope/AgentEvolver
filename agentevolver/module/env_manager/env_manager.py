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

from agentevolver.module.agent_flow.agent_flow import AgentFlow
from agentevolver.module.agent_flow.base_agent_flow import BaseAgentFlow
from agentevolver.module.env_manager.env_worker import EnvWorker
from agentevolver.module.trainer.ae_async_llm_server_manager import BaAsyncLLMServerManager
from agentevolver.module.task_manager.rewards import grader_manager
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory, Sample
from agentevolver.utils.step_parser import parse_response_ids_to_steps
# do not delete this line
from agentevolver.module.task_manager.rewards import LlmAsJudgeRewardCalculator,LlmAsJudgeRewardCalculatorWithGT,LlmAsJudgeBinaryRewardCalculator,LlmAsJudgeBinaryRewardCalculatorWithGT,EnvGrader, AvgBinaryGTJudge, AvgLlmJudge
from beast_logger import register_logger
from agentevolver.module.exp_manager.exp_manager import TaskExpConfig, TrajExpConfig


def init_logger(experiment_name):
    """
    Initializes the logger with the given experiment name and sets up the logging environment.

    Args:
        experiment_name (str): The name of the experiment for which the logger is being initialized.
    """
    if 'BEST_LOGGER_INIT' in os.environ: return  # Prevents re-initialization in ray environment
    os.environ['BEST_LOGGER_INIT'] = '1'
    os.environ['BEST_LOGGER_WEB_SERVICE_URL'] = "http://127.0.0.1:8181/"
    from datetime import datetime
    final_log_path = os.path.join( "experiments", experiment_name, "trace_rollout", datetime.now().strftime("%Y_%m_%d_%H_%M"))
    non_console_mods = ["conversation", "rollout", "token_clip", "bad_case", "env_clip"]
    register_logger(mods=["evaluation", "exception"], non_console_mods=non_console_mods, auto_clean_mods=[], base_log_path=final_log_path, debug=False)
    print('Run `beast_logger_go` and click the url to inspect rollout logs. Continue in 5 seconds')
    time.sleep(2.5)



class ParallelEnvManager(object):
    """
    Manages a parallel environment for running multiple tasks, handling retries, logging, and using a language model to generate responses, ultimately returning the trajectories of the tasks.
    """
    def __init__(self, config: DictConfig, async_rollout_manager: BaAsyncLLMServerManager, max_parallel: int,
                 max_llm_retries: int = 3, **kwargs):
        """
        Initializes the ParallelEnvManager with the provided configuration and settings.

        Args:
            config (DictConfig): Configuration dictionary containing all necessary parameters.
            async_rollout_manager (BaAsyncLLMServerManager): Manager for asynchronous rollouts.
            max_parallel (int): Maximum number of parallel tasks that can be run.
            max_llm_retries (int, optional): Maximum number of retries for the language model. Defaults to 3.
            **kwargs: Additional keyword arguments.
        """
        init_logger(experiment_name=config.trainer.experiment_name)  # ⭐ Initializes the logger
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

        # self.experience_template = config.hybrid_experience_training.experience_template
        self.llm_mode = "local" # use fsdp worker ("local") or use foreign server ("remote")
        self.current_token = 0
        self.current_token_count_time = time.time()


    def get_llm_chat_fn(self, sampling_params: dict = None) -> callable:
        """
        Returns a callable function for chatting with a language model. The returned function is either
        `llm_chat` or `llm_chat_remote` based on the `llm_mode` attribute.

        Args:
            sampling_params (dict, optional): Default sampling parameters for the chat function.

        Returns:
            callable: A function to chat with the language model.
        """

        def llm_chat(messages: List[Dict[str, str]],
                     custom_sampling_params: dict = None,
                     request_id: str = None) -> dict:
            """
            Sends messages to the language model and returns the assistant's response. This function handles
            retries and updates the sampling parameters.

            Args:
                messages (List[Dict[str, str]]): A list of message dictionaries with "role" and "value" keys.
                custom_sampling_params (dict, optional): Custom sampling parameters to override the default ones.
                request_id (str, optional): A unique identifier for the request.

            Returns:
                dict: The last message in the input messages, which is expected to be the assistant's response.
            """
            # TODO: sending sampling_params to rollout server
            updated_sampling_params = {}
            if sampling_params:
                updated_sampling_params.update(sampling_params)
            if custom_sampling_params:
                updated_sampling_params.update(custom_sampling_params)
            updated_sampling_params.update({"logprobs": 1, "return_tokens_as_token_ids": True})  # ⭐ Update sampling parameters

            input_messages = copy.deepcopy(messages)
            weighted_addresses = self.async_rollout_manager.chat_scheduler.weighted_addresses
            # logger.info(f"weighted_addresses={weighted_addresses}")
            for i in range(self.max_llm_retries):
                try:
                    self.async_rollout_manager.submit_chat_completions(messages=input_messages,
                                                                       sampling_params=updated_sampling_params,
                                                                       request_id=request_id)  # ⭐ Submit chat completions
                    break

                except Exception as e:
                    logger.exception(f"rollout_server.{i} error: {e.args}")
                    time.sleep(i + 1)

            return input_messages[-1]

        def llm_chat_remote(messages: List[Dict[str, str]],
                           custom_sampling_params: dict = None,
                           request_id: str = None) -> dict:
            """
            Sends messages to the remote language model and returns the assistant's response. This function
            handles retries and updates the sampling parameters.

            Args:
                messages (List[Dict[str, str]]): A list of message dictionaries with "role" and "value" keys.
                custom_sampling_params (dict, optional): Custom sampling parameters to override the default ones.
                request_id (str, optional): A unique identifier for the request.

            Returns:
                dict: The last message in the output messages, which is expected to be the assistant's response.
            """
            updated_sampling_params = {}
            if sampling_params:
                updated_sampling_params.update(sampling_params)
            if custom_sampling_params:
                updated_sampling_params.update(custom_sampling_params)
            updated_sampling_params.update({"logprobs": 1, "return_tokens_as_token_ids": True})  # ⭐ Update sampling parameters
            input_messages = copy.deepcopy(messages)
            for i in range(self.max_llm_retries):
                try:
                    output_message = self.async_rollout_manager.submit_chat_completions(messages=input_messages,
                                                                                         sampling_params=updated_sampling_params,
                                                                                         request_id=request_id)  # ⭐ Submit chat completions
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
        """
        Prints the current status of the steps in the parallel environment, including the number of threads in different step ranges and the token generation rate.

        Args:
            tmux (dict): A dictionary containing the 'step' and 'token' information for the current state of the environment.
        """
        # Initialize a counter to keep track of the number of threads in each step range
        step_counter = {}

        # Calculate the total tokens and the time elapsed since the last count
        current_token = sum(tmux['token'])
        current_time = time.time()
        delta_token = current_token - self.current_token
        delta_time = current_time - self.current_token_count_time
        self.current_token = current_token
        self.current_token_count_time = current_time
        token_gen_per_sec_str = f"{delta_token/delta_time:.2f} tokens/s" if delta_time > 0 else "N/A"

        # Categorize the steps into bins and count the number of threads in each bin
        for step in tmux['step']:
            if step == -1:
                step_counter[(-1, 'terminated')] = step_counter.get((-1, 'terminated'), 0) + 1
                continue
            else:
                start = (step // 5) * 5
                end = start + 5
                step_counter[(start, end)] = step_counter.get((start, end), 0) + 1

        # Sort the step counter by the start value of each bin
        step_counter = dict(sorted(step_counter.items(), key=lambda x: x[0][0]))  # ⭐ Sort the step counter to ensure the output is in ascending order

        # Prepare the print buffer with the formatted step ranges and thread counts
        print_buf = []
        for (start, end), count in step_counter.items():
            if start != -1:
                print_buf += [f"[{start}-{end}]:{count} threads"]
        for (start, end), count in step_counter.items():
            if start == -1:
                print_buf += [f"[finished]:{count} threads"]

        # Print the rollout progress with the token generation rate and the step ranges
        print(f"Rollout progress ({token_gen_per_sec_str}): " + "  //  ".join(print_buf))



    def rollout_env_worker(self, task: Task, traj_exp_config: TrajExpConfig, data_id: str, rollout_id: str, mode: Literal["sample", "validate"],
                           thread_index: int, tmux: dict, stop:list, **kwargs) -> Trajectory:
        """
        Processes a single task in a thread-safe way, handling retries and exceptions.

        Args:
            task (Task): The task to be processed.
            traj_exp_config (TrajExpConfig): Experience Configuration for the trajectory.
            data_id (str): The ID of the data.
            rollout_id (str): The ID of the rollout.
            mode (Literal["sample", "validate"]): The mode of operation, either 'sample' or 'validate'.
            thread_index (int): The index of the thread.
            tmux (dict): TMUX configuration.
            stop (list): List of stop conditions.
            **kwargs: Additional keyword arguments.

        Returns:
            Trajectory: The trajectory generated from the task execution.
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
                trajectory: Trajectory = env_worker.execute(data_id=data_id, rollout_id=rollout_id, traj_exp_config=traj_exp_config, agent_flow=agent_flow, tmux=tmux, stop=stop) # ⭐ Execute the task and generate the trajectory
                return trajectory

            except Exception as e:
                if retry < max_retry - 1:
                    logger.bind(exception=True).exception(f"rollout_env_worker error: {e.args}, retrying {retry + 1}/{max_retry}")
                    time.sleep(2 ** retry)
                else:
                    logger.bind(exception=True).exception(f"rollout_env_worker failed after {max_retry} retries: {e.args}")
                    raise e

    def rollout(self, tasks: List[Task], task_exp_configs: List[TaskExpConfig], mode: Literal["sample", "validate"], epoch: str) -> List[Trajectory]:
        """
        Executes a list of tasks in a parallel environment using a thread pool, with automatic retries for failed tasks.

        Args:
            tasks (List[Task]): A list of tasks to be processed.
            task_exp_configs (List[TaskExpConfig]): A list of experience configurations corresponding to each task.
            mode (Literal["sample", "validate"]): The mode of operation, either 'sample' or 'validate'.
            epoch (str): The current epoch identifier, used for logging and progress bar.

        Returns:
            List[Trajectory]: A sorted list of Trajectory objects representing the results of the successfully completed tasks.
        """
        traj_cmt_array = []
        rollout_n = self.rollout_config.val_kwargs.n if mode == "validate" else self.rollout_n
        future_to_params: Dict[Future, Tuple[Task, TrajExpConfig, str, str, str, int, dict, list[bool]]] = {}

        tmux = {
            'step': [0 for _ in range(len(tasks) * rollout_n)],
            'token': [0 for _ in range(len(tasks) * rollout_n)],
        }
        stop = [False for _ in range(len(tasks) * rollout_n)]

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            # 2. submit: submit all tasks to the thread pool
            for data_id, (task, task_exp_config) in enumerate(zip(tasks, task_exp_configs)):
                for rollout_id in range(rollout_n):
                    thread_index = data_id * rollout_n + rollout_id
                    add_exp = task_exp_config.add_exp[rollout_id]
                    train_mode = task_exp_config.train_mode
                    traj_exp_config = TrajExpConfig(
                        add_exp=add_exp, train_mode=train_mode, task_id=task.task_id, data_id=data_id, rollout_id=rollout_id, mode=mode)

                    params = (task, traj_exp_config, str(data_id), str(rollout_id), mode, thread_index, tmux,stop)
                    future = executor.submit(self.rollout_env_worker, *params)
                    future_to_params[future] = params

            total_rollouts = len(future_to_params)
            pbar = tqdm(total=total_rollouts, desc=f"Epoch {epoch}: Collecting rollouts")

            # 3. wait for all tasks to complete
            while future_to_params:
                # if any future is done, process it
                for future in as_completed(future_to_params):
                    # get the corresponding params, and remove it from the dict
                    params = future_to_params.pop(future)
                    self.step_status_printer(tmux) # cc: i dont know what this is

                    # 4. get the results and handle errors
                    try:
                        result = future.result()  # ⭐ Retrieve the result from the completed future

                        # if the result has metadata error, try to recover
                        if 'error' in result.metadata:
                            error_msg = result.metadata['error']
                            logger.warning(f"Task {params[1]}-{params[2]} failed with metadata error: {error_msg}. Retrying... \n Task: {params[0]}")
                            # as most errors are internet error or quota, we wait before resubmit it
                            time.sleep(30)
                            # resubmit and reset tmux and stop
                            thread_index=params[5]
                            for k in tmux: tmux[k][thread_index] = 0
                            stop[thread_index]=False
                            new_future = executor.submit(self.rollout_env_worker, *params) # type: ignore
                            future_to_params[new_future] = params
                            continue

                        # 5. if the task is successful, add it to the result list
                        traj_cmt_array.append(result)
                        pbar.update(1) # update progress bar when success

                    except Exception as e:
                        # handle the uncaught exception
                        logger.error(f"Task {params[1]}-{params[2]} raised an exception: {e}. Retrying... \n Task: {params[0]}")
                        # resubmit, and reset tmux and stop
                        thread_index=params[5]
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


    # TODO: define an extra class for trajectory-dataproto converting.
    def to_dataproto(self, cmt_array) -> DataProto:
        """
        Converts a list of trajectories into a DataProto object.

        Args:
            cmt_array (list): A list of trajectories that need to be converted.

        Returns:
            DataProto: The resulting DataProto object after conversion.
        """
        # Step 1: Convert trajectories to samples: tokenizing
        samples = self.trajectories_to_samples(cmt_array)  # ⭐ Tokenize the trajectories to create samples

        # Step 2: Convert samples to DataProto: padding
        dataproto = self.samples_to_dataproto(samples)  # ⭐ Pad the samples and convert them to DataProto

        return dataproto


    def get_extra(self, cmt):
        """
        Extracts and returns extra information from the comment's metadata.

        Args:
            cmt (object): The comment object containing metadata.

        Returns:
            dict: A dictionary with keys 'add_exp', 'task_train_expmode', and 'experience_list' corresponding to their respective values in the comment's metadata.
        """
        extras = {
            "add_exp": cmt.metadata.get("add_exp", None),  # ⭐ Retrieves the 'add_exp' value from metadata
            "task_train_expmode": cmt.metadata.get("task_train_exp_mode", None),  # ⭐ Retrieves the 'task_train_exp_mode' value from metadata
            "experience_list": cmt.metadata.get("experience_list", [])  # ⭐ Retrieves the 'experience' list from metadata
        }
        return extras


    def trajectories_to_samples(self, cmt_array: List) -> List[Sample]:
        """
        Converts a list of trajectories into a list of samples, ensuring the number of samples is divisible by the total number of GPUs across all nodes.

        Args:
            cmt_array (List): A list of trajectories to be converted into samples.

        Returns:
            List[Sample]: A list of samples with extras added and adjusted to be divisible by the world size.
        """
        # Step 1: Conversion
        sample_arr_final = []
        for cmt in cmt_array:
            extras = self.get_extra(cmt)
            # cc: message returned by the new env will be tagged as initialization, with no loss-mask
            sample_arr = cmt.group_tokenize()  # ⭐ Tokenize the trajectory into samples
            for sample in sample_arr:
                sample.extras = extras  # ⭐ Add extra information to each sample
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
                sample_arr_final.pop(idx)  # ⭐ Remove samples to make the total number divisible by world size

        # Randomly remove some samples, so that the number of samples is divisible by 8
        return sample_arr_final

    def samples_to_dataproto(self, samples: list[Sample]) -> DataProto:
        """
        Converts a list of Sample objects into a DataProto object, batching and padding the data.

        Args:
            samples (list[Sample]): A list of Sample objects to be converted.

        Returns:
            DataProto: A DataProto object containing the batched and padded data.
        """
        # Initialize lists to store batched data
        step_ids_list  = []
        steps_texts_list = []
        prompt_ids, response_ids = [], []
        prompt_attention_mask, response_attention_mask = [], []
        prompt_position_ids, response_position_ids = [], []
        prompt_loss_mask, response_loss_mask = [], []
        prompt_exp_mask_list, response_exp_mask_list = [], []  # List of binary masks indicating whether to consider off_clip_high for each sample in the batch
        messages = []
        reward_scores = []
        task_ids = []
        rollout_ids = []
        extras = [] # List of dictionaries containing supplementary data for each trajectory, including "add_exp", "task_train_expmode", "experience_list"
        k_text_list = []
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
                logger.warning(
                    f"Sample {sample.request_id} has response_ids length {len(sample.response_ids)} "
                    f"greater than max_response_length {self.config.data.max_response_length}."
                )
                raise RuntimeError(f"Sample has prompt_ids length {len(sample.prompt_ids)} ")

            # ------------- shuchang 0714: append step_ids and steps_texts ------------
            resp_ids = sample.response_ids
            # shuchang: 0809
            # FIXME: Solve the issue of misaligned step IDs, use the unified step parsing function parse_response_ids_to_steps
            resp_ids = sample.response_ids
            parse_result = parse_response_ids_to_steps(resp_ids, self.tokenizer) # ⭐ Parse the response IDs into step IDs and texts

            step_ids_list.append(torch.tensor(parse_result.step_ids, dtype=torch.long))
            # generate steps_texts (for semantic evaluation)
            steps_texts_list.append([
                {"action": s["action_text"], "observation": s["observation_text"]}
                for s in parse_result.steps
            ])

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
        # ------------- shuchang 0714: pad step_ids and steps_texts ------------
        step_ids_pad = pad_sequence(
            step_ids_list, batch_first=True, padding_value=-1
        )

        step_ids_pad = pad_sequence_to_length(
            step_ids_pad, self.config.data.max_response_length, -1
        )  # ⭐ Pad step IDs to the maximum response length
        # ------------- shuchang 0714: pad step_ids and steps_texts ------------


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
        # response_exp_mask_list =  pad_sequence(response_exp_mask_list, batch_first=True, padding_value=0, padding_side="left")
        response_exp_mask_list  = pad_sequence(response_exp_mask_list,  batch_first=True, padding_value=0)        # shuchang debug: Remove padding_side="left"


        response_ids =            pad_sequence_to_length(response_ids, max_response_length_this_batch, self.pad_token_id)  # ⭐ Pad response IDs to the maximum response length
        response_attention_mask = pad_sequence_to_length(response_attention_mask, max_response_length_this_batch, 0)  # ⭐ Pad response attention mask to the maximum response length
        response_loss_mask =      pad_sequence_to_length(response_loss_mask, max_response_length_this_batch, 0)  # ⭐ Pad response loss mask to the maximum response length
        # response_exp_mask_list =  pad_sequence_to_length(response_exp_mask_list, max_prompt_length_this_batch, 0, left_pad=True)  # ⭐ Pad response experience mask list to the maximum prompt length
        response_exp_mask_list =  pad_sequence_to_length(response_exp_mask_list, max_response_length_this_batch, 0) # shuchang debug: Should pad to max_response_length_this_batch, not the previous max_prompt_length_this_batch

        delta_position_id = torch.arange(1, response_ids.size(1) + 1, device=response_ids.device).unsqueeze(0).repeat(len(samples), 1)
        response_position_ids = prompt_position_ids[:, -1:] + delta_position_id  # ⭐ Calculate the position IDs for the response

        # Concatenate prompt and response tensors
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)  # ⭐ Concatenate prompt and response IDs
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)  # ⭐ Concatenate prompt and response attention masks
        position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)  # ⭐ Concatenate prompt and response position IDs
        loss_mask = torch.cat((prompt_loss_mask, response_loss_mask), dim=-1)  # ⭐ Concatenate prompt and response loss masks
        # shuchang: construct group_id
        group_ids = torch.tensor([int(s.data_id) for s in samples], dtype=torch.long)  # ⭐ Construct group IDs from sample data
        # Validate masks have same shape
        exp_mask = torch.cat((prompt_exp_mask_list, response_exp_mask_list), dim=-1)  # ⭐ Concatenate prompt and response experience masks

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
                "exp_mask": exp_mask,        # add exp_mask by ANNI
                "step_ids": step_ids_pad,
                "group_ids": group_ids,   # ★ add groupid
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
                "extras": np.array(extras),
                "steps": np.array(steps_texts_list, dtype=object)
            }
        )
