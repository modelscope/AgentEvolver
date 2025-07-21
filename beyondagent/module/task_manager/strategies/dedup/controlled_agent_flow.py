import json
import time
from typing import Optional, cast

from loguru import logger

from beyondagent.client.em_client import EMClient
from beyondagent.client.env_client import EnvClient
from beyondagent.module.agent_flow.base_agent_flow import BaseAgentFlow
from beyondagent.module.agent_flow.reward_calculator import RewardCalculator
from beyondagent.module.task_manager.strategies.dedup.embedding import StateRecorder
from beyondagent.schema.trajectory import Trajectory
from beyondagent.utils.utils import convert_tool_to_user_message, clip_state_content_correctly


class ControlledAgentFlow(BaseAgentFlow):

    def __init__(self,state_recorder:StateRecorder,reward_calculator:Optional[RewardCalculator]=None, **kwargs):
        super().__init__(**kwargs)
        self._state_recorder=state_recorder
        # 优先传入的参数
        self._reward_calculator = reward_calculator
        if self._reward_calculator is not None:
            logger.info(f"reward_calculator={self._reward_calculator}")
        self._enable_context_generator=False

        self.instruction_template_ids = self.tokenizer.encode("<|im_start|>user\n")
        self.response_template_ids = self.tokenizer.encode("<|im_start|>assistant\n")
        self.em_client = EMClient(base_url=self.config.experience_maker.base_url)
        

    def execute(self, trajectory: Trajectory, env: EnvClient, instance_id: str, **kwargs) -> Trajectory:
        request_id: str = ""
        for act_step in range(self.max_steps):
            # if use qwen3, add /no_think
            if self.config.actor_rollout_ref.rollout.use_qwen3:
                trajectory.steps[-1]["content"] = " /no_think     "+trajectory.steps[-1]["content"]

            prompt_text = self.tokenizer.apply_chat_template(trajectory.steps, 
                                                             tokenize=False,
                                                             add_generation_prompt=True)
            current_token_len = len(self.tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"][0])

            # yunpeng 0623: to prevent add an imend token to an uncompleted seq, 
            # because the message-type output will be applied chat_template.
            max_response_length = self.config.actor_rollout_ref.rollout.response_length
            if current_token_len + max_response_length > self.max_model_len:
                logger.warning(f"exceed max model len={self.max_model_len}")
                break

            enable_request_id = False
            if hasattr(self.config.actor_rollout_ref.rollout, "enable_request_id"):
                enable_request_id: bool = self.config.actor_rollout_ref.rollout.enable_request_id
                assert isinstance(enable_request_id, bool), f"enable_request_id is bool value"

            t_start = time.time()
            request_id = request_id if enable_request_id else None # type: ignore
            # callback llm server, messages.size=1
            llm_output: dict = {}
            try:
                llm_output = self.llm_chat_fn(trajectory.steps, request_id=request_id)
            except Exception as e:
                logger.exception(f"call llm_chat_fn error with {e}")
                break

            time_cost = round(time.time() - t_start, 4)
            new_request_id: str = llm_output.pop("request_id", "")

            info_dict = {
                "act_step": act_step,
                "llm_output": llm_output,
                "new_request_id": new_request_id,
                "request_id": request_id,
                "time_cost": time_cost,
            }
            logger.info(f"info_dict={json.dumps(info_dict)}")

            request_id = new_request_id
            old_trajectory=trajectory.copy()
            trajectory.steps.append(llm_output)

            try:
                env_output = env.step(instance_id, llm_output)
                env_messages: list[dict] = env_output["state"]
            except Exception as e:
                logger.exception(f"call env.step error with {e}")
                break
            # convert role_tool to role_user message
            # breakpoint()
            
            # useless: for tool role
            assert len(env_messages)>0, "env returns empty messages"
            for env_message in env_messages:
                if env_message["role"] == "tool":
                    env_message = cast(dict, convert_tool_to_user_message(env_message, format="qwen"))
                
                state_content: str = env_message["content"]
                
                env_message["content"] = clip_state_content_correctly(
                    self.tokenizer, 
                    state_content,
                    self.max_env_len
                )
                

                trajectory.steps.append(sanitize_env_state(env_message))
                # log state
                self._state_recorder.add_state(old_trajectory, llm_output['content'], env_message['content'])
            trajectory.is_terminated = env_output["is_terminated"]
            
            # add instruction
            records=self._state_recorder.get_state(trajectory)
            if len(records)>0:
                instruction="In the past interactions at this moment, you have output these action and observed these states already:\n"
                for id, record in enumerate(records):
                    instruction+=f"## {id+1}.\n"
                    instruction+=f"action:\n{record[0]}\n\n"
                    instruction+=f"state:\n{record[1]}\n\n"
                instruction+="## Continue your work."
                instruction+="Please continue your work. You are not expected to repeat the action you have already observed."
                trajectory.steps.append({"role":"system","content":instruction})

            if trajectory.is_terminated:
                break
        if self._reward_calculator is not None:
            score = self._reward_calculator.calculate_reward(trajectory, env)
        else:
            score = env.evaluate(instance_id, params={"sparse": False})
        trajectory.reward.outcome = score
        trajectory.reward.description = "Outcome 1 = success, 0 = failure."

        if trajectory.steps[-1]["role"] == "user":
            trajectory.steps = trajectory.steps[:-1]

        return trajectory


def sanitize_env_state(state: dict):
    """
    sanitize env state
    """
    # remove empty tool_calls
    if "tool_calls" in state and not state["tool_calls"]:
        state.pop("tool_calls")
    
    return state