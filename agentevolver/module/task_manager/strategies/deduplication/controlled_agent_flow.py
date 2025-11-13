import json
import random
import time
from typing import Optional, cast

from loguru import logger

from agentevolver.client.em_client import EMClient
from agentevolver.client.env_client import EnvClient
from agentevolver.module.agent_flow.base_agent_flow import BaseAgentFlow
from agentevolver.module.agent_flow.reward_calculator import RewardCalculator
from agentevolver.module.task_manager.strategies.deduplication.embedding import StateRecorder
from agentevolver.module.task_manager.strategies.deduplication.prompts import prompt_state_summary
from agentevolver.schema.trajectory import Trajectory
from agentevolver.utils.utils import convert_tool_to_user_message, clip_state_content_correctly


# I want a extensible AgentFlow rather than patch.
class ControlledAgentFlow(BaseAgentFlow):

    def __init__(self,state_recorder:StateRecorder,reward_calculator:Optional[RewardCalculator]=None, max_record_len:int=200, **kwargs):
        super().__init__(**kwargs)
        self._state_recorder=state_recorder
        self._reward_calculator = reward_calculator
        self._max_record_len=max_record_len
        self._enable_context_generator=False

        self.instruction_template_ids = self.tokenizer.encode("<|im_start|>user\n")
        self.response_template_ids = self.tokenizer.encode("<|im_start|>assistant\n")
        self.em_client = EMClient(base_url=self.config.experience_maker.base_url)
        

    def execute(self, trajectory: Trajectory, env: EnvClient, instance_id: str, **kwargs) -> Trajectory:
        request_id: str = ""
        trajectory=trajectory.copy(deep=True) # clone the trajectory to avoid side effect
        
        rng=random.Random(trajectory.steps[0]['content']) # use system prompt as seed
        for act_step in range(self.max_steps):
            # 1. remove old system prompt
            new_steps = [
                s for s in trajectory.steps
                if not (s["role"] == "system" and
                        "Now decide the single best next action" in s["content"])
            ]
            trajectory.steps = new_steps

            # 2. generate records
            records = self._state_recorder.get_state(trajectory)
            if len(records) != 0:
                records_str = []
                for idx, (action, state) in enumerate(records, start=1):
                    records_str.append(f"## {idx}\n[action]\n{action[:self._max_record_len]}"
                                    f"\n\n[state]\n{state[:self._max_record_len]}\n")
                records_str = "\n".join(records_str)
                records_str = self._compress_state_action(records_str)

                # 3. construct enhanced prompt
                instruction = f"""## Local Context
You are currently **at the same location / state** as in the log below.

### Interaction Log (collapsed)
{records_str}

## Decision Guidelines
1. Prefer an **untried** action if it has a plausible chance of revealing new reward-relevant information.
2. You may **repeat** a previous action only if
a) it delivered high reward and you expect the reward to persist, **or**
b) you need confirmation to reduce uncertainty.
3. Avoid infinite loops: never repeat the same low-value action more than **2** times.
4. Think step-by-step and output your reasoning before taking action (max 100 tokens).
5. Return **exactly one** action.

Now decide the single best next action.""".strip()

                logger.debug(f"retrieve #records={len(records)}, prompt_len={len(instruction)}")
                
                # 4. add prompt to trajectory
                if rng.random() <= 1.0:     # p
                    trajectory.steps.append({
                        "role": "user",
                        "content": instruction
                    })
            
            assert len(trajectory.steps)>2
            assert trajectory.steps[0]['role'] == 'system'
            
            # if use qwen3, add /no_think
            if self.config.actor_rollout_ref.rollout.use_qwen3:
                trajectory.steps[-1]["content"] += " /no_think"

            prompt_text = self.tokenizer.apply_chat_template(trajectory.steps, 
                                                             tokenize=False,
                                                             add_generation_prompt=True)
            current_token_len = len(self.tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"][0])

            # yunpeng 0623: to prevent add an imend token to an uncompleted seq, 
            # because the message-type output will be applied chat_template.
            max_response_length = self.config.actor_rollout_ref.rollout.response_length # TODO qwen-plus max_length
            if current_token_len + max_response_length > self.max_model_len:
                logger.warning(f"exceed max model, current_token_len={current_token_len}, max_response_length={max_response_length}, max_model_len={self.max_model_len}")
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
                # use the trajectory that has no last user message as key
                self._state_recorder.add_state(old_trajectory, llm_output['content'], env_message['content'])
            trajectory.is_terminated = env_output["is_terminated"]
            
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
    
    
    def _compress_state_action(self, records: str)-> str:
        msg=[
            {
                'role': 'user',
                'content': prompt_state_summary.get_prompt_compress_states(records)
            }
        ]
        llm_output = self.llm_chat_fn(msg)
        return prompt_state_summary.parse_compressed_results(llm_output['content'])


def sanitize_env_state(state: dict):
    """
    sanitize env state
    """
    # remove empty tool_calls
    if "tool_calls" in state and not state["tool_calls"]:
        state.pop("tool_calls")
    
    return state