from loguru import logger

from beyondagent.client.em_client import EMClient
from beyondagent.client.env_client import EnvClient
from beyondagent.module.agent_flow.base_agent_flow import BaseAgentFlow
from beyondagent.schema.trajectory import Trajectory
from beyondagent.utils.utils import convert_tool_to_user_message


class AgentFlow(BaseAgentFlow):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.instruction_template_ids = self.tokenizer.encode("<|im_start|>user\n")
        self.response_template_ids = self.tokenizer.encode("<|im_start|>assistant\n")
        self.em_client = EMClient(base_url=self.config.experience_maker.base_url)

    def execute(self, trajectory: Trajectory, env: EnvClient, instance_id: str, **kwargs) -> Trajectory:
        if self.config.experience_maker.enable_context_generator:  # add by jinli 0618
            history_experience = self.em_client.call_context_generator(
                trajectory=trajectory,
                retrieve_top_k=self.config.experience_maker.retrieve_top_k,
                workspace_id=self.config.experience_maker.workspace_id)


            if history_experience:
                logger.info(f"history_experience={history_experience}")
                new_content = history_experience + "\n\n" + trajectory.steps[-1]["content"]
                trajectory.steps[-1]["content"] = new_content
            else:
                logger.info(f"history_experience is empty!")

        request_id: str = ""
        for act_step in range(self.max_steps):
            # if use qwen3, add /no_think
            if self.config.actor_rollout_ref.rollout.use_qwen3:
                trajectory.steps[-1]["content"] += " /no_think"

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

            # callback llm server, messages.size=1
            llm_output = self.llm_chat_fn(trajectory.steps, request_id=request_id)
            new_request_id = llm_output.pop("request_id", None)
            logger.info(f"llm_output={llm_output} "
                        f"new_request_id={new_request_id} "
                        f"request_id={request_id}")
            request_id = new_request_id

            trajectory.steps.append(llm_output)

            env_output = env.step(instance_id, llm_output)
            # convert role_tool to role_user message
            # breakpoint()
            
            # useless: for tool role
            if env_output["state"]["role"] == "tool":
                env_output["state"] = convert_tool_to_user_message(env_output["state"], self.tokenizer, format="qwen")
            
            state_content: str = env_output["state"]["content"]
            if len(self.tokenizer(state_content, return_tensors="pt", padding=False)["input_ids"][
                       0]) > self.max_env_len:
                env_output["state"]["content"] = state_content[:self.max_env_len]

            trajectory.steps.append(env_output["state"])
            trajectory.is_terminated = env_output["is_terminated"]

            # TODO require env
            # trajectory.reward.outcome = env_output["reward"]["outcome"]
            # trajectory.reward.description = env_output["reward"]["description"]
            # trajectory.reward.outcome = env_output["reward"]
            # trajectory.reward.description = "Outcome 1 = success, 0 = failure."

            if trajectory.is_terminated:
                break
        
        score = env.evaluate(instance_id, params={"sparse": False})
        trajectory.reward.outcome = score
        trajectory.reward.description = "Outcome 1 = success, 0 = failure."

        if trajectory.steps[-1]["role"] == "user":
            trajectory.steps = trajectory.steps[:-1]

        return trajectory
