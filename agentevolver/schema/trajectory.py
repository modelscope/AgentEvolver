from typing import Any, Dict, List

from pydantic import BaseModel, Field


class Reward(BaseModel):
    outcome: float = Field(default=0.0)
    success_rate: float = Field(default=0.0)
    madness: float = Field(default=0.0)
    description: str = Field(default="Outcome 1 denotes success, and 0 denotes failure.")

    metadata: dict = Field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.outcome > 0


class Trajectory(object):
    data_id: str = ""
    rollout_id: str = ""

    steps: List[dict] | None = None
    query: str = ""

    is_terminated: bool = False
    reward: Reward | None = None

    metadata: dict | None = None
    
    def __init__(self, 
                 data_id: str="",
                 rollout_id: str="",
                 steps: List[dict] | None = None,
                 query: str="",
                 is_terminated: bool = False,
                 reward: Reward | None = None,
                 metadata: dict | None = None):

        self.data_id = data_id
        self.rollout_id = rollout_id
        if steps is not None:
            self.steps = steps  # when init in cmt: skip this because steps is a function in cmt
        self.query = query
        self.is_terminated = is_terminated
        self.reward = reward
        self.metadata = metadata if metadata is not None else {}
    

    @property
    def success(self) -> bool:
        return self.reward is not None and self.reward.outcome > 0


class Sample(BaseModel):
    """The data model for single sample."""

    data_id: str = 0
    task_id: str = 0
    rollout_id: str = 0
    minor_index_id: int = 0
    messages: List[dict] = []
    extras: Dict[str, Any] = {}
    input_ids: List[int] = None
    prompt_ids: List[int] = None
    response_ids: List[int] = None
    attention_mask: List[int] = None
    prompt_attention_mask: List[int] = None
    response_attention_mask: List[int] = None
    position_ids: List[int] = None
    prompt_position_ids: List[int] = None
    response_position_ids: List[int] = None
    loss_mask: List[int] = None
    prompt_loss_mask: List[int] = None
    response_loss_mask: List[int] = None
    reward_scores: Dict[str, Any] = None
    max_prompt_len: int
    max_response_len: int
    max_model_len: int

    def truncate_output_ids(self) -> None:

        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask)
        assert len(self.prompt_ids) == len(self.prompt_attention_mask) == len(self.prompt_position_ids) == len(self.prompt_loss_mask)
        assert len(self.response_ids) == len(self.response_attention_mask) == len(self.response_position_ids) == len(self.response_loss_mask)
        assert isinstance(self.input_ids, list) and isinstance(self.prompt_ids, list) and isinstance(self.response_ids, list)

        truncate_any = False

        if len(self.prompt_ids) > self.max_prompt_len:
            truncate_any = True
            print(f"-------------------------------------------------------------------------------------------------------")
            print(f"Warning: prompt_ids length {len(self.prompt_ids)} exceeds max_prompt_len {self.max_prompt_len}, truncating.")
            print(f"-------------------------------------------------------------------------------------------------------")
            raise RuntimeError("Prompt length exceeds maximum allowed length. Please adjust the input data.")
            self.prompt_ids = self.prompt_ids[-self.max_prompt_len:]
            self.prompt_attention_mask = self.prompt_attention_mask[-self.max_prompt_len:]
            self.prompt_position_ids = self.prompt_position_ids[-self.max_prompt_len:]
            self.prompt_loss_mask = self.prompt_loss_mask[-self.max_prompt_len:]

        if len(self.response_ids) > self.max_response_len:
            truncate_any = True
            print(f"-------------------------------------------------------------------------------------------------------")
            print(f"Warning: response_ids length {len(self.response_ids)} exceeds max_response_len {self.max_response_len}, truncating.")
            print(f"-------------------------------------------------------------------------------------------------------")
            self.response_ids = self.response_ids[: self.max_response_len]
            self.response_attention_mask = self.response_attention_mask[: self.max_response_len]
            self.response_position_ids = self.response_position_ids[: self.max_response_len]
            self.response_loss_mask = self.response_loss_mask[: self.max_response_len]

        if truncate_any:
            self.input_ids = self.prompt_ids + self.response_ids
            self.attention_mask = self.prompt_attention_mask + self.response_attention_mask
            self.position_ids = self.prompt_position_ids + self.response_position_ids
            self.loss_mask = self.prompt_loss_mask + self.response_loss_mask



    def discard(self) -> None:
        """
        Discard the experience.
        """
        raise RuntimeError('Never use this method.')
