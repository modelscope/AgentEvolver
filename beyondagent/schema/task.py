from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class Task(BaseModel):
    task_id: str = Field(default=...)

    env_type: str = Field(default="appworld")

    metadata: dict = Field(default_factory=dict)

    query: List | str = Field(default="")
    
    evaluator: str = Field(default="env")
    
    @property
    def is_query_empty(self):
        if isinstance(self.query, str):
            return self.query.strip() == ""
        elif isinstance(self.query, list):
            return len(self.query) == 0
    
    @property
    def first_query(self)->str:
        if isinstance(self.query, str):
            return self.query
        elif isinstance(self.query, list):
            return self.query[0]
        else:
            raise ValueError(f"query is not str or list")


class TaskObjective(BaseModel):
    task:Task=Field(...,description="task")
    ground_truth:str=Field(...,description="ground truth")
    confidence:Optional[float]=Field(None,description="confidence")
    reward:Optional[float]=Field(None,description="reward")
    
    @property
    def objective(self):
        return self.task.query