from pydantic import BaseModel, Field
from typing import List, Dict


class Task(BaseModel):
    task_id: str = Field(default=...)

    env_type: str = Field(default="appworld")

    metadata: dict = Field(default_factory=dict)

    query: List | str = Field(default="")