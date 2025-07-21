from typing import Optional

from beyondagent.schema.trajectory import Trajectory


class EmbeddingClient:
    def __init__(self, similarity_threshold:float):
        pass
    
    
    def add(self,text:str, id:int):
        raise NotImplementedError("TODO")
    
    
    def find_by_text(self, text: str) -> Optional[int]:
        raise NotImplementedError("TODO")
    
    
    def _embedding(self, text: str) -> list[float]:
        raise NotImplementedError("TODO")


class StateRecorder:
    def __init__(self, similarity_threshold:float):
        pass
    
    
    def add_state(self, trajectory:Trajectory, action: str, observation:str):
        raise NotImplementedError("TODO")
    
    
    def get_state(self,trajectory:Trajectory)->list[tuple[str,str]]:
        raise NotImplementedError("TODO")