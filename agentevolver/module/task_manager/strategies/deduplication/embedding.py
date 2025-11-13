import os
import uuid
from typing import Any, Optional, Sequence
import chromadb
from chromadb.config import Settings
from loguru import logger

from agentevolver.client.embedding_client import OpenAIEmbeddingClient
from agentevolver.schema.trajectory import Trajectory

MAX_INPUT_LEN=8192

class EmbeddingClient:
    def __init__(self, similarity_threshold: float, base_url: str = 'https://dashscope.aliyuncs.com/compatible-mode/v1', 
                 api_key: Optional[str] = None, model: str = "text-embedding-v4",
                 chroma_db_path: str = "./chroma_db", collection_name: str = "trajectories"):
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        assert api_key is not None, "DASHSCOPE_API_KEY is required"
        
        self._client = OpenAIEmbeddingClient(api_key=api_key, base_url=base_url, model_name=model)
        self.similarity_threshold = similarity_threshold
        
        self._chroma_client = chromadb.PersistentClient(
            path=chroma_db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self._collection = self._chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self._id_mapping: dict[int, str] = {}
        self._reverse_id_mapping: dict[str, int] = {}
    
    def add(self, text: str, id: int):
        """
        Add text and ID to ChromaDB
        """
        embedding = self._client.get_single_embedding(text)
        
        chroma_id = f"doc_{id}_{uuid.uuid4().hex[:8]}"
        
        self._id_mapping[id] = chroma_id
        self._reverse_id_mapping[chroma_id] = id
        
        self._collection.add(
            embeddings=[embedding],
            documents=[text],
            ids=[chroma_id],
            metadatas=[{"original_id": id, "text_length": len(text)}]
        )
    
    def find_by_text(self, text: str) -> Optional[int]:
        """
        Find a similar text in ChromaDB, return the corresponding ID
        """
        if self._collection.count() == 0:
            return None
        
        query_embedding = self._client.get_single_embedding(text)
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=1,  # only the top result
            include=["documents", "metadatas", "distances"]
        )
        
        if not results["ids"] or not results["ids"][0]:
            return None
        
        distance = results["distances"][0][0] # type: ignore
        similarity = 1 - distance
        
        if similarity >= self.similarity_threshold:
            chroma_id = results["ids"][0][0]
            return self._reverse_id_mapping.get(chroma_id)
        else:
            return None
    
    def find_top_k_by_text(self, text: str, k: int = 5) -> list[tuple[int, float, str]]:
        """
        Find the top k similar documents
        """
        if self._collection.count() == 0:
            return []
        
        query_embedding = self._client.get_single_embedding(text)
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"]
        )
        
        if not results["ids"] or not results["ids"][0]:
            return []
        
        result_list = []
        for i, chroma_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i] # type: ignore
            similarity = 1 - distance
            document = results["documents"][0][i] # type: ignore
            original_id = self._reverse_id_mapping.get(chroma_id)
            
            if original_id is not None:
                result_list.append((original_id, similarity, document))
        
        return result_list
    
    def _embedding(self, texts: Sequence[str], bs=10) -> list[list[float]]:
        """
        Get the embedding of texts
        """
        res: list[list[float]] = []
        for i in range(0, len(texts), bs):
            res.extend(self._client.get_multiple_embeddings(texts[i:i+bs]))
        
        return res
    
    def get_all_stored_texts(self) -> dict[int, str]:
        """
        Get all stored texts
        """
        all_data = self._collection.get(include=["documents", "metadatas"])
        result = {}
        
        if all_data["ids"]:
            for i, chroma_id in enumerate(all_data["ids"]):
                original_id = self._reverse_id_mapping.get(chroma_id)
                if original_id is not None:
                    result[original_id] = all_data["documents"][i] # type: ignore
        
        return result
    
    def remove(self, id: int) -> bool:
        """
        Remove the text and embedding vector of the specified ID
        """
        chroma_id = self._id_mapping.get(id)
        if chroma_id is None:
            return False
        
        try:
            self._collection.delete(ids=[chroma_id])
            
            del self._id_mapping[id]
            del self._reverse_id_mapping[chroma_id]
            
            return True
        except Exception:
            return False
    
    def clear(self):
        """clear all stored texts and embeddings"""
        try:
            self._chroma_client.delete_collection(self._collection.name)
            self._collection = self._chroma_client.get_or_create_collection(
                name=self._collection.name,
                metadata={"hnsw:space": "cosine"}
            )
            
            self._id_mapping.clear()
            self._reverse_id_mapping.clear()
        except Exception as e:
            print(f"failed to clear stores: {e}")
    
    def size(self) -> int:
        """get the number of stored texts"""
        return self._collection.count()
    
    def get_collection_info(self) -> dict:
        """get the collection info of ChromaDB"""
        return {
            "name": self._collection.name,
            "count": self._collection.count(),
            "metadata": self._collection.metadata
        }


def pack_trajectory(trajectory: Trajectory) -> str:
    """
    pack the trajectory into a string
    
    Args:
        trajectory (Trajectory): the trajectory
        
    Returns:
        str: packed trajectory
    """
    res = ""
    for message in trajectory.steps:
        res += f"{message['role']}\n{message['content']}\n\n"
    
    return res[-MAX_INPUT_LEN:] # TODO: text-embedding-v4 max length


class StateRecorder:
    def __init__(self, similarity_threshold: float, chroma_db_path: str = "./chroma_db", collection_name: str = "trajectories"):
        self._client = EmbeddingClient(
            similarity_threshold=similarity_threshold,
            chroma_db_path=chroma_db_path,
            collection_name=collection_name
        )
        
        self._mp: dict[int, list[tuple[str, str]]] = {}
        self._idx = 0
    
    def add_state(self, trajectory: Trajectory, action: str, observation: str):
        """
        add state record
        
        Args:
            trajectory (Trajectory): trajectory
            action (str): action
            observation (str): observation
        """
        key = pack_trajectory(trajectory)
        id = self._client.find_by_text(key)
        if id is None:
            id = self._idx
            self._mp[id] = []
            self._client.add(key, id)
            self._idx += 1
        
        self._mp[id].append((action, observation))
    
    def get_state(self, trajectory: Trajectory) -> list[tuple[str, str]]:
        """
        get the state record
        
        Args:
            trajectory (Trajectory): trajectory
            
        Returns:
            list[tuple[str, str]]: list of (action, observation)
        """
        key = pack_trajectory(trajectory)
        id = self._client.find_by_text(key)
        if id is None:
            return []
        else:
            logger.debug(f"[embedding] key hit, similar state detected!, #state={len(self._mp)}")
            return self._mp[id]
    
    def get_similar_states(self, trajectory: Trajectory, k: int = 5) -> list[tuple[int, float, list[tuple[str, str]]]]:
        """
        get the similar state records
        
        Args:
            trajectory (Trajectory): trajectory
            k (int): top k
            
        Returns:
            list[tuple[int, float, list[tuple[str, str]]]]: list, (ID, similarity, list of (action, observation))
        """
        key = pack_trajectory(trajectory)
        similar_results = self._client.find_top_k_by_text(key, k)
        
        result = []
        for original_id, similarity, _ in similar_results:
            if original_id in self._mp:
                result.append((original_id, similarity, self._mp[original_id]))
        
        return result

    
    def clear(self):
        """clear all records"""
        self._mp.clear()
        self._client.clear()
        self._idx = 0


# demo
if __name__ == "__main__":
    # install chromadb first: pip install chromadb
    
    # init StateRecorder
    recorder = StateRecorder(
        similarity_threshold=0.8,
        chroma_db_path="./my_chroma_db",
        collection_name="trajectory_states"
    )
    
    print("inited ChromaDB")
    
    