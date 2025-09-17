import os
import time
import threading
from loguru import logger
import requests
import json
from typing import List, Sequence, Union, Optional, Dict, Any


class RateLimiter:
    """
    A thread-safe rate limiter using the token bucket algorithm.

    Attributes:
        max_calls (int): Maximum number of calls allowed within the time window.
        time_window (int): Time window in seconds for which the call limit applies.
        interval (float): Minimum required interval between two consecutive calls.
        _lock (threading.Lock): Lock to ensure thread safety.
        _last_call_time (float): Timestamp of the last call made.
    """
    
    def __init__(self, max_calls: int, time_window: int = 60):
        """
        Initializes the rate limiter with given maximum calls and time window.

        Args:
            max_calls (int): Maximum number of calls allowed within the time window.
            time_window (int): Time window in seconds for which the call limit applies, default is 60 seconds.
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.interval = time_window / max_calls  # ⭐ Calculate the minimum interval between calls
        
        self._lock = threading.Lock()
        self._last_call_time = 0
        
        logger.info(f"Initializing rate limiter: {max_calls} calls/{time_window} seconds, minimum interval: {self.interval:.2f} seconds")
    
    def acquire(self):
        """
        Acquires permission to execute. If the call limit is exceeded, it blocks until the next available slot.

        This method ensures that the calls are spaced out according to the defined rate limit.
        """
        with self._lock:
            current_time = time.time()
            time_since_last_call = current_time - self._last_call_time
            
            if time_since_last_call < self.interval:
                wait_time = self.interval - time_since_last_call
                # logger.debug(f"Rate limit triggered, waiting for {wait_time:.2f} seconds")
                # Wait inside the lock to ensure thread safety
                time.sleep(wait_time)
                current_time = time.time()
            
            self._last_call_time = current_time  # ⭐ Update the last call time
            # logger.debug(f"Execution permission acquired, time: {current_time}")


class OpenAIEmbeddingClient:
    """
    Client class for OpenAI Embedding API.
    Supports calling embedding APIs in OpenAI format with rate limiting.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", 
                 model_name: str = "text-embedding-ada-002",
                 rate_limit_calls: int = 60, rate_limit_window: int = 60):
        """
        Initializes the OpenAI Embedding API client.

        Args:
            api_key (str): The API key for authentication.
            base_url (str): The base URL for the API, defaulting to the official OpenAI address.
            model_name (str): The name of the model to use, defaulting to text-embedding-ada-002.
            rate_limit_calls (int): The number of allowed calls within the rate limit window, defaulting to 60.
            rate_limit_window (int): The time window in seconds for the rate limit, defaulting to 60 seconds.
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')  # ⭐ Ensures the base URL does not end with a trailing slash
        self.model_name = model_name
        
        # Initialize the rate limiter
        self.rate_limiter = RateLimiter(rate_limit_calls, rate_limit_window)  # ⭐ Sets up the rate limiter with specified limits
        
        # Set up the request headers
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"  # ⭐ Constructs the authorization header using the provided API key
        }
        
        logger.info(f"初始化OpenAI Embedding客户端，限流: {rate_limit_calls}次/{rate_limit_window}秒")

    def get_embeddings(self, texts: Union[str, Sequence[str]], 
                      model: Optional[str] = None,
                      encoding_format: str = "float",
                      dimensions: Optional[int] = None,
                      user: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetches the embedding vectors for the provided texts with rate limiting.

        Args:
            texts (Union[str, Sequence[str]]): Text(s) for which to fetch the embeddings, can be a single string or a list of strings.
            model (Optional[str]): Name of the model to use; if not specified, the model set during initialization is used.
            encoding_format (str): Encoding format for the embeddings, default is "float".
            dimensions (Optional[int]): Output dimensionality (supported by some models).
            user (Optional[str]): User identifier.

        Returns:
            Dict[str, Any]: The API response as a dictionary.

        Raises:
            requests.RequestException: If there is an issue with the request.
            ValueError: If the input parameters are invalid.
        """
        # Rate limiting control
        self.rate_limiter.acquire()  # ⭐ Acquires a token from the rate limiter to ensure the request does not exceed the allowed rate
        
        # Parameter validation
        if not texts:
            raise ValueError("texts不能为空")
        
        # Construct the request payload
        payload = {
            "input": texts,
            "model": model or self.model_name,
            "encoding_format": encoding_format
        }
        
        # Add optional parameters
        if dimensions is not None:
            payload["dimensions"] = dimensions
        if user is not None:
            payload["user"] = user
        
        # Send the request
        url = f"{self.base_url}/embeddings"
        
        try:
            response = requests.post(
                url, 
                headers=self.headers, 
                json=payload,
                timeout=30
            )
            if not response.ok:
                logger.error(f"请求失败: {response.status_code} {response.reason}")
                logger.error(f"失败json: {response.json()}")
                response.raise_for_status()
            
            return response.json()
            
        except requests.RequestException as e:
            raise requests.RequestException(f"请求失败: {e}")

    def get_single_embedding(self, text: str, **kwargs) -> List[float]:
        """
        Retrieves the embedding vector for a single piece of text. This is a simplified method that wraps around the `get_embeddings` method.

        Args:
            text (str): The text for which to retrieve the embedding vector.
            **kwargs: Additional arguments to pass to the `get_embeddings` method.

        Returns:
            List[float]: The embedding vector for the provided text.
        """
        result = self.get_embeddings(text, **kwargs)  # ⭐ Calls the get_embeddings method with the given text and additional arguments
        return result['data'][0]['embedding']

    def get_multiple_embeddings(self, texts: Sequence[str], **kwargs) -> List[List[float]]:
        """
        Retrieves the embedding vectors for multiple texts (simplified method).

        Args:
            texts (Sequence[str]): A list of texts to get the embedding vectors for.
            **kwargs: Additional arguments to pass to the `get_embeddings` method.

        Returns:
            List[List[float]]: A list of embedding vectors.
        """
        result = self.get_embeddings(texts, **kwargs)  # ⭐ Calls the `get_embeddings` method with provided texts and additional arguments
        return [item['embedding'] for item in result['data']]  # ⭐ Extracts the 'embedding' field from each item in the returned data

    def set_model(self, model_name: str):
        """
        Sets the default model name for the API client.

        Args:
            model_name (str): The name of the model to be used.
        """
        self.model_name = model_name  # ⭐ Set the model name

    def set_base_url(self, base_url: str):
        """
        Sets the base URL for the API, ensuring it does not end with a trailing slash.

        Args:
            base_url (str): The base URL for the API.
        """
        self.base_url = base_url.rstrip('/')  # ⭐ Remove trailing slash if present

    def set_api_key(self, api_key: str):
        """
        Sets the API key and updates the authorization header for the API requests.

        Args:
            api_key (str): The API key for authentication.
        """
        self.api_key = api_key
        self.headers["Authorization"] = f"Bearer {self.api_key}"  # ⭐ Update the authorization header

    def set_rate_limit(self, max_calls: int, time_window: int = 60):
        """
        Configures the rate limiter for the API client, specifying the maximum number of calls within a given time window.

        Args:
            max_calls (int): The maximum number of calls allowed in the time window.
            time_window (int): The time window in seconds. Default is 60 seconds.
        """
        self.rate_limiter = RateLimiter(max_calls, time_window)  # ⭐ Initialize the rate limiter
        logger.info(f"更新限流设置: {max_calls}次/{time_window}秒")

# 使用示例
if __name__ == "__main__":
    import threading
    import concurrent.futures
    
    # 初始化客户端，设置每分钟最多10次请求
    client = OpenAIEmbeddingClient(
        api_key=os.environ.get('OPENAI_API_KEY', 'test-key'),
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        model_name="text-embedding-v4",
        rate_limit_calls=10,  # 每分钟10次
        rate_limit_window=60
    )

    def test_embedding(thread_id: int, text: str):
        """
        Tests the embedding retrieval process in a multi-threaded environment.

        Args:
            thread_id (int): Identifier for the current thread.
            text (str): The text for which the embedding is to be retrieved.

        Returns:
            bool: True if the embedding was successfully retrieved, False otherwise.
        """
        try:
            start_time = time.time()
            embedding = client.get_single_embedding(f"{text} - Thread {thread_id}")  # ⭐ Retrieve the embedding for the provided text
            end_time = time.time()
            print(f"线程 {thread_id}: 成功获取embedding，耗时 {end_time - start_time:.2f}秒，维度: {len(embedding)}")
            return True
        except Exception as e:
            print(f"线程 {thread_id}: 错误 - {e}")
            return False
    
    try:
        print("=== 单线程测试 ===")
        # 单线程测试
        for i in range(3):
            start_time = time.time()
            embedding = client.get_single_embedding(f"Test text {i}")  # ⭐ Retrieve the embedding for the test text
            end_time = time.time()
            print(f"请求 {i+1}: 完成，耗时 {end_time - start_time:.2f}秒")
        
        print("\n=== 多线程测试 ===")
        # 多线程测试
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(8):  # 提交8个任务，超过限流数量
                future = executor.submit(test_embedding, i+1, "Hello world")  # ⭐ Submit tasks to the thread pool
                futures.append(future)
            
            # 等待所有任务完成
            results = [future.result() for future in concurrent.futures.as_completed(futures)]  # ⭐ Collect results from all completed futures
            successful_count = sum(results)
            print(f"多线程测试完成，成功: {successful_count}/{len(results)}")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")