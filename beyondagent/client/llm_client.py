from abc import ABC, abstractmethod
import json
import os
import time
from typing import Any, Optional, Protocol, Iterator, Generator, cast

from loguru import logger
import requests

class LlmException(Exception):
    def __init__(self,typ: str):
        self._type=typ
    
    @property
    def typ(self):
        return self._type
        

class DashScopeClient:
    """阿里云百炼API客户端"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "qwen-plus", 
                 temperature: float = 0.7, max_tokens: int = 2048):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Please set DASHSCOPE_API_KEY environment variable or pass it directly.")
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def set_model(self, model_name: str):
        self.model_name = model_name
    
    def chat(self, messages: list[dict[str, str]], sampling_params: dict[str, Any]) -> str:
        res = ""
        for x in self.chat_stream(messages, sampling_params):
            res += x
        return res
    
    def chat_stream(self, messages: list[dict[str, str]], sampling_params: dict[str, Any]) -> Generator[str, None, None]:
        """流式聊天，返回生成器"""
        return self.chat_stream_with_retry(messages, **sampling_params)
    
    def chat_completion(self, messages: list[dict[str, str]], stream: bool = False, **kwargs) -> str | Generator[str, None, None]:
        """发起聊天完成请求"""
        url = f"{self.base_url}/chat/completions"
        
        # 合并参数
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
            **kwargs
        }
        
        try:
            if stream:
                return self._handle_stream_response(url, params)
            else:
                return self._handle_normal_response(url, params)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return "" if not stream else (x for x in [])
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response: {e}")
            return "" if not stream else (x for x in [])
        except Exception as e:
            logger.error(f"Unexpected error in API call: {e}")
            return "" if not stream else (x for x in [])
    
    def _handle_normal_response(self, url: str, params: dict) -> str:
        """处理非流式响应"""
        response = requests.post(url, headers=self.headers, json=params, timeout=600)
        if not response.ok:
            # check inappropriate content
            try:
                error_json=response.json()['error']
                if "inappropriate content" in error_json['message']:
                    raise LlmException("inappropriate content")
                if "limit" in error_json['message']:
                    raise LlmException("hit limit")
            except LlmException as e:
                raise
            except:
                logger.error(f"API request failed: {response.text}")
                response.raise_for_status()
        
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"].strip()
        else:
            logger.error(f"Unexpected response format: {result}")
            return ""
    
    def _handle_stream_response(self, url: str, params: dict) -> Generator[str, None, None]:
        """处理流式响应"""
        response = requests.post(url, headers=self.headers, json=params, stream=True, timeout=600)
        if not response.ok:
            # check inappropriate content
            try:
                error_json=response.json()['error']
                if "inappropriate content" in error_json['message']:
                    raise LlmException("inappropriate content")
                if "limit" in error_json['message']:
                    raise LlmException("hit limit")
            except LlmException as e:
                raise
            except:
                logger.error(f"API request failed: {response.text}")
                response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # 移除 'data: ' 前缀
                    if data == '[DONE]':
                        break
                    
                    try:
                        chunk = json.loads(data)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            choice = chunk["choices"][0]
                            if "delta" in choice and "content" in choice["delta"]:
                                content = choice["delta"]["content"]
                                if content:
                                    yield content
                    except json.JSONDecodeError:
                        continue  # 跳过无法解析的行
    
    def chat_with_retry(self, messages: list[dict[str, str]], max_retries: int = 3, 
                       retry_delay: float = 1.0, **kwargs) -> str:
        """带重试机制的聊天完成"""
        for attempt in range(max_retries):
            try:
                result = cast(str,self.chat_completion(messages, stream=False, **kwargs))
                if result:  # 如果获得了有效响应
                    return result
            
            except LlmException as e:
                if e.typ=='inappropriate content':
                    logger.warning(f"llm return inappropriate content, which is blocked by the remote")
                    return "[inappropriate content]"
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
            if attempt < max_retries - 1:  # 不是最后一次尝试
                time.sleep(retry_delay * (2 ** attempt))  # 指数退避
        
        logger.error(f"All {max_retries} attempts failed")
        return ""
    
    def chat_stream_with_retry(self, messages: list[dict[str, str]], max_retries: int = 3, 
                              retry_delay: float = 10.0, **kwargs) -> Generator[str, None, None]:
        """带重试机制的流式聊天完成"""
        for attempt in range(max_retries):
            try:
                stream_generator = cast(Generator[str, None, None], self.chat_completion(messages, stream=True, **kwargs))
                # 尝试获取第一个chunk来验证连接
                first_chunk = next(stream_generator, None)
                if first_chunk is not None:
                    yield first_chunk
                    # 继续生成剩余内容
                    for chunk in stream_generator:
                        yield chunk
                    return  # 成功完成，退出重试循环
            except LlmException as e:
                if e.typ=='inappropriate content':
                    logger.warning(f"llm return inappropriate content, which is blocked by the remote")
                    yield "[inappropriate content]"
                    return
            except Exception as e:
                logger.warning(f"Stream attempt {attempt + 1} failed: {e}")
                
            if attempt < max_retries - 1:  # 不是最后一次尝试
                time.sleep(retry_delay * (2 ** attempt))  # 指数退避
        
        logger.error(f"All {max_retries} stream attempts failed")
        
        return


# 使用示例
if __name__ == "__main__":
    client = DashScopeClient(model_name='qwq-32b')
    
    messages = [
        {"role": "user", "content": "写一首关于春天的诗"}
    ]
    
    # # 非流式调用
    # print("=== 非流式响应 ===")
    # response = client.chat_completion(messages)
    # print(response)
    
    print("\n=== 流式响应 ===")
    # 流式调用
    for chunk in client.chat_completion(messages, stream=True):
        print(chunk, end='', flush=True)
    
    print("\n\n=== 带重试的流式响应 ===")
    # 带重试的流式调用
    for chunk in client.chat_stream(messages, {}):
        print(chunk, end='', flush=True)