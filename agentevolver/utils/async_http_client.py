import asyncio
import time
from typing import Any

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from agentevolver.enumeration.http_enum import HttpEnum


class AsyncHttpClient(BaseModel):
    url: str = Field(default="")
    keep_alive: bool = Field(default=False, description="if true, use session to keep long connection")
    timeout: int = Field(default=300, description="request timeout, second")
    return_default_if_error: bool = Field(default=True)

    request_start_time: float = Field(default_factory=time.time)
    request_time_cost: float = Field(default=0.0, description="request time cost")

    retry_sleep_time: float = Field(default=0.5, description="interval time for retry")
    retry_time_multiplier: float = Field(default=2.0, description="retry time multiplier")
    retry_max_count: int = Field(default=1, description="maximum number of retries")

    _client: Any | aiohttp.ClientSession = PrivateAttr()

    @model_validator(mode="after")
    def init_client(self):
        self._client = aiohttp.ClientSession(timeout=self.timeout) if self.keep_alive else aiohttp
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
        self.request_time_cost: float = time.time() - self.request_start_time

    async def close(self):
        if isinstance(self._client, aiohttp.ClientSession):
            await self._client.close()

    def parse_result(self, response: aiohttp.ClientResponse | None = None, **kwargs):
        return response.json()

    def return_default(self, **kwargs) -> Any:
        return None

    async def request(
            self,
            data: str | Any = None,
            json_data: dict = None,
            headers: dict = None,
            http_enum: HttpEnum | str = HttpEnum.POST,
            **kwargs,
    ) -> Any:
        retry_sleep_time = self.retry_sleep_time
        method = http_enum
        if isinstance(http_enum, HttpEnum):
            http_enum = method.value

        for i in range(self.retry_max_count):
            try:
                response = await self._client.request(method=method, url=self.url, data=data, json=json_data,
                                                      headers=headers)

                result = self.parse_result(response=response, data=data, json_data=json_data, headers=headers,
                                           http_enum=http_enum, **kwargs)
                return result

            except Exception as e:
                logger.exception(f"{self.__class__.__name__} {i}th request failed with args={e.args}")

                if i == self.retry_max_count - 1:
                    if self.return_default_if_error:
                        return self.return_default()
                    else:
                        raise e

                retry_sleep_time *= self.retry_time_multiplier
                await asyncio.sleep(retry_sleep_time)

        return None
