import datetime
import json
import threading
from typing import cast

from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import ModelResponse

from agent_scaling.logger import LLM_LEVEL_NAME, PROMPT_LEVEL_NAME, logger


class LocalLogger(CustomLogger):

    _log_lock = threading.Lock()

    def __init__(self, prompt_only: bool = False, **kwargs):
        self.prompt_only = prompt_only
        super().__init__(**kwargs)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        response: ModelResponse = cast(ModelResponse, response_obj)
        def timestamp(value):
            return value.timestamp() if isinstance(value, datetime.datetime) else float(value)

        elapsed_time = timestamp(end_time) - timestamp(start_time)

        if hasattr(response, "usage"):
            usage = response.usage.model_dump()  # type: ignore
        else:
            usage = {"completion_tokens": 0, "total_tokens": 0}
        if self.prompt_only:
            message = response.choices[0].message.model_dump()  # type: ignore
            if message.get("content"):
                message = message["content"]
            elif message.get("tool_calls"):
                message = json.dumps(message["tool_calls"], indent=2)
            elif message.get("function_call"):
                message = json.dumps(message["function_call"], indent=2)
            else:
                message = ""
        else:
            message = response.choices[0].message.model_dump_json(indent=2)  # type: ignore

        with self._log_lock:
            logger.bind(
                messages=kwargs.get("messages", []),  # type: ignore
            ).log(PROMPT_LEVEL_NAME, "")
            logger.bind(
                model=response.model,
                message=message,
                elapsed_time=elapsed_time,
                usage=usage,
                from_cache=response._hidden_params.get("cache_hit", False),
            ).log(LLM_LEVEL_NAME, "")
