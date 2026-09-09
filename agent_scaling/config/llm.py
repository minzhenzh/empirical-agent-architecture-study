from typing import Any, Dict, Literal, Optional

import litellm
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

from agent_scaling.llm.litellm_lc import ChatLiteLLMLC


class LLMParams(BaseModel):
    """
    see litellm.completion() for parameter details
    """

    model_config = ConfigDict(extra="allow")

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[list[str]] = None
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None
    # cache controls: https://docs.litellm.ai/docs/proxy/caching#dynamic-cache-controls
    cache: Optional[Dict[Literal["no-cache", "no-store"], bool]] = None

    @model_validator(mode="before")
    @classmethod
    def convert_omegaconf_values(cls, data: Any) -> Any:
        """Convert Hydra containers in provider-specific parameters to Python values."""

        def convert(value: Any) -> Any:
            if OmegaConf.is_config(value):
                return OmegaConf.to_container(value, resolve=True)
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, list):
                return [convert(item) for item in value]
            if isinstance(value, tuple):
                return tuple(convert(item) for item in value)
            return value

        return convert(data)

    @model_validator(mode="after")
    def check_cache(self) -> Self:
        if not self.cache:
            self.cache = {}
        return self


class LLMConfig(BaseModel):
    """
    use litellm.get_valid_models() to see model names
    """

    params: LLMParams
    model: str = "gemini/gemini-2.0-flash"

    def get_llm(
        self,
        *,
        agent_identifier: str = "agent_1",
        agent_role: str = "agent",
        architecture: str = "single-agent",
        log_langfuse: bool = False,
    ) -> ChatLiteLLMLC:
        # Disable litellm's internal retries - key rotation in ChatLiteLLMLC handles this
        litellm.num_retries = 0
        litellm.request_timeout = 120

        return ChatLiteLLMLC(
            model=self.model,
            max_retries=1,  # Disable tenacity retries in langchain_litellm
            agent_identifier=agent_identifier,
            agent_role=agent_role,
            architecture=architecture,
            log_langfuse=log_langfuse,
            **self.params.model_dump(exclude={"cache"}, exclude_none=True),
        )
