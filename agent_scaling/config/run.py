from typing import Any, Dict, List, Optional, Self

from litellm import _logging as litellm_logging
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_scaling.agents import AgentSystem, get_agent_cls
from agent_scaling.langfuse_client import get_lf_client
from agent_scaling.llm.call_logging import configure_llm_call_logging
from agent_scaling.utils import (
    disable_local_cache,
    enable_local_cache,
    enable_local_logging,
)

from .dataset import DatasetConfig
from .llm import LLMConfig
from .prompts import Prompt


class AgentConfig(BaseModel):
    """Agent configuration that captures all YAML fields and passes them to the agent."""

    model_config = ConfigDict(extra="allow")

    name: str
    prompts: Dict[str, Prompt] = Field(default_factory=dict)
    agents: Optional[List[str]] = None
    orchestrator: Optional[str] = None
    subagent: Optional[List[str]] = None

    @model_validator(mode="after")
    def check_prompts(self) -> Self:
        agent_cls = get_agent_cls(self.name)
        agent_cls.check_required_prompts(self.prompts)
        return self

    @model_validator(mode="before")
    @classmethod
    def add_prompt_names(cls, data: Any) -> Dict[str, Any]:
        data = dict(data)
        prompts = dict(data.get("prompts", {}))
        for k, prompt in prompts.items():
            prompt = dict(prompt)
            if prompt.get("name") is None:
                prompt["name"] = k
            prompts[k] = prompt
        data["prompts"] = prompts
        return data

    def get_agent_kwargs(self) -> Dict[str, Any]:
        """Get all extra fields from YAML config to pass to the agent constructor."""
        if not self.__pydantic_extra__:
            return {}
        # Convert OmegaConf DictConfig/ListConfig to plain Python types
        result = {}
        for k, v in self.__pydantic_extra__.items():
            try:
                from omegaconf import OmegaConf
                if OmegaConf.is_config(v):
                    v = OmegaConf.to_container(v, resolve=True)
            except ImportError:
                pass
            result[k] = v
        return result

    def get_run_metadata(self) -> Dict[str, Any]:
        prompts = {}
        assert self.prompts is not None, "Prompts must be defined in AgentConfig"
        for k, prompt in self.prompts.items():
            prompts[k] = {
                "name": prompt.name,
            }
        metadata: Dict[str, Any] = {
            "name": self.name,
            "prompts": prompts,
        }
        agent_kwargs = self.get_agent_kwargs()
        if agent_kwargs:
            metadata["agent_params"] = agent_kwargs
        for role in ("agents", "orchestrator", "subagent"):
            value = getattr(self, role)
            if value is not None:
                metadata[role] = value
        return metadata


class RunConfig(BaseModel):
    agent: AgentConfig
    dataset: DatasetConfig
    llm: LLMConfig
    run_name: str
    save_dir: Optional[str] = None
    log_langfuse: bool = False
    use_disk_cache: bool = False
    debug: bool = False
    max_instances: Optional[int] = None
    num_workers: int = 1

    @property
    def run_parallel(self) -> bool:
        return self.num_workers > 1

    def model_post_init(self, context: Any) -> None:
        self._validate_role_models()
        client = get_lf_client() if self.log_langfuse else None
        self.log_langfuse = (
            self.log_langfuse and client is not None
            # and self.dataset.langfuse_dataset is not None
        )
        if not self.log_langfuse:
            enable_local_logging(prompt_only=False)
        configure_llm_call_logging(self.save_dir)
        litellm_logging._disable_debugging()  # type: ignore
        # Disable litellm debugging logs
        if self.use_disk_cache:
            enable_local_cache()
        else:
            disable_local_cache()

    def get_agent(self) -> AgentSystem:
        architecture = self.agent.name
        agent_kwargs = self.agent.get_agent_kwargs()
        resolved = self._resolved_assignment()

        if architecture in {"multi-agent-independent", "multi-agent-decentralized"}:
            primary_config = resolved["agents"][0]["config"]
            agent_kwargs["agent_llms"] = [
                item["config"].get_llm(
                    agent_identifier=item["agent_identifier"],
                    agent_role="peer",
                    architecture=architecture,
                    log_langfuse=self.log_langfuse,
                )
                for item in resolved["agents"]
            ]
            identity = {
                "agent_identifier": "agent_1",
                "agent_role": "peer",
                "architecture": architecture,
                "log_langfuse": self.log_langfuse,
            }
        elif architecture in {"multi-agent-centralized", "multi-agent-hybrid"}:
            primary_config = resolved["orchestrator"]["config"]
            agent_kwargs["subagent_llms"] = [
                item["config"].get_llm(
                    agent_identifier=item["agent_identifier"],
                    agent_role="subagent",
                    architecture=architecture,
                    log_langfuse=self.log_langfuse,
                )
                for item in resolved["subagent"]
            ]
            identity = {
                "agent_identifier": "orchestrator",
                "agent_role": "orchestrator",
                "architecture": architecture,
                "log_langfuse": self.log_langfuse,
            }
        else:
            primary_config = self.llm
            identity = {
                "agent_identifier": "agent_1",
                "agent_role": "agent",
                "architecture": architecture,
                "log_langfuse": self.log_langfuse,
            }

        return get_agent_cls(architecture).from_config(
            llm_config=primary_config,
            dataset_config=self.dataset,
            prompts=self.agent.prompts,
            llm_identity=identity,
            **agent_kwargs,
        )

    def _model_config(self, model: str) -> LLMConfig:
        """Apply the base LLM parameters to a model identifier assigned to a role."""
        return LLMConfig(model=model, params=self.llm.params.model_copy(deep=True))

    def _validate_role_models(self) -> None:
        architecture = self.agent.name
        count = int(self.agent.get_agent_kwargs().get("n_base_agents", 3))
        has_roles = any(
            value is not None
            for value in (
                self.agent.agents,
                self.agent.orchestrator,
                self.agent.subagent,
            )
        )
        if architecture in {"multi-agent-independent", "multi-agent-decentralized"}:
            if self.agent.orchestrator is not None or self.agent.subagent is not None:
                raise ValueError(
                    f"{architecture} accepts agent.agents, not hierarchical roles"
                )
            if self.agent.agents is not None and len(self.agent.agents) != count:
                raise ValueError(
                    f"{architecture} requires exactly {count} agent models; "
                    f"got {len(self.agent.agents)}"
                )
        elif architecture in {"multi-agent-centralized", "multi-agent-hybrid"}:
            if self.agent.agents is not None:
                raise ValueError(
                    f"{architecture} accepts agent.orchestrator and agent.subagent, "
                    "not agent.agents"
                )
            if has_roles and self.agent.orchestrator is None:
                raise ValueError(
                    f"{architecture} requires agent.orchestrator"
                )
            if self.agent.subagent is not None and len(self.agent.subagent) != count:
                raise ValueError(
                    f"{architecture} requires exactly {count} subagent models; "
                    f"got {len(self.agent.subagent)}"
                )
            if has_roles and self.agent.subagent is None:
                raise ValueError(f"{architecture} requires agent.subagent")
        elif has_roles:
            raise ValueError(
                f"role model assignment is not supported for architecture '{architecture}'"
            )

    def _resolved_assignment(self) -> Dict[str, Any]:
        architecture = self.agent.name
        count = int(self.agent.get_agent_kwargs().get("n_base_agents", 3))
        if architecture in {"multi-agent-independent", "multi-agent-decentralized"}:
            models = self.agent.agents or [self.llm.model] * count
            return {
                "agents": [
                    {
                        "agent_identifier": f"agent_{index + 1}",
                        "model": model,
                        "config": self._model_config(model),
                    }
                    for index, model in enumerate(models)
                ]
            }
        if architecture in {"multi-agent-centralized", "multi-agent-hybrid"}:
            orchestrator_model = self.agent.orchestrator or self.llm.model
            subagent_models = self.agent.subagent or [self.llm.model] * count
            return {
                "orchestrator": {
                    "agent_identifier": "orchestrator",
                    "model": orchestrator_model,
                    "config": self._model_config(orchestrator_model),
                },
                "subagent": [
                    {
                        "agent_identifier": f"agent_{index + 1}",
                        "model": model,
                        "config": self._model_config(model),
                    }
                    for index, model in enumerate(subagent_models)
                ],
            }
        return {}

    def get_resolved_model_assignment(self) -> Dict[str, Any]:
        """Serializable role-to-model mapping saved with every run."""
        resolved = self._resolved_assignment()

        def serialize(value: Any) -> Any:
            if isinstance(value, LLMConfig):
                return value.model_dump(exclude_none=True)
            if isinstance(value, list):
                return [serialize(item) for item in value]
            if isinstance(value, dict):
                return {key: serialize(item) for key, item in value.items()}
            return value

        return serialize(resolved)

    def get_run_metadata(self) -> Dict[str, Any]:
        ret: Dict[str, Any] = {
            "agent": self.agent.get_run_metadata(),
            "llm": self.llm.model_dump(exclude_none=True),
            "dataset": self.dataset.model_dump(exclude_none=True),
            "resolved_model_assignment": self.get_resolved_model_assignment(),
        }
        if self.save_dir is not None:
            ret["save_dir"] = self.save_dir
        ret["run_name"] = self.run_name
        ret["num_workers"] = self.num_workers
        ret["log_langfuse"] = self.log_langfuse
        ret["use_disk_cache"] = self.use_disk_cache
        return ret
