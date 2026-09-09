"""Stateful WorkBench environment backed by the pinned upstream benchmark."""

import json
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool

from agent_scaling.env.base import AgentEnvironment

from .registry import register_env
from .tools import cls_tool
from .workbench_backend import DOMAIN_ALIASES, get_workbench_backend


@register_env("workbench")
class WorkbenchEnvironment(AgentEnvironment):
    """Expose upstream domain tools with state isolated per environment."""

    def __init__(self, *args, tools=None, dataset_instance=None, **kwargs):
        self.is_done: bool = False
        self.success: bool = False
        self.num_steps: int = 0
        self.final_answer: Optional[str] = None
        self.action_log: List[Dict[str, Any]] = []
        self.backend = get_workbench_backend()
        self._workbench_state = self.backend.new_state()

        metadata = getattr(dataset_instance, "metadata", {}) or {}
        domains = metadata.get("domains") or []
        if isinstance(domains, str):
            domains = [domains]
        if not domains and dataset_instance is not None:
            inferred = str(getattr(dataset_instance, "task_id", "")).split("-", 1)[0]
            domains = [inferred]
        if dataset_instance is None:
            # Base agent construction creates a bootstrap environment before a
            # concrete instance is assigned. Actual run environments are
            # recreated with task-specific domains.
            domains = list(DOMAIN_ALIASES)
        normalized_domains = [DOMAIN_ALIASES[d] for d in domains if d in DOMAIN_ALIASES]
        if not normalized_domains:
            raise ValueError(
                f"WorkBench instance has no recognized domains: {domains!r}"
            )

        dynamic_tools: Dict[str, StructuredTool] = {}
        for canonical_name in self.backend.tools_for_domains(normalized_domains):
            upstream_tool = self.backend.tools[canonical_name]
            alias = self.backend.alias_for(canonical_name)

            def invoke_upstream(
                _canonical_name=canonical_name,
                **tool_args,
            ):
                self.action_log.append(
                    {"tool": _canonical_name, "args": dict(tool_args)}
                )
                return self.backend.invoke(
                    _canonical_name, tool_args, self._workbench_state
                )

            dynamic_tools[alias] = StructuredTool.from_function(
                func=invoke_upstream,
                name=alias,
                description=(
                    f"WorkBench tool `{canonical_name}`. "
                    f"{upstream_tool.description}"
                ),
                args_schema=upstream_tool.args_schema,
            )

        submit_tool = self.submit
        dynamic_tools[submit_tool.name] = submit_tool
        self.env_tools = dynamic_tools
        # WorkBench tools are selected per instance. An empty YAML tool list
        # means "use the selected domain toolkit"; stale explicit lists would
        # otherwise silently recreate the incomplete five-tool proxy.
        if tools:
            raise ValueError(
                "WorkBench selects tools from each instance's domains; set "
                "dataset.env.tools to [] in run_conf/dataset/workbench.yaml."
            )
        super().__init__(
            *args,
            tools=None,
            dataset_instance=dataset_instance,
            **kwargs,
        )

    @cls_tool
    def submit(self, reasoning: str) -> str:
        """Finish the task after executing the necessary business tools."""
        self.final_answer = reasoning
        self.is_done = True
        self.success = True
        return self.export_submission(reasoning)

    def export_action_block(self) -> str:
        """Return the canonical machine-readable trace for dataset grading."""
        return (
            "<workbench_actions>\n"
            f"{json.dumps(self.action_log, default=str)}\n"
            "</workbench_actions>"
        )

    def export_submission(self, reasoning: str) -> str:
        """Combine human-readable status with the canonical action trace."""
        return f"Final Answer: {reasoning}\n{self.export_action_block()}"

    def evaluation_artifact(self) -> Dict[str, object]:
        """Expose executed actions through the generic environment interface."""
        return {
            "kind": "workbench_actions",
            "agent_id": self.agent_id,
            "actions": [
                {"tool": action["tool"], "args": dict(action["args"])}
                for action in self.action_log
            ],
        }

    def env_done(self) -> bool:
        return self.is_done
