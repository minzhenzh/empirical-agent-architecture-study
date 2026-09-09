"""WorkBench dataset adapter using upstream state-change evaluation."""

from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from agent_scaling.datasets.base import (
    Dataset,
    DatasetInstance,
    DatasetInstanceOutput,
)
from agent_scaling.datasets.registry import (
    register_dataset,
    register_dataset_instance,
)
from agent_scaling.env.workbench_backend import get_workbench_backend

DATASET_IDS = ["workbench"]

@register_dataset_instance(DATASET_IDS)
class WorkbenchInstance(DatasetInstance):
    task_id: str
    instructions: str
    expected_actions: List[Dict[str, Any]] = Field(default_factory=list)
    expected_answer: Optional[str] = None
    tools: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context):
        # If only expected_actions is given, derive a textual reference that
        # contains the function-call strings for downstream comparison.
        if self.expected_answer:
            self.expected_output = self.expected_answer
        else:
            calls = [a.get("call") for a in self.expected_actions if a.get("call")]
            self.expected_output = "\n".join(calls) if calls else ""

    def get_prompt_info(self) -> Dict[str, str]:
        return {
            "task_id": self.task_id,
            "question": self.instructions,
        }


@register_dataset(DATASET_IDS)
class WorkbenchDataset(Dataset):
    instances: List[WorkbenchInstance]
    dataset_id: str = "workbench"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # Grading is deterministic and uses the upstream mock database state.
    _required_eval_prompts: List[str] = []
    _require_llm_eval: bool = False

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        # Load and validate the pinned checkout before the experiment runner
        # starts worker threads. Upstream imports read fixture paths relative
        # to their repository root during initialization.
        get_workbench_backend()

    def get_instance_eval_output(
        self, instance_output: DatasetInstanceOutput[WorkbenchInstance]
    ) -> Dict[str, Any]:
        return {
            "answer": instance_output.agent_output,
            "expected_output": instance_output.data_instance.expected_output,
        }

    def get_instance_eval_metrics(
        self, instance_output: DatasetInstanceOutput[WorkbenchInstance]
    ) -> Dict[str, Union[int, float, str, bool]]:
        instance = instance_output.data_instance
        response = instance_output.agent_output or ""
        backend = get_workbench_backend()
        expected_actions = backend.parse_actions(
            "\n".join((action.get("call") or "") for action in instance.expected_actions)
        )
        predicted_actions = [
            action
            for artifact in instance_output.environment_artifacts
            if artifact.get("kind") == "workbench_actions"
            for action in artifact.get("actions", [])
            if isinstance(action, dict)
        ]
        if not predicted_actions:
            predicted_actions = backend.parse_actions(str(response))
        exact_match = backend.is_exact_match(predicted_actions, expected_actions)
        is_correct = backend.is_correct(predicted_actions, expected_actions)

        return {
            "is_correct": is_correct,
            "is_exact_match": exact_match,
            "n_predicted_calls": len(predicted_actions),
            "n_expected_calls": len(expected_actions),
            "grading_method": "upstream_state_change",
            "extracted_answer": response[:500],
        }

    def get_metrics(self, eval_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Defensive: exp_runner inserts {"resolved": 0, ...} on agent failure.
        if not eval_outputs:
            return {"avg_accuracy": 0.0, "n": 0}
        exact = sum(1 for e in eval_outputs if bool(e.get("is_exact_match")))
        correct = sum(1 for e in eval_outputs if bool(e.get("is_correct")))
        return {
            "avg_accuracy": correct / len(eval_outputs),
            "exact_match_rate": exact / len(eval_outputs),
            "n": len(eval_outputs),
            "n_correct": correct,
        }
