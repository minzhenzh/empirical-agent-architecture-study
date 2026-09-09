import concurrent.futures
import contextlib
import json
import os
import random
import time
import traceback
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel
from tqdm import tqdm

from agent_scaling.config.run import RunConfig
from agent_scaling.datasets import Dataset, DatasetInstance
from agent_scaling.logger import logger
from agent_scaling.llm.call_logging import llm_instance_context
from agent_scaling.utils import write_json, write_yaml


class InstanceSave(BaseModel):
    inp: Dict[str, Any]
    output: Dict[str, Any]
    metrics: Dict[str, Union[int, float, str]]
    expected_output: Optional[Any] = None


class InstanceExecutionError(RuntimeError):
    def __init__(self, error: Exception, metrics: Dict[str, Union[int, float, str]]):
        super().__init__(str(error))
        self.original_error = error
        self.metrics = metrics


class ExperimentRunner:
    def __init__(self, config: RunConfig):
        self.config = config
        self.log_langfuse = config.log_langfuse
        self.output_dir = config.save_dir
        self.dataset: Dataset = self.config.dataset.dataset
        self.lf_dataset = self.config.dataset.langfuse_dataset
        logger.debug(f"Logging to langfuse: {self.log_langfuse}")

    def _get_context_manager(self, index: int) -> contextlib.AbstractContextManager:
        if self.log_langfuse and self.lf_dataset is not None:
            return self.lf_dataset.items[index].run(
                run_name=self.config.run_name,
                run_description=self.config.llm.model,
                run_metadata=self.config.get_run_metadata(),
            )
        return contextlib.nullcontext()

    def _get_instances(self) -> List[DatasetInstance]:
        max_instances = 10 if self.config.debug else len(self.dataset.instances)
        max_instances = (
            min(max_instances, self.config.max_instances)
            if self.config.max_instances is not None
            else max_instances
        )

        if self.config.dataset.dataset_filter is not None:
            instances = [
                instance
                for i, instance in enumerate(self.dataset.instances)
                if eval(self.config.dataset.dataset_filter, {}, {"x": instance, "i": i})
            ]
        else:
            instances = list(self.dataset.instances)

        # Deterministic shuffle for SWE-bench to avoid alphabetical bias (all astropy first)
        # Skip shuffle for other datasets where ordering is already reasonable
        if "swebench" in self.dataset.dataset_id:
            rng = random.Random(42)
            rng.shuffle(instances)

        instances = instances[:max_instances]
        return instances

    def run(self):
        metrics = []
        instances = self._get_instances()

        if self.log_langfuse:
            iterator = tqdm(
                enumerate(instances),
                total=len(instances),
                desc=f"Evaluating {self.dataset.dataset_id} dataset instances",
            )
        else:
            iterator = enumerate(instances)

        consecutive_failures = 0
        max_consecutive_failures = 3
        for i, instance in iterator:
            if self.config.debug and i >= 10:
                break
            instance_dir = None
            if self.output_dir is not None:
                instance_dir = os.path.join(
                    self.output_dir, "instance_runs", f"{i:04d}"
                )
                os.makedirs(instance_dir, exist_ok=True)

            # Use the core worker function
            try:
                inst_metrics = self._process_single_instance(i, instance, instance_dir)
                metrics.append(inst_metrics)
                consecutive_failures = 0
            except Exception as e:
                logger.error(f"Instance {i} failed with error: {e}")
                # Record failure metrics so experiment continues
                # Use "resolved" key to match dataset get_metrics expectations
                metrics.append(
                    e.metrics
                    if isinstance(e, InstanceExecutionError)
                    else {"resolved": 0, "num_steps": 0, "success": 0}
                )
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(
                        f"Aborting experiment: {consecutive_failures} consecutive failures. "
                        f"Last error: {e}"
                    )
                    break

        return self._finalize_dataset_metrics(metrics)

    def _finalize_dataset_metrics(self, metrics: List[Any]) -> Dict[str, Any]:
        """Combine benchmark, execution-time, and LLM-call metrics for a run."""
        all_metrics = dict(self.dataset.get_metrics(metrics))
        all_metrics.update(self._execution_time_summary(metrics))
        all_metrics.update(self._llm_call_summary())
        if self.output_dir is not None:
            write_json(
                all_metrics,
                os.path.join(self.output_dir, "dataset_eval_metrics.json"),
                indent=True,
            )
        return all_metrics

    @staticmethod
    def _execution_time_summary(metrics: List[Any]) -> Dict[str, Any]:
        values = [
            float(metric["execution_time_s"])
            for metric in metrics
            if isinstance(metric, dict)
            and isinstance(metric.get("execution_time_s"), (int, float))
            and not isinstance(metric.get("execution_time_s"), bool)
        ]
        return {
            "execution_time_s_total": sum(values) if values else None,
        }

    def _llm_call_summary(self) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        if self.output_dir is not None:
            path = os.path.join(self.output_dir, "llm_calls.jsonl")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as stream:
                    for line_number, line in enumerate(stream, start=1):
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as error:
                            logger.warning(
                                f"Skipping malformed llm_calls.jsonl line "
                                f"{line_number}: {error}"
                            )
                            continue
                        if isinstance(row, dict):
                            rows.append(row)

        token_keys = (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "reasoning_tokens",
            "cached_tokens",
        )

        def numeric_values(key: str) -> List[float | int]:
            return [
                row[key]
                for row in rows
                if isinstance(row.get(key), (int, float))
                and not isinstance(row.get(key), bool)
            ]

        def token_totals(model_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
            totals: Dict[str, Any] = {}
            for key in token_keys:
                values = [
                    row[key]
                    for row in model_rows
                    if isinstance(row.get(key), (int, float))
                    and not isinstance(row.get(key), bool)
                ]
                totals[key] = sum(values) if values else None
            return totals

        summary: Dict[str, Any] = {
            "llm_call_count": len(rows),
            "llm_call_success_count": sum(row.get("success") is True for row in rows),
            "llm_call_failure_count": sum(row.get("success") is False for row in rows),
        }
        for key in token_keys:
            values = numeric_values(key)
            summary[key] = sum(values) if values else None
        rows_by_model: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            model_identifier = row.get("model_identifier")
            model_key = (
                str(model_identifier)
                if model_identifier not in (None, "")
                else "unknown"
            )
            rows_by_model.setdefault(model_key, []).append(row)

        summary["tokens_by_model"] = {
            model_key: token_totals(model_rows)
            for model_key, model_rows in sorted(rows_by_model.items())
        }

        latencies = numeric_values("latency_s")
        summary["llm_latency_s_total"] = sum(latencies) if latencies else None
        return summary

    def _process_single_instance(
        self,
        i: int,
        instance,
        instance_dir: Optional[str] = None,
    ) -> Dict[str, Union[int, float]]:
        """
        Core worker function to process a single instance.
        This can be reused for both sequential and parallel processing.

        Args:
            i: Index of the instance
            instance: Dataset instance to process
            instance_dir: Optional directory to save outputs

        Returns:
            Tuple of (index, metrics) to maintain order
        """

        context_manager = self._get_context_manager(i)
        instance_identifier = self._get_instance_identifier(instance, i)
        with llm_instance_context(
            benchmark=self.dataset.dataset_id,
            benchmark_instance_identifier=instance_identifier,
            benchmark_instance_index=i,
        ), context_manager as span:
            agent = self.config.get_agent()
            try:
                started_at = time.perf_counter()
                try:
                    output = agent.run_agent(
                        instance,
                        instance_dir=instance_dir,
                        llm_params=self.config.llm.params,
                        instance_idx=i,
                    )
                except Exception as error:
                    execution_time_s = time.perf_counter() - started_at
                    failure_metrics: Dict[str, Union[int, float, str]] = {
                        "resolved": 0,
                        "num_steps": 0,
                        "success": 0,
                        "execution_time_s": execution_time_s,
                    }
                    if instance_dir is not None:
                        write_yaml(
                            InstanceSave(
                                inp=instance.get_prompt_info(),
                                output={
                                    "error_type": type(error).__name__,
                                    "error": str(error),
                                },
                                metrics=failure_metrics,
                                expected_output=instance.expected_output,
                            ).model_dump(),
                            os.path.join(instance_dir, "instance_save.yaml"),
                            indent=True,
                        )
                    raise InstanceExecutionError(error, failure_metrics) from error
                execution_time_s = time.perf_counter() - started_at
                collect_artifacts = getattr(
                    agent, "collect_environment_artifacts", None
                )
                if callable(collect_artifacts):
                    output.environment_artifacts = collect_artifacts()
                inst_metrics = self.dataset.get_instance_eval_metrics(output)
                inst_metrics["execution_time_s"] = execution_time_s
                inst_output = self.dataset.get_instance_eval_output(output)
                if self.log_langfuse and hasattr(span, "update"):
                    span.update(
                        input=instance.get_prompt_info(),
                        output=inst_output,
                    )
                    for name, metric in inst_metrics.items():
                        span.score_trace(
                            name=name,
                            value=metric,
                        )

                if instance_dir is not None:
                    output_save = InstanceSave(
                        inp=instance.get_prompt_info(),
                        output=inst_output,
                        metrics=inst_metrics,  # type: ignore
                        expected_output=instance.expected_output,
                    )
                    write_yaml(
                        output_save.model_dump(),
                        os.path.join(instance_dir, "instance_save.yaml"),
                        indent=True,
                    )

                return inst_metrics
            finally:
                # Cleanup Docker containers — handle both single and multi-agent
                self._cleanup_agent(agent)

    @staticmethod
    def _get_instance_identifier(instance: Any, index: int) -> str:
        for field in ("instance_id", "task_id", "id", "doc_id"):
            value = getattr(instance, field, None)
            if value is not None:
                return str(value)
        return str(index)

    def _cleanup_agent(self, agent) -> None:
        """Clean up Docker containers for any agent type."""
        # Single-agent: agent.env._cleanup()
        if hasattr(agent, "env") and hasattr(agent.env, "_cleanup"):
            try:
                agent.env._cleanup()
            except Exception as e:
                logger.warning(f"Docker cleanup error: {e}")

        # Multi-agent: iterate over subagents
        lead = getattr(agent, "lead_agent", None)
        if lead is not None:
            subagents = getattr(lead, "subagents", {})
            for agent_id, sub in subagents.items():
                env = getattr(sub, "env", None)
                if env is not None and hasattr(env, "_cleanup"):
                    try:
                        env._cleanup()
                    except Exception as e:
                        logger.warning(f"Docker cleanup error for {agent_id}: {e}")

    def run_parallel(self, num_workers: int = 4):
        """
        Run the experiment in parallel using multiprocessing.

        Args:
            num_workers: Maximum number of worker processes
        """
        # Prepare work items
        instances = self._get_instances()

        work_items = []
        for i, instance in enumerate(instances):
            instance_dir = None
            if self.output_dir is not None:
                instance_dir = os.path.join(
                    self.output_dir, "instance_runs", f"{i:04d}"
                )
                os.makedirs(instance_dir, exist_ok=True)

            work_items.append((i, instance, instance_dir))

        # Process in parallel
        metrics: List[Dict[str, int | float] | str] = [""] * len(
            work_items
        )  # Pre-allocate to maintain order

        # Parallel processing with progress bar

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    self._process_single_instance,
                    *work_item,
                ): i
                for i, work_item in enumerate(work_items)
            }
            with tqdm(
                total=len(work_items),
                desc=f"Evaluating {self.dataset.dataset_id} dataset instances (parallel)",
            ) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    i = futures[future]
                    try:
                        result = future.result()
                        metrics[i] = result
                    except Exception as exc:
                        tb_str = "".join(
                            traceback.format_exception(
                                type(exc), exc, exc.__traceback__
                            )
                        )
                        logger.error(
                            f"Error generating output: {exc}\nTraceback:\n{tb_str}"
                        )
                        metrics[i] = (
                            exc.metrics
                            if isinstance(exc, InstanceExecutionError)
                            else {
                                "resolved": 0,
                                "num_steps": 0,
                                "success": 0,
                            }
                        )

                    pbar.update(1)

        return self._finalize_dataset_metrics(metrics)
