"""Safe bridge to the pinned upstream WorkBench tool implementation.

The upstream project stores each mock database in module-level pandas
DataFrames and evaluates model-produced Python call strings with ``eval``.
This bridge keeps the upstream tool behavior while providing two safeguards:

* every environment gets an isolated copy of the mock database state; and
* model output is parsed with ``ast`` and dispatched through a whitelist.

The upstream repository is downloaded by ``scripts/setup_workbench.sh``.
"""

from __future__ import annotations

import ast
import importlib
import os
import re
import sys
import threading
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping


UPSTREAM_ENV_VAR = "WORKBENCH_ROOT"
DEFAULT_UPSTREAM_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "WorkBench"

DOMAIN_ALIASES = {
    "analytics": "analytics",
    "calendar": "calendar",
    "crm": "customer_relationship_manager",
    "customer_relationship_manager": "customer_relationship_manager",
    "email": "email",
    "project": "project_management",
    "project_management": "project_management",
}

TOOL_PREFIX_ALIASES = {
    "analytics": "analytics",
    "calendar": "calendar",
    "company_directory": "directory",
    "customer_relationship_manager": "crm",
    "email": "email",
    "project_management": "project",
}

ACTION_BLOCK_RE = re.compile(
    r"<workbench_actions>\s*(.*?)\s*</workbench_actions>", re.DOTALL
)


def upstream_root() -> Path:
    return Path(os.environ.get(UPSTREAM_ENV_VAR, DEFAULT_UPSTREAM_ROOT)).expanduser().resolve()


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _copy_value(value: Any) -> Any:
    if hasattr(value, "copy"):
        try:
            return value.copy(deep=True)
        except TypeError:
            return value.copy()
    return deepcopy(value)


class WorkbenchBackend:
    """Loads upstream tools and provides isolated, thread-safe execution."""

    _STATE_FIELDS = {
        "calendar": ("CALENDAR_EVENTS",),
        "email": ("EMAILS",),
        "analytics": ("ANALYTICS_DATA", "PLOTS_DATA"),
        "project_management": ("PROJECT_TASKS",),
        "customer_relationship_manager": ("CRM_DATA",),
    }
    _GRADED_STATE_FIELDS = {
        "calendar": ("CALENDAR_EVENTS",),
        "email": ("EMAILS",),
        "analytics": ("PLOTS_DATA",),
        "project_management": ("PROJECT_TASKS",),
        "customer_relationship_manager": ("CRM_DATA",),
    }

    def __init__(self, root: Path | None = None):
        self.root = (root or upstream_root()).resolve()
        if not (self.root / "src" / "tools" / "toolkits.py").is_file():
            raise RuntimeError(
                "Upstream WorkBench is not installed at "
                f"{self.root}. Run `bash scripts/setup_workbench.sh` first, "
                f"or set {UPSTREAM_ENV_VAR} to the checkout path."
            )

        self._lock = threading.RLock()
        root_str = str(self.root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        with self._lock, _working_directory(self.root):
            toolkits = importlib.import_module("src.tools.toolkits")

        self.modules = {
            name: importlib.import_module(f"src.tools.{name}")
            for name in (
                "analytics",
                "calendar",
                "company_directory",
                "customer_relationship_manager",
                "email",
                "project_management",
            )
        }
        self.tools = {tool.name: tool for tool in toolkits.all_tools}
        self.side_effect_tool_names = {
            tool.name for tool in toolkits.tools_with_side_effects
        }
        self.alias_to_canonical = {
            self.alias_for(name): name for name in self.tools
        }
        self._baseline_state = self._snapshot_modules()

    @staticmethod
    def alias_for(canonical_name: str) -> str:
        domain, tool_name = canonical_name.split(".", 1)
        return f"{TOOL_PREFIX_ALIASES[domain]}_{tool_name}"

    def canonical_name(self, name: str) -> str | None:
        name = name.removesuffix(".func")
        if name in self.tools:
            return name
        return self.alias_to_canonical.get(name)

    def tools_for_domains(self, domains: Iterable[str]) -> list[str]:
        normalized = {
            DOMAIN_ALIASES[domain]
            for domain in domains
            if domain in DOMAIN_ALIASES
        }
        names = [
            name for name in self.tools if name.split(".", 1)[0] in normalized
        ]
        directory_name = "company_directory.find_email_address"
        if directory_name in self.tools:
            names.append(directory_name)
        return list(dict.fromkeys(names))

    def _snapshot_modules(self) -> dict[tuple[str, str], Any]:
        return {
            (module_name, field): _copy_value(getattr(self.modules[module_name], field))
            for module_name, fields in self._STATE_FIELDS.items()
            for field in fields
        }

    def _restore_modules(self, state: Mapping[tuple[str, str], Any]) -> None:
        for (module_name, field), value in state.items():
            setattr(self.modules[module_name], field, _copy_value(value))

    def new_state(self) -> dict[tuple[str, str], Any]:
        return {key: _copy_value(value) for key, value in self._baseline_state.items()}

    def invoke(
        self,
        canonical_name: str,
        arguments: Mapping[str, Any],
        state: dict[tuple[str, str], Any],
    ) -> Any:
        """Invoke one whitelisted upstream tool against an isolated state."""
        if canonical_name not in self.tools:
            raise ValueError(f"Unknown WorkBench tool: {canonical_name}")
        with self._lock:
            self._restore_modules(state)
            try:
                result = self.tools[canonical_name].invoke(dict(arguments))
                state.clear()
                state.update(self._snapshot_modules())
                return result
            finally:
                self._restore_modules(self._baseline_state)

    def parse_actions(self, text: str) -> list[dict[str, Any]]:
        """Parse canonical actions from a submit block or plain model text."""
        blocks = list(ACTION_BLOCK_RE.finditer(text or ""))
        if blocks:
            try:
                import json

                actions = []
                for block in blocks:
                    payload = json.loads(block.group(1))
                    for item in payload:
                        canonical = self.canonical_name(str(item.get("tool", "")))
                        args = item.get("args", {})
                        if canonical and isinstance(args, dict):
                            actions.append({"tool": canonical, "args": args})
                return actions
            except (TypeError, ValueError):
                pass

        return self._parse_calls_from_text(text or "")

    def _parse_calls_from_text(self, text: str) -> list[dict[str, Any]]:
        names = sorted(
            [*self.tools, *self.alias_to_canonical], key=len, reverse=True
        )
        name_pattern = "|".join(re.escape(name) for name in names)
        starts = re.finditer(
            rf"(?<![\w.])(?P<name>{name_pattern})(?:\.func)?\s*\(", text
        )
        actions: list[dict[str, Any]] = []
        for match in starts:
            end = self._find_call_end(text, match.end() - 1)
            if end is None:
                continue
            call_text = text[match.start() : end]
            action = self._parse_one_call(call_text)
            if action is not None:
                actions.append(action)
        return actions

    @staticmethod
    def _find_call_end(text: str, opening_index: int) -> int | None:
        depth = 0
        quote: str | None = None
        escaped = False
        for index in range(opening_index, len(text)):
            char = text[index]
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                continue
            if char in {"'", '"'}:
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return index + 1
        return None

    def _parse_one_call(self, call_text: str) -> dict[str, Any] | None:
        try:
            expression = ast.parse(call_text.strip(), mode="eval").body
        except SyntaxError:
            return None
        if not isinstance(expression, ast.Call):
            return None
        dotted_name = self._ast_dotted_name(expression.func)
        if dotted_name is None:
            return None
        canonical = self.canonical_name(dotted_name)
        if canonical is None:
            return None

        schema = self.tools[canonical].args_schema.model_json_schema()
        parameter_names = list(schema.get("properties", {}))
        if len(expression.args) > len(parameter_names):
            return None
        arguments: dict[str, Any] = {}
        try:
            for name, value in zip(parameter_names, expression.args):
                arguments[name] = ast.literal_eval(value)
            for keyword in expression.keywords:
                if keyword.arg is None:
                    return None
                arguments[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError):
            return None
        return {"tool": canonical, "args": arguments}

    @staticmethod
    def _ast_dotted_name(node: ast.AST) -> str | None:
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        parts.append(node.id)
        return ".".join(reversed(parts))

    def is_exact_match(
        self,
        predicted: Iterable[Mapping[str, Any]],
        expected: Iterable[Mapping[str, Any]],
    ) -> bool:
        """Match upstream exact-match semantics for state-changing actions."""
        predicted_side_effects = [
            action
            for action in predicted
            if action.get("tool") in self.side_effect_tool_names
        ]
        return sorted(map(self._normalized_action, predicted_side_effects)) == sorted(
            map(self._normalized_action, expected)
        )

    @staticmethod
    def _normalized_action(action: Mapping[str, Any]) -> str:
        import json

        normalized = {
            "tool": str(action.get("tool", "")).lower(),
            "args": WorkbenchBackend._lower_strings(dict(action.get("args", {}))),
        }
        return json.dumps(normalized, sort_keys=True, default=str)

    @staticmethod
    def _lower_strings(value: Any) -> Any:
        if isinstance(value, str):
            return value.lower()
        if isinstance(value, dict):
            return {key: WorkbenchBackend._lower_strings(item) for key, item in value.items()}
        if isinstance(value, list):
            return [WorkbenchBackend._lower_strings(item) for item in value]
        return value

    def is_correct(
        self,
        predicted: Iterable[Mapping[str, Any]],
        expected: Iterable[Mapping[str, Any]],
    ) -> bool:
        """Compare database states after safely executing both action lists."""
        predicted_state = self._execute_actions(predicted)
        expected_state = self._execute_actions(expected)
        for module_name, fields in self._GRADED_STATE_FIELDS.items():
            for field in fields:
                key = (module_name, field)
                if not self._frames_equal(predicted_state[key], expected_state[key]):
                    return False
        return True

    def _execute_actions(
        self, actions: Iterable[Mapping[str, Any]]
    ) -> dict[tuple[str, str], Any]:
        state = self.new_state()
        for action in actions:
            canonical = self.canonical_name(str(action.get("tool", "")))
            arguments = action.get("args", {})
            if canonical is None or not isinstance(arguments, Mapping):
                continue
            try:
                self.invoke(canonical, arguments, state)
            except Exception:
                # Upstream ``execute_actions_and_reset_state`` also skips calls
                # that fail during execution.
                continue
        return state

    @staticmethod
    def _frames_equal(left: Any, right: Any) -> bool:
        fields_not_to_lower = {"status", "list_name", "board"}
        left = _copy_value(left)
        right = _copy_value(right)
        for frame in (left, right):
            for column in frame.columns:
                if column not in fields_not_to_lower:
                    frame[column] = frame[column].map(
                        lambda value: value.lower() if isinstance(value, str) else value
                    )
        return bool(left.equals(right))


_BACKEND: WorkbenchBackend | None = None
_BACKEND_LOCK = threading.Lock()


def get_workbench_backend() -> WorkbenchBackend:
    global _BACKEND
    if _BACKEND is None:
        with _BACKEND_LOCK:
            if _BACKEND is None:
                _BACKEND = WorkbenchBackend()
    return _BACKEND
