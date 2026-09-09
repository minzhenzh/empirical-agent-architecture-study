"""Thread-safe, run-local logging for raw LLM call metrics."""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional


_writer_lock = threading.Lock()
_output_path: Optional[Path] = None
_instance_context: ContextVar[dict[str, Any]] = ContextVar(
    "llm_instance_context", default={}
)


def configure_llm_call_logging(output_dir: Optional[str]) -> None:
    """Configure one JSONL destination for the current Hydra run."""
    global _output_path
    with _writer_lock:
        _output_path = Path(output_dir) / "llm_calls.jsonl" if output_dir else None
        if _output_path is not None:
            _output_path.parent.mkdir(parents=True, exist_ok=True)
            _output_path.write_text("", encoding="utf-8")


@contextmanager
def llm_instance_context(**metadata: Any) -> Iterator[None]:
    """Attach benchmark-instance metadata to calls made in this context."""
    token = _instance_context.set({**_instance_context.get(), **metadata})
    try:
        yield
    finally:
        _instance_context.reset(token)


def write_llm_call(record: Mapping[str, Any]) -> None:
    """Atomically append one JSON object, enriched with instance context."""
    with _writer_lock:
        if _output_path is None:
            return
        payload = {**_instance_context.get(), **dict(record)}
        with _output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, default=str))
            stream.write(os.linesep)
