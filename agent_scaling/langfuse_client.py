from typing import Any

try:
    from langfuse import get_client
except ImportError:  # Langfuse is an optional dependency.
    get_client = None  # type: ignore[assignment]
from agent_scaling.logger import logger


def get_lf_client() -> Any | None:
    if get_client is None:
        return None
    try:
        client = get_client()
        return client
    except Exception as e:
        logger.warning(
            f"Error getting Langfuse client:\n{e}\nProceeding without Langfuse client."
        )
        return None
