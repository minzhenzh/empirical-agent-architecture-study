import os
import time
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, cast

import litellm
try:
    import langfuse
except ImportError:  # Langfuse is optional for local experiment runs.
    langfuse = None  # type: ignore[assignment]

# Auto-drop unsupported params (e.g., temperature=0 for gpt-5 models)
litellm.drop_params = True
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_litellm.chat_models.litellm import ChatLiteLLM
from litellm.exceptions import APIConnectionError, AuthenticationError, RateLimitError, ServiceUnavailableError
from litellm.types.utils import ModelResponse
from loguru import logger as _logger
from pydantic import Field

from agent_scaling.llm.call_logging import write_llm_call


def _is_quota_exhaustion(error: Exception) -> bool:
    """Check if a RateLimitError is a permanent quota exhaustion (not transient throttling).

    Quota errors should not be retried — they persist until billing is updated.
    """
    err_str = str(error).lower()
    return any(phrase in err_str for phrase in [
        "exceeded your current quota",
        "insufficient_quota",
        "billing hard limit",
        "account is not active",
        "credit balance is too low",
        "credits have been exhausted",
    ])


# Global concurrency limiter to prevent API rate exhaustion
# in multi-agent setups (e.g., hybrid with concurrent sub-agents)
_api_semaphore = threading.Semaphore(3)  # Max 3 concurrent API calls


class _GeminiKeyRotator:
    """Thread-safe round-robin API key rotator for Gemini.

    Lazy-loads keys on first use (after dotenv has been loaded).
    """

    def __init__(self):
        self._keys: List[str] = []
        self._idx = 0
        self._lock = threading.Lock()
        self._loaded = False

    def _load_keys(self):
        if self._loaded:
            return
        self._loaded = True
        base = os.environ.get("GEMINI_API_KEY", "")
        if base:
            self._keys.append(base)
        for i in range(1, 20):
            k = os.environ.get(f"GEMINI_API_KEY_{i}", "")
            if k:
                self._keys.append(k)
        _logger.info(f"Loaded {len(self._keys)} Gemini API keys for rotation")

    def next_key(self) -> Optional[str]:
        self._load_keys()
        if not self._keys:
            return None
        with self._lock:
            key = self._keys[self._idx % len(self._keys)]
            self._idx += 1
            return key

    def remove_key(self, key: str) -> None:
        """Remove an invalid key from rotation."""
        with self._lock:
            if key in self._keys:
                self._keys.remove(key)
                _logger.info(f"Removed invalid key, {len(self._keys)} remaining")

    @property
    def num_keys(self) -> int:
        self._load_keys()
        return len(self._keys)


_gemini_rotator = _GeminiKeyRotator()


class ChatLiteLLMLC(ChatLiteLLM):
    log_langfuse: bool = False
    agent_identifier: str = Field(default="agent_1", exclude=True)
    agent_role: str = Field(default="agent", exclude=True)
    architecture: str = Field(default="single-agent", exclude=True)

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        res: ChatResult = super()._create_chat_result(response)
        if res.llm_output is None:
            res.llm_output = {}
        res.llm_output["litellm_response"] = response
        return res

    def invoke(self, *args, **kwargs) -> AIMessage:
        """Invoke the model and persist one raw, attributable call-metric row."""
        started = time.perf_counter()
        try:
            response = self._invoke_with_retries(*args, **kwargs)
        except Exception as error:
            self._record_call(
                response=None,
                latency_s=time.perf_counter() - started,
                error=error,
            )
            raise
        self._record_call(
            response=response,
            latency_s=time.perf_counter() - started,
            error=None,
        )
        return response

    def _invoke_with_retries(self, *args, **kwargs) -> AIMessage:
        # Retry on IndexError from empty model responses (thinking models)
        # and on rate limit / transient errors that bubble up past _generate
        max_retries = 5
        for attempt in range(max_retries):
            try:
                return cast(AIMessage, super().invoke(*args, **kwargs))
            except IndexError:
                if attempt < max_retries - 1:
                    wait = 3 * (attempt + 1)
                    _logger.warning(
                        f"Empty model response (attempt {attempt+1}/{max_retries}), "
                        f"retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    _logger.error(
                        f"Empty model response after {max_retries} attempts, "
                        "returning fallback message"
                    )
                    return AIMessage(
                        content="[Model returned empty response]",
                        response_metadata={
                            "call_error": {
                                "type": "EmptyModelResponse",
                                "message": f"empty response after {max_retries} attempts",
                            }
                        },
                    )
            except APIConnectionError as e:
                # Don't retry here — _generate already retried 8 times.
                # Retrying in invoke creates zombie threads that hold the
                # API semaphore, causing deadlocks in multi-agent systems.
                _logger.error(
                    f"Connection error in invoke after _generate exhausted retries: "
                    f"{type(e).__name__}, returning fallback to avoid semaphore deadlock"
                )
                return AIMessage(
                    content="[API connection failed — submit your current work]",
                    response_metadata={
                        "call_error": {
                            "type": type(e).__name__,
                            "message": str(e),
                        }
                    },
                )
            except (RateLimitError, ServiceUnavailableError) as e:
                if isinstance(e, RateLimitError) and _is_quota_exhaustion(e):
                    _logger.error(f"Quota exhausted (not retryable): {e}")
                    raise
                if attempt < max_retries - 1:
                    wait = min(10 * (2 ** attempt), 120)
                    _logger.warning(
                        f"Transient error in invoke (attempt {attempt+1}/{max_retries}): "
                        f"{type(e).__name__}, retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    raise
            except Exception as e:
                if _is_quota_exhaustion(e):
                    _logger.error(f"Quota exhausted (not retryable): {e}")
                    raise
                if (
                    "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                ) and attempt < max_retries - 1:
                    wait = min(10 * (2**attempt), 120)
                    _logger.warning(
                        f"Rate limit in invoke (attempt {attempt+1}/{max_retries}), "
                        f"retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    raise

    def _record_call(
        self,
        *,
        response: Optional[AIMessage],
        latency_s: float,
        error: Optional[Exception],
    ) -> None:
        usage: Dict[str, Any] = {}
        response_metadata: Dict[str, Any] = {}
        if response is not None:
            usage = dict(response.usage_metadata or {})
            response_metadata = dict(response.response_metadata or {})
            token_usage = response_metadata.get("token_usage") or response_metadata.get(
                "usage"
            )
            if isinstance(token_usage, Mapping):
                usage = {**dict(token_usage), **usage}

        input_details = usage.get("input_token_details") or usage.get(
            "prompt_tokens_details"
        ) or {}
        output_details = usage.get("output_token_details") or usage.get(
            "completion_tokens_details"
        ) or {}
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
        total_tokens = usage.get("total_tokens")
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens

        provider = response_metadata.get("provider")
        if provider is None and self.model:
            provider = self.model.split("/", 1)[0] if "/" in self.model else None

        call_error = response_metadata.get("call_error")
        error_record = (
            {"type": type(error).__name__, "message": str(error)}
            if error is not None
            else call_error
        )
        write_llm_call(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "architecture": self.architecture,
                "agent_identifier": self.agent_identifier,
                "agent_role": self.agent_role,
                "model_identifier": response_metadata.get("model_name") or self.model,
                "provider": provider,
                "backend_metadata": {
                    key: response_metadata[key]
                    for key in ("id", "system_fingerprint", "finish_reason")
                    if response_metadata.get(key) is not None
                },
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "reasoning_tokens": output_details.get("reasoning")
                or output_details.get("reasoning_tokens")
                or usage.get("reasoning_tokens"),
                "cached_tokens": input_details.get("cache_read")
                or input_details.get("cached_tokens")
                or usage.get("cached_tokens"),
                "latency_s": latency_s,
                "success": error_record is None,
                "retry_metadata": {
                    "outer_max_attempts": 5,
                    "langchain_max_retries": self.max_retries,
                },
                "error": error_record,
            }
        )
    def _log_langfuse(
        self,
        message_dicts: List[Dict[str, Any]],
        params: Dict[str, Any],
        response: ModelResponse,
    ) -> None:
        if langfuse is None:
            raise RuntimeError("Langfuse logging requested, but langfuse is not installed")
        client = langfuse.Langfuse()  # type: ignore
        model_params = {
            k: v
            for k, v in params.items()
            if k not in ["model", "stream"] and v is not None
        }

        gen_context = client.start_generation(
            name=f"call {params.get('model')}"
            + (" (from cache)" if response._hidden_params.get("cache_hit", "") else ""),
            input=message_dicts,
            model=params.get("model"),
            model_parameters=model_params,
        )

        gen_context.update(
            output=response.choices[0].message,  # type: ignore
            metadata=response.model_dump(),
            usage_details=response.usage if hasattr(response, "usage") else None,  # type: ignore
        )
        gen_context.end()
        client.flush()

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        log_langfuse = kwargs.pop("log_langfuse", None)
        if log_langfuse is None:
            log_langfuse = self.log_langfuse

        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}

        # Ensure a request timeout to prevent indefinite hangs
        # Scale timeout with message size — large multi-agent contexts need more time
        if "timeout" not in params or params["timeout"] is None:
            total_chars = sum(len(str(m.get("content", ""))) for m in message_dicts)
            if total_chars > 50000:
                params["timeout"] = 600  # 10 min for large contexts (multi-agent)
            else:
                params["timeout"] = 300  # 5 min for normal requests

        # Key rotation with retry on rate limit for Gemini models
        is_gemini = "gemini" in (self.model or "").lower()
        num_keys = _gemini_rotator.num_keys if is_gemini else 0
        # More aggressive retry: cycle through all keys multiple times
        max_attempts = max(num_keys * 3, 5) if is_gemini else 3
        # Connection errors are usually persistent (request too large, timeout)
        # Keep well under 300s round timeout to avoid zombie threads holding semaphore
        max_connection_attempts = min(max_attempts, 5)
        connection_error_count = 0
        last_err = None

        for attempt in range(max(max_attempts, 1)):
            try:
                if is_gemini and num_keys > 0:
                    params["api_key"] = _gemini_rotator.next_key()
                with _api_semaphore:
                    response = self.completion_with_retry(
                        messages=message_dicts, run_manager=run_manager, **params
                    )
                if log_langfuse:
                    self._log_langfuse(message_dicts, params, response)
                return self._create_chat_result(response)
            except AuthenticationError as e:
                # Invalid key — remove from rotation and retry immediately
                bad_key = params.get("api_key")
                if bad_key:
                    _gemini_rotator.remove_key(bad_key)
                num_keys = _gemini_rotator.num_keys
                if num_keys == 0:
                    raise
                _logger.warning(
                    f"Auth error (attempt {attempt+1}/{max_attempts}), "
                    f"removing bad key and retrying..."
                )
                last_err = e
            except RateLimitError as e:
                if _is_quota_exhaustion(e):
                    _logger.error(f"Quota exhausted (not retryable): {e}")
                    raise
                last_err = e
                # Exponential backoff with jitter: 5s, 10s, 20s, 40s... up to 120s
                import random as _rnd
                base_wait = min(5 * (2 ** (attempt // max(num_keys, 1))), 120)
                wait = base_wait + _rnd.uniform(0, 5)
                _logger.warning(
                    f"Rate limit hit (attempt {attempt+1}/{max_attempts}), "
                    f"rotating key and waiting {wait:.1f}s..."
                )
                time.sleep(wait)
            except ServiceUnavailableError as e:
                last_err = e
                wait = min(10 * (attempt + 1), 60)
                _logger.warning(
                    f"Service unavailable (attempt {attempt+1}/{max_attempts}), "
                    f"rotating key and waiting {wait}s..."
                )
                time.sleep(wait)
            except APIConnectionError as e:
                last_err = e
                connection_error_count += 1
                if connection_error_count >= max_connection_attempts:
                    _logger.error(
                        f"API connection error persisted after {connection_error_count} attempts, "
                        f"giving up on this call (likely request too large or timeout)"
                    )
                    break  # Exit retry loop — will raise last_err below
                wait = min(10 * (connection_error_count), 15)  # Max 15s wait, total ~50s
                _logger.warning(
                    f"API connection error (attempt {connection_error_count}/{max_connection_attempts}), "
                    f"rotating key and waiting {wait}s..."
                )
                time.sleep(wait)
            except Exception as e:
                # Check if wrapped exception contains rate limit
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    last_err = e
                    import random as _rnd
                    base_wait = min(5 * (2 ** (attempt // max(num_keys, 1))), 120)
                    wait = base_wait + _rnd.uniform(0, 5)
                    _logger.warning(
                        f"Rate limit (wrapped) hit (attempt {attempt+1}/{max_attempts}), "
                        f"rotating key and waiting {wait:.1f}s..."
                    )
                    time.sleep(wait)
                elif "API_KEY_INVALID" in str(e) or "API key not valid" in str(e):
                    bad_key = params.get("api_key")
                    if bad_key:
                        _gemini_rotator.remove_key(bad_key)
                    num_keys = _gemini_rotator.num_keys
                    if num_keys == 0:
                        raise
                    last_err = e
                else:
                    raise

        raise last_err  # type: ignore
