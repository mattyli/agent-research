"""
HermesNativeRunner: wraps Hermes AIAgent for MedAgentBench native evaluation.

Must be run with the Hermes Python 3.11 interpreter:
    /Users/02matt/.hermes/hermes-agent/venv/bin/python3

The runner:
  1. Registers the fhir-medagent toolset (idempotent, once per process).
  2. Creates a TaskContext and registers it under a unique Hermes task_id.
  3. Instantiates AIAgent with restricted toolset + memory disabled.
  4. Calls run_conversation() and reads ctx.finished / ctx.final_answer.
  5. Teardown clears the task context.
"""
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.native.interface import NativeHarnessResult, NativeHarnessRunner

logger = logging.getLogger(__name__)

_HERMES_PATH = Path.home() / ".hermes" / "hermes-agent"


def _ensure_hermes_on_path() -> None:
    hermes_str = str(_HERMES_PATH)
    if hermes_str not in sys.path:
        sys.path.insert(0, hermes_str)


class HermesNativeRunner(NativeHarnessRunner):
    """
    Hermes-specific adapter for MedAgentBench native evaluation.

    Memory mode 'disabled' is the only mode fully implemented here.
    Phase 6 will add task_local / warmup_frozen / persistent_eval.
    """

    def setup_task(
        self,
        task_metadata: Dict[str, Any],
        task_prompt: str,
        tool_specs: List[Dict],
        model_config: Dict[str, Any],
        runtime_constraints: Dict[str, Any],
        memory_config: Dict[str, Any],
    ) -> None:
        self._task_metadata = task_metadata
        self._task_prompt = task_prompt
        self._model_config = model_config
        self._runtime = runtime_constraints
        self._memory_config = memory_config

        self._hermes_task_id = uuid.uuid4().hex
        self._agent: Optional[Any] = None
        self._artifact_dir: Optional[Path] = None

    def run(self) -> NativeHarnessResult:
        _ensure_hermes_on_path()

        from src.native.fhir_tools import (
            TaskContext,
            clear_task_context,
            register_fhir_toolset,
            set_task_context,
        )

        register_fhir_toolset()

        # Determine artifact directory — driver sets this via task_metadata or we use tmp.
        task_id = self._task_metadata.get("id", "unknown")
        artifact_dir = Path(self._model_config.get("_artifact_dir", f"/tmp/hermes_native/{task_id}"))
        artifact_dir.mkdir(parents=True, exist_ok=True)

        ctx = TaskContext(
            fhir_base_url=self._model_config.get(
                "fhir_base_url", "http://localhost:8080/fhir/"
            ),
            max_calls=self._runtime.get("max_fhir_calls", 8),
            output_dir=artifact_dir,
            hermes_task_id=self._hermes_task_id,
        )
        set_task_context(self._hermes_task_id, ctx)

        start = time.time()
        messages: List[Dict] = []
        errors: List[str] = []

        try:
            agent = self._build_agent(ctx)
            self._agent = agent

            conv_result = agent.run_conversation(
                user_message=self._task_prompt,
                task_id=self._hermes_task_id,
            )
            messages = conv_result.get("messages", [])

            if not conv_result.get("completed", True) and conv_result.get("error"):
                errors.append(f"hermes: {conv_result['error']}")

        except Exception as exc:
            errors.append(f"run_conversation raised: {exc}")
            logger.exception("HermesNativeRunner: run_conversation failed for %s", task_id)

        latency = time.time() - start
        ctx_after = self._agent_task_context()

        final_answer, parse_status = self._extract_answer(ctx_after)
        transcript = _build_transcript(messages)
        token_usage = self._token_usage()

        return NativeHarnessResult(
            final_answer=final_answer,
            answer_parse_status=parse_status,
            transcript=transcript,
            tool_call_log=ctx_after.tool_call_log if ctx_after else [],
            raw_harness_logs="",
            fhir_call_count=ctx_after.call_count if ctx_after else 0,
            token_usage=token_usage,
            latency_seconds=latency,
            timed_out=False,
            memory_mode=self._memory_config.get("mode", "disabled"),
            errors=errors,
            artifact_dir=str(artifact_dir),
        )

    def teardown(self) -> None:
        from src.native.fhir_tools import clear_task_context
        clear_task_context(self._hermes_task_id)
        self._agent = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_agent(self, ctx: "TaskContext") -> Any:
        from run_agent import AIAgent

        memory_mode = self._memory_config.get("mode", "disabled")
        skip_memory = memory_mode == "disabled"

        kwargs = dict(
            max_iterations=self._runtime.get("max_iterations", 12),
            enabled_toolsets=["fhir-medagent"],
            skip_context_files=True,
            skip_memory=skip_memory,
            quiet_mode=True,
            tool_delay=0.0,
            max_tokens=self._model_config.get("max_tokens", 4096),
            request_overrides={
                "temperature": self._model_config.get("temperature", 0.0),
            },
        )

        model_name = self._model_config.get("model_name") or ""
        if model_name:
            kwargs["model"] = model_name

        api_key = self._model_config.get("api_key") or ""
        base_url = self._model_config.get("base_url") or ""
        provider = (self._model_config.get("provider") or "").strip().lower()

        # "openai" is not in Hermes' PROVIDER_REGISTRY, so we must resolve
        # credentials ourselves.  run_agent.py loads ~/.hermes/.env at import
        # time, so OPENAI_API_KEY is already in os.environ here.
        if provider == "openai" and not api_key and not base_url:
            import os
            api_key = os.environ.get("OPENAI_API_KEY", "")
            base_url = "https://api.openai.com/v1"
            # Force chat completions — Hermes auto-upgrades api.openai.com to
            # the Responses API, which rejects non-GPT-5 models with
            # "Encrypted content is not supported with this model."
            kwargs["api_mode"] = "chat_completions"

        # Passing both api_key and base_url bypasses resolve_provider_client
        # entirely; AIAgent constructs the OpenAI client directly.
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        if provider and provider != "openai":
            kwargs["provider"] = provider

        return AIAgent(**kwargs)

    def _agent_task_context(self) -> Optional["TaskContext"]:
        from src.native.fhir_tools import get_task_context
        return get_task_context(self._hermes_task_id)

    def _extract_answer(
        self, ctx: Optional["TaskContext"]
    ) -> tuple:
        if ctx is not None and ctx.finished and ctx.final_answer is not None:
            # Validate that final_answer is a JSON-loadable list.
            try:
                parsed = json.loads(ctx.final_answer)
                if isinstance(parsed, list):
                    return ctx.final_answer, "ok"
                return ctx.final_answer, "malformed"
            except (json.JSONDecodeError, ValueError):
                return ctx.final_answer, "malformed"
        return None, "missing"

    def _token_usage(self) -> Optional[Dict]:
        if self._agent is None:
            return None
        return {
            "input": getattr(self._agent, "session_input_tokens", 0),
            "output": getattr(self._agent, "session_output_tokens", 0),
        }


def _build_transcript(messages: List[Dict]) -> List[Dict]:
    """
    Convert Hermes OpenAI-format messages to a flat [{role, content}] list.
    Tool role messages are included for completeness (scored separately).
    """
    transcript = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""

        if role == "assistant":
            # May have tool_calls; serialise them into content for reference.
            tool_calls = msg.get("tool_calls")
            if tool_calls and not content:
                content = json.dumps(tool_calls)
            transcript.append({"role": "assistant", "content": content})
        elif role == "user":
            transcript.append({"role": "user", "content": content})
        elif role == "tool":
            transcript.append({
                "role": "tool",
                "name": msg.get("name", ""),
                "tool_call_id": msg.get("tool_call_id", ""),
                "content": content,
            })
    return transcript
