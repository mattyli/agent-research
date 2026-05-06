"""
Tests for HermesNativeRunner (Phase 3).

Unit tests (no Hermes import):
    /opt/anaconda3/envs/medagentbench/bin/python3 -m pytest \
        src/native/tests/test_hermes_runner.py -v -m "not integration"

Integration tests (require Hermes + FHIR server):
    /Users/02matt/.hermes/hermes-agent/venv/bin/python3 -m pytest \
        src/native/tests/test_hermes_runner.py -v -m integration
"""
import json
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import requests

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.native.interface import NativeHarnessResult

_HERMES_AGENT_PATH = Path.home() / ".hermes" / "hermes-agent"
_HERMES_AVAILABLE = _HERMES_AGENT_PATH.exists()

_FHIR_BASE = "http://localhost:8080/fhir/"


def _fhir_is_up() -> bool:
    try:
        r = requests.get(_FHIR_BASE + "Patient?_count=1&_format=json", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


_FHIR_UP = _fhir_is_up()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _default_model_config(fhir_base_url: str = _FHIR_BASE) -> dict:
    return {
        "model_name": "anthropic/claude-haiku-4-5",
        "api_key": None,
        "base_url": None,
        "temperature": 0.0,
        "max_tokens": 2048,
        "fhir_base_url": fhir_base_url,
    }


def _default_runtime() -> dict:
    return {
        "max_fhir_calls": 4,
        "max_iterations": 6,
        "timeout_seconds": 60,
        "seed": 42,
    }


def _default_memory() -> dict:
    return {"mode": "disabled", "memory_store_path": None, "warmup_task_ids": None}


def _make_case(task_id: str = "task1_1") -> dict:
    return {
        "id": task_id,
        "instruction": "What is the MRN of patient Peter Stafford with DOB 1932-12-29?",
        "context": "EHR system",
        "sol": ["S1234567"],
        "eval_MRN": "S1234567",
    }


# ---------------------------------------------------------------------------
# Unit tests — no Hermes import
# ---------------------------------------------------------------------------

class TestHermesRunnerSetup:
    def _make_runner(self):
        from src.native.hermes.runner import HermesNativeRunner
        return HermesNativeRunner()

    def test_setup_stores_config(self):
        runner = self._make_runner()
        runner.setup_task(
            task_metadata=_make_case(),
            task_prompt="Do the task.",
            tool_specs=[],
            model_config=_default_model_config(),
            runtime_constraints=_default_runtime(),
            memory_config=_default_memory(),
        )
        assert runner._task_metadata["id"] == "task1_1"
        assert runner._task_prompt == "Do the task."
        assert runner._runtime["max_fhir_calls"] == 4
        assert runner._memory_config["mode"] == "disabled"

    def test_setup_generates_unique_task_id(self):
        from src.native.hermes.runner import HermesNativeRunner

        ids = set()
        for _ in range(5):
            runner = HermesNativeRunner()
            runner.setup_task(
                task_metadata=_make_case(),
                task_prompt="Do the task.",
                tool_specs=[],
                model_config=_default_model_config(),
                runtime_constraints=_default_runtime(),
                memory_config=_default_memory(),
            )
            ids.add(runner._hermes_task_id)
        assert len(ids) == 5  # all unique

    def test_teardown_clears_context(self):
        from src.native.hermes.runner import HermesNativeRunner
        from src.native.fhir_tools import TaskContext, set_task_context, get_task_context

        runner = HermesNativeRunner()
        runner.setup_task(
            task_metadata=_make_case(),
            task_prompt="Do the task.",
            tool_specs=[],
            model_config=_default_model_config(),
            runtime_constraints=_default_runtime(),
            memory_config=_default_memory(),
        )
        task_id = runner._hermes_task_id

        # Manually register a context (as run() would do).
        ctx = TaskContext(
            fhir_base_url=_FHIR_BASE,
            max_calls=8,
            output_dir=Path("/tmp"),
            hermes_task_id=task_id,
        )
        set_task_context(task_id, ctx)
        assert get_task_context(task_id) is not None

        runner.teardown()
        assert get_task_context(task_id) is None


class TestTranscriptBuilding:
    def test_assistant_role_mapped(self):
        from src.native.hermes.runner import _build_transcript

        messages = [{"role": "assistant", "content": "Hello"}]
        transcript = _build_transcript(messages)
        assert transcript == [{"role": "assistant", "content": "Hello"}]

    def test_user_role_preserved(self):
        from src.native.hermes.runner import _build_transcript

        messages = [{"role": "user", "content": "Query"}]
        transcript = _build_transcript(messages)
        assert transcript[0]["role"] == "user"
        assert transcript[0]["content"] == "Query"

    def test_tool_role_included(self):
        from src.native.hermes.runner import _build_transcript

        messages = [
            {"role": "tool", "name": "fhir_patient_search",
             "tool_call_id": "tc1", "content": '{"total": 1}'},
        ]
        transcript = _build_transcript(messages)
        assert len(transcript) == 1
        assert transcript[0]["role"] == "tool"
        assert transcript[0]["name"] == "fhir_patient_search"

    def test_assistant_tool_calls_serialised_when_no_content(self):
        from src.native.hermes.runner import _build_transcript

        tool_calls = [{"id": "tc1", "function": {"name": "fhir_patient_search", "arguments": "{}"}}]
        messages = [{"role": "assistant", "content": None, "tool_calls": tool_calls}]
        transcript = _build_transcript(messages)
        assert json.loads(transcript[0]["content"]) == tool_calls

    def test_empty_messages(self):
        from src.native.hermes.runner import _build_transcript

        assert _build_transcript([]) == []


class TestAnswerExtraction:
    def _runner_with_setup(self):
        from src.native.hermes.runner import HermesNativeRunner

        runner = HermesNativeRunner()
        runner.setup_task(
            task_metadata=_make_case(),
            task_prompt="Do the task.",
            tool_specs=[],
            model_config=_default_model_config(),
            runtime_constraints=_default_runtime(),
            memory_config=_default_memory(),
        )
        return runner

    def test_finished_ctx_returns_ok(self):
        from src.native.fhir_tools import TaskContext

        runner = self._runner_with_setup()
        ctx = TaskContext(
            fhir_base_url=_FHIR_BASE,
            max_calls=8,
            output_dir=Path("/tmp"),
            hermes_task_id=runner._hermes_task_id,
        )
        ctx.finished = True
        ctx.final_answer = '["S1234567"]'

        answer, status = runner._extract_answer(ctx)
        assert answer == '["S1234567"]'
        assert status == "ok"

    def test_non_list_final_answer_is_malformed(self):
        from src.native.fhir_tools import TaskContext

        runner = self._runner_with_setup()
        ctx = TaskContext(
            fhir_base_url=_FHIR_BASE,
            max_calls=8,
            output_dir=Path("/tmp"),
            hermes_task_id=runner._hermes_task_id,
        )
        ctx.finished = True
        ctx.final_answer = '"just a string"'  # not a list

        answer, status = runner._extract_answer(ctx)
        assert status == "malformed"

    def test_not_finished_returns_missing(self):
        from src.native.fhir_tools import TaskContext

        runner = self._runner_with_setup()
        ctx = TaskContext(
            fhir_base_url=_FHIR_BASE,
            max_calls=8,
            output_dir=Path("/tmp"),
            hermes_task_id=runner._hermes_task_id,
        )
        # ctx.finished is False by default

        answer, status = runner._extract_answer(ctx)
        assert answer is None
        assert status == "missing"

    def test_none_ctx_returns_missing(self):
        runner = self._runner_with_setup()
        answer, status = runner._extract_answer(None)
        assert answer is None
        assert status == "missing"


# ---------------------------------------------------------------------------
# Mocked run() tests — avoids real AIAgent, verifies integration logic
# Requires Hermes Python 3.11 (runner.run() imports from fhir_tools → Hermes)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestMockedRunnerRun:
    def _make_mock_agent(self, finished: bool = True, fhir_calls: int = 2):
        """Return a mock AIAgent that simulates a completed conversation."""
        agent = MagicMock()
        agent.session_input_tokens = 500
        agent.session_output_tokens = 100

        def side_effect_run_conversation(user_message, task_id=None):
            from src.native.fhir_tools import get_task_context, set_task_context
            ctx = get_task_context(task_id) if task_id else None
            if ctx and finished:
                ctx.finished = True
                ctx.final_answer = '["S1234567"]'
                ctx.call_count = fhir_calls
            return {
                "final_response": "Done" if finished else None,
                "messages": [
                    {"role": "user", "content": "Do the task."},
                    {"role": "assistant", "content": ""},
                    {"role": "tool", "name": "fhir_patient_search",
                     "tool_call_id": "tc1", "content": '{}'},
                    {"role": "assistant", "content": ""},
                    {"role": "tool", "name": "fhir_finish",
                     "tool_call_id": "tc2", "content": '{"status":"finished"}'},
                ],
                "api_calls": fhir_calls,
                "completed": finished,
            }

        agent.run_conversation.side_effect = side_effect_run_conversation
        return agent

    @pytest.mark.skipif(
        not _HERMES_AVAILABLE,
        reason="Hermes not installed — skipping mocked AIAgent test"
    )
    def test_run_returns_result_with_answer(self, tmp_path):
        from src.native.hermes.runner import HermesNativeRunner

        runner = HermesNativeRunner()
        runner.setup_task(
            task_metadata=_make_case(),
            task_prompt="Do the task.",
            tool_specs=[],
            model_config=_default_model_config(str(_FHIR_BASE)),
            runtime_constraints=_default_runtime(),
            memory_config=_default_memory(),
        )

        mock_agent = self._make_mock_agent(finished=True, fhir_calls=1)

        with patch("src.native.hermes.runner.HermesNativeRunner._build_agent",
                   return_value=mock_agent):
            result = runner.run()

        assert result.final_answer == '["S1234567"]'
        assert result.answer_parse_status == "ok"
        assert result.fhir_call_count == 1
        assert result.timed_out is False
        assert result.errors == []

    @pytest.mark.skipif(
        not _HERMES_AVAILABLE,
        reason="Hermes not installed"
    )
    def test_run_handles_agent_exception(self, tmp_path):
        from src.native.hermes.runner import HermesNativeRunner

        runner = HermesNativeRunner()
        runner.setup_task(
            task_metadata=_make_case(),
            task_prompt="Do the task.",
            tool_specs=[],
            model_config=_default_model_config(),
            runtime_constraints=_default_runtime(),
            memory_config=_default_memory(),
        )

        boom_agent = MagicMock()
        boom_agent.session_input_tokens = 0
        boom_agent.session_output_tokens = 0
        boom_agent.run_conversation.side_effect = RuntimeError("API error")

        with patch("src.native.hermes.runner.HermesNativeRunner._build_agent",
                   return_value=boom_agent):
            result = runner.run()

        assert result.final_answer is None
        assert result.answer_parse_status == "missing"
        assert len(result.errors) > 0

    @pytest.mark.skipif(
        not _HERMES_AVAILABLE,
        reason="Hermes not installed"
    )
    def test_teardown_after_run_clears_context(self, tmp_path):
        from src.native.hermes.runner import HermesNativeRunner
        from src.native.fhir_tools import get_task_context

        runner = HermesNativeRunner()
        runner.setup_task(
            task_metadata=_make_case(),
            task_prompt="Do the task.",
            tool_specs=[],
            model_config=_default_model_config(),
            runtime_constraints=_default_runtime(),
            memory_config=_default_memory(),
        )
        task_id = runner._hermes_task_id

        mock_agent = self._make_mock_agent()
        with patch("src.native.hermes.runner.HermesNativeRunner._build_agent",
                   return_value=mock_agent):
            runner.run()

        runner.teardown()
        assert get_task_context(task_id) is None


# ---------------------------------------------------------------------------
# Integration tests — requires Hermes + FHIR server
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(
    not _HERMES_AVAILABLE or not _FHIR_UP,
    reason="Requires Hermes installation and running FHIR server"
)
class TestHermesRunnerIntegration:
    """
    Full end-to-end test: real AIAgent, real FHIR server, real tools.
    Verifies that the runner produces a scored NativeHarnessResult for task1.
    """

    def test_task1_patient_lookup(self):
        """Task 1: find patient by name + DOB, using Hermes' pre-configured model."""
        from src.native.hermes.runner import HermesNativeRunner
        from src.native.driver import _NATIVE_PROMPT

        # Stafford patient known to be in the FHIR server.
        case = {
            "id": "task1_1",
            "instruction": "What's the MRN of the patient with name Peter Stafford and DOB of 1932-12-29?",
            "context": "",
            "sol": None,
            "eval_MRN": None,
        }

        prompt = _NATIVE_PROMPT.format(
            context=case["context"],
            instruction=case["instruction"],
        )

        # Use empty model_name so Hermes falls back to its pre-configured model
        # (openai-codex / gpt-5.4-mini from ~/.hermes/config.yaml).
        model_config = {
            "model_name": "",     # empty → Hermes uses its own default
            "api_key": None,
            "base_url": None,
            "temperature": 0.0,
            "max_tokens": 2048,
            "fhir_base_url": _FHIR_BASE,
        }

        runner = HermesNativeRunner()
        runner.setup_task(
            task_metadata=case,
            task_prompt=prompt,
            tool_specs=[],
            model_config=model_config,
            runtime_constraints=_default_runtime(),
            memory_config=_default_memory(),
        )

        result = runner.run()
        runner.teardown()

        assert isinstance(result, NativeHarnessResult)
        assert result.timed_out is False
        # Agent should have called fhir_finish (answer present) or reported errors.
        assert result.final_answer is not None or result.errors, (
            f"Expected answer or errors; got: {result}"
        )
        print(f"\nfinal_answer={result.final_answer}")
        print(f"fhir_calls={result.fhir_call_count}")
        print(f"parse_status={result.answer_parse_status}")
        print(f"errors={result.errors}")

    def test_toolset_restriction_only_fhir_tools_registered(self):
        """
        Verify that the fhir-medagent toolset contains exactly the 10 expected
        tools and that the Hermes registry has them all registered.
        Does not create an AIAgent (avoids LLM provider dependency).
        """
        import sys
        if str(_HERMES_AGENT_PATH) not in sys.path:
            sys.path.insert(0, str(_HERMES_AGENT_PATH))

        from src.native.fhir_tools import register_fhir_toolset

        register_fhir_toolset()

        from tools.registry import registry
        from toolsets import TOOLSETS, resolve_toolset, validate_toolset

        expected_tools = {
            "patient_search", "condition_search",
            "observation_search", "vitals_search",
            "vitals_create", "medication_request_search",
            "medication_request_create", "procedure_search",
            "service_request_create", "finish",
            "calculator", "show_plot",
        }

        # Toolset is registered in TOOLSETS dict.
        assert "fhir-medagent" in TOOLSETS

        # validate_toolset returns True.
        assert validate_toolset("fhir-medagent") is True

        # All 12 tool names appear in the registry.
        registered_names = set(registry._tools.keys()) if hasattr(registry, "_tools") else set()
        if not registered_names:
            registered_names = set(registry.tools.keys()) if hasattr(registry, "tools") else set()
        assert expected_tools.issubset(registered_names), (
            f"Missing from registry: {expected_tools - registered_names}"
        )

        # Resolved toolset tool names match expected.
        resolved = resolve_toolset("fhir-medagent")
        resolved_names = set(resolved) if resolved else set()
        assert expected_tools == resolved_names, (
            f"Toolset tool mismatch. Extra: {resolved_names - expected_tools}, "
            f"Missing: {expected_tools - resolved_names}"
        )
