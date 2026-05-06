"""
Tests for src/native/fhir_tools.py

Unit tests: no FHIR server or Hermes required.
Integration tests: require FHIR server at http://localhost:8080/fhir/ and
    the Hermes installation at ~/.hermes/hermes-agent.
    Skip automatically when the server is unreachable.

Run:
    pytest src/native/tests/test_fhir_tools.py -v
    pytest src/native/tests/test_fhir_tools.py -v -m integration  # integration only
"""

import json
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

# ── helpers ─────────────────────────────────────────────────────────────────

FHIR_BASE = "http://localhost:8080/fhir/"


def fhir_reachable() -> bool:
    try:
        r = requests.get(FHIR_BASE + "metadata", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


FHIR_UP = fhir_reachable()


def make_ctx(tmp_path: Path, max_calls: int = 8, task_id: str = "test-task") -> "TaskContext":
    from src.native.fhir_tools import TaskContext
    return TaskContext(
        fhir_base_url=FHIR_BASE,
        max_calls=max_calls,
        output_dir=tmp_path,
        hermes_task_id=task_id,
    )


# ── import ───────────────────────────────────────────────────────────────────

def test_module_imports():
    """fhir_tools must import without Hermes or FHIR available."""
    from src.native import fhir_tools  # noqa: F401


def test_tool_names_complete():
    from src.native.fhir_tools import get_tool_names
    names = get_tool_names()
    expected = {
        "patient_search",
        "condition_search",
        "observation_search",
        "vitals_search",
        "vitals_create",
        "medication_request_search",
        "medication_request_create",
        "procedure_search",
        "service_request_create",
        "finish",
        "calculator",
        "show_plot",
    }
    assert set(names) == expected, f"unexpected tool names: {set(names) ^ expected}"


# ── TaskContext ───────────────────────────────────────────────────────────────

class TestTaskContext:
    def test_budget_consumed(self, tmp_path):
        ctx = make_ctx(tmp_path, max_calls=3)
        assert ctx.try_consume() is True   # 1
        assert ctx.try_consume() is True   # 2
        assert ctx.try_consume() is True   # 3
        assert ctx.try_consume() is False  # exceeded
        assert ctx.budget_exceeded is True
        assert ctx.call_count == 3

    def test_budget_thread_safety(self, tmp_path):
        ctx = make_ctx(tmp_path, max_calls=10)
        results = []

        def consume():
            results.append(ctx.try_consume())

        threads = [threading.Thread(target=consume) for _ in range(15)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        allowed = sum(1 for r in results if r)
        denied = sum(1 for r in results if not r)
        assert allowed == 10
        assert denied == 5
        assert ctx.call_count == 10

    def test_log_call_writes_jsonl(self, tmp_path):
        ctx = make_ctx(tmp_path)
        ctx.log_call("patient_search", {"name": "Alice"}, '{"data": "..."}', True)
        log_path = tmp_path / "fhir_tool_calls.jsonl"
        assert log_path.exists()
        line = json.loads(log_path.read_text().strip())
        assert line["tool"] == "patient_search"
        assert line["counted"] is True
        assert line["seq"] == 0

    def test_multiple_calls_append_jsonl(self, tmp_path):
        ctx = make_ctx(tmp_path)
        for i in range(3):
            ctx.log_call(f"tool_{i}", {}, "response", True)
        lines = (tmp_path / "fhir_tool_calls.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3
        seqs = [json.loads(l)["seq"] for l in lines]
        assert seqs == [0, 1, 2]

    def test_initial_state(self, tmp_path):
        ctx = make_ctx(tmp_path)
        assert ctx.call_count == 0
        assert ctx.finished is False
        assert ctx.final_answer is None
        assert ctx.budget_exceeded is False
        assert ctx.tool_call_log == []


# ── context registry ─────────────────────────────────────────────────────────

class TestContextRegistry:
    def setup_method(self):
        from src.native.fhir_tools import _active_contexts, _ctx_lock
        with _ctx_lock:
            _active_contexts.clear()

    def test_set_and_get(self, tmp_path):
        from src.native.fhir_tools import set_task_context, get_task_context
        ctx = make_ctx(tmp_path, task_id="abc")
        set_task_context("abc", ctx)
        assert get_task_context("abc") is ctx

    def test_get_missing_returns_none(self):
        from src.native.fhir_tools import get_task_context
        assert get_task_context("nonexistent") is None

    def test_clear(self, tmp_path):
        from src.native.fhir_tools import set_task_context, get_task_context, clear_task_context
        ctx = make_ctx(tmp_path, task_id="xyz")
        set_task_context("xyz", ctx)
        clear_task_context("xyz")
        assert get_task_context("xyz") is None

    def test_multiple_tasks_isolated(self, tmp_path):
        from src.native.fhir_tools import set_task_context, get_task_context
        ctx_a = make_ctx(tmp_path / "a", task_id="a")
        ctx_b = make_ctx(tmp_path / "b", task_id="b")
        set_task_context("a", ctx_a)
        set_task_context("b", ctx_b)
        assert get_task_context("a") is ctx_a
        assert get_task_context("b") is ctx_b


# ── handler budget enforcement ────────────────────────────────────────────────

class TestHandlerBudget:
    def _call_handler(self, tool_name: str, args: dict, task_id: str) -> str:
        """Find the handler in _TOOLS and call it with a fake task_id kwarg."""
        from src.native.fhir_tools import _TOOLS, _make_handler
        for spec in _TOOLS:
            if spec["name"] == tool_name:
                h = _make_handler(spec["name"], spec["counted"], spec["impl"])
                return h(args, task_id=task_id)
        raise KeyError(tool_name)

    def test_budget_error_returned_as_json(self, tmp_path):
        from src.native.fhir_tools import set_task_context, clear_task_context
        ctx = make_ctx(tmp_path, max_calls=0, task_id="t1")
        set_task_context("t1", ctx)
        try:
            result = self._call_handler("patient_search", {"name": "Alice"}, "t1")
            parsed = json.loads(result)
            assert "error" in parsed
            assert "budget" in parsed["error"].lower() or "exhausted" in parsed["error"].lower()
        finally:
            clear_task_context("t1")

    def test_counted_tool_increments_counter(self, tmp_path):
        from src.native.fhir_tools import set_task_context, clear_task_context
        ctx = make_ctx(tmp_path, max_calls=5, task_id="t2")
        set_task_context("t2", ctx)
        try:
            with patch("src.native.fhir_tools._fhir_get", return_value='{"bundle": "ok"}'):
                self._call_handler("patient_search", {"name": "Alice"}, "t2")
            assert ctx.call_count == 1
        finally:
            clear_task_context("t2")

    def test_finish_not_counted(self, tmp_path):
        from src.native.fhir_tools import set_task_context, clear_task_context
        ctx = make_ctx(tmp_path, max_calls=0, task_id="t3")  # zero budget
        set_task_context("t3", ctx)
        try:
            # finish is not counted → should succeed even with zero budget
            result = self._call_handler("finish", {"answer": ["hello"]}, "t3")
            parsed = json.loads(result)
            assert parsed["status"] == "finished"
        finally:
            clear_task_context("t3")


# ── fhir_finish ──────────────────────────────────────────────────────────────

class TestFinishTool:
    def test_sets_context_state(self, tmp_path):
        from src.native.fhir_tools import set_task_context, clear_task_context, _TOOLS, _make_handler
        ctx = make_ctx(tmp_path, task_id="fin1")
        set_task_context("fin1", ctx)
        try:
            spec = next(s for s in _TOOLS if s["name"] == "finish")
            h = _make_handler("finish", False, spec["impl"])
            result = h({"answer": [42, "some_id"]}, task_id="fin1")
            parsed = json.loads(result)
            assert parsed["status"] == "finished"
            assert parsed["answer"] == [42, "some_id"]
            assert ctx.finished is True
            assert json.loads(ctx.final_answer) == [42, "some_id"]
        finally:
            clear_task_context("fin1")

    def test_empty_answer(self, tmp_path):
        from src.native.fhir_tools import set_task_context, clear_task_context, _TOOLS, _make_handler
        ctx = make_ctx(tmp_path, task_id="fin2")
        set_task_context("fin2", ctx)
        try:
            spec = next(s for s in _TOOLS if s["name"] == "finish")
            h = _make_handler("finish", False, spec["impl"])
            h({"answer": []}, task_id="fin2")
            assert ctx.finished is True
            assert json.loads(ctx.final_answer) == []
        finally:
            clear_task_context("fin2")

    def test_no_context_still_returns_ok(self, tmp_path):
        from src.native.fhir_tools import _TOOLS, _make_handler
        spec = next(s for s in _TOOLS if s["name"] == "finish")
        h = _make_handler("finish", False, spec["impl"])
        # No task_id → no context → should still return a valid response
        result = h({"answer": ["x"]}, task_id=None)
        parsed = json.loads(result)
        assert parsed["status"] == "finished"


# ── tool schema validation ────────────────────────────────────────────────────

class TestToolSchemas:
    def test_all_schemas_have_description(self):
        from src.native.fhir_tools import _TOOLS
        for spec in _TOOLS:
            assert "description" in spec["schema"], f"{spec['name']} missing description"
            assert spec["schema"]["description"], f"{spec['name']} has empty description"

    def test_all_schemas_have_parameters(self):
        from src.native.fhir_tools import _TOOLS
        for spec in _TOOLS:
            assert "parameters" in spec["schema"], f"{spec['name']} missing parameters"

    def test_required_fields_present_in_properties(self):
        from src.native.fhir_tools import _TOOLS
        for spec in _TOOLS:
            params = spec["schema"]["parameters"]
            required = params.get("required", [])
            props = params.get("properties", {})
            for req in required:
                assert req in props, (
                    f"{spec['name']}: required field '{req}' not in properties"
                )

    def test_finish_requires_answer(self):
        from src.native.fhir_tools import _TOOLS
        spec = next(s for s in _TOOLS if s["name"] == "finish")
        required = spec["schema"]["parameters"].get("required", [])
        assert "answer" in required

    def test_patient_search_requires_nothing(self):
        """Patient search is flexible; no FHIR query fields are required (only explanation)."""
        from src.native.fhir_tools import _TOOLS
        spec = next(s for s in _TOOLS if s["name"] == "patient_search")
        required = spec["schema"]["parameters"].get("required", [])
        fhir_fields = {"identifier", "name", "birthdate"}
        assert not fhir_fields.intersection(required), (
            f"patient_search should not require any FHIR fields, got required={required}"
        )

    def test_condition_search_requires_patient(self):
        from src.native.fhir_tools import _TOOLS
        spec = next(s for s in _TOOLS if s["name"] == "condition_search")
        assert "patient" in spec["schema"]["parameters"]["required"]

    def test_post_tools_require_resource(self):
        from src.native.fhir_tools import _TOOLS
        post_tools = [
            "vitals_create",
            "medication_request_create",
            "service_request_create",
        ]
        for name in post_tools:
            spec = next(s for s in _TOOLS if s["name"] == name)
            assert "resource" in spec["schema"]["parameters"]["required"], (
                f"{name}: 'resource' should be required"
            )

    def test_all_schemas_have_explanation_field(self):
        from src.native.fhir_tools import _TOOLS
        for spec in _TOOLS:
            props = spec["schema"]["parameters"].get("properties", {})
            assert "explanation" in props, (
                f"{spec['name']} schema missing 'explanation' field"
            )

    def test_all_tools_require_explanation(self):
        from src.native.fhir_tools import _TOOLS
        for spec in _TOOLS:
            required = spec["schema"]["parameters"].get("required", [])
            assert "explanation" in required, (
                f"{spec['name']} does not require 'explanation'"
            )


# ── Hermes registration ───────────────────────────────────────────────────────

@pytest.mark.integration
class TestHermesRegistration:
    def test_register_fhir_toolset_idempotent(self):
        """register_fhir_toolset() must be safe to call twice."""
        from src.native.fhir_tools import register_fhir_toolset, is_registered
        register_fhir_toolset()
        register_fhir_toolset()
        assert is_registered()

    def test_toolset_in_hermes_toolsets(self):
        from src.native.fhir_tools import register_fhir_toolset, TOOLSET_NAME
        register_fhir_toolset()
        from toolsets import TOOLSETS
        assert TOOLSET_NAME in TOOLSETS, f"'{TOOLSET_NAME}' not found in TOOLSETS"

    def test_all_tools_in_hermes_registry(self):
        from src.native.fhir_tools import register_fhir_toolset, get_tool_names, TOOLSET_NAME
        register_fhir_toolset()
        from tools.registry import registry
        registered = registry.get_tool_names_for_toolset(TOOLSET_NAME)
        for name in get_tool_names():
            assert name in registered, f"'{name}' not found in Hermes registry"

    def test_validate_toolset_returns_true(self):
        from src.native.fhir_tools import register_fhir_toolset, TOOLSET_NAME
        register_fhir_toolset()
        from toolsets import validate_toolset
        assert validate_toolset(TOOLSET_NAME)

    def test_get_tool_definitions_contains_fhir_tools(self):
        from src.native.fhir_tools import register_fhir_toolset, get_tool_names, TOOLSET_NAME
        register_fhir_toolset()
        from model_tools import get_tool_definitions
        defs = get_tool_definitions(enabled_toolsets=[TOOLSET_NAME], quiet_mode=True)
        def_names = {d["function"]["name"] for d in defs}
        for name in get_tool_names():
            assert name in def_names, f"'{name}' missing from tool definitions"

    def test_enabled_toolset_excludes_other_tools(self):
        """When enabled_toolsets=['fhir-medagent'], no built-in tools should appear."""
        from src.native.fhir_tools import register_fhir_toolset, get_tool_names, TOOLSET_NAME
        register_fhir_toolset()
        from model_tools import get_tool_definitions
        defs = get_tool_definitions(enabled_toolsets=[TOOLSET_NAME], quiet_mode=True)
        def_names = {d["function"]["name"] for d in defs}
        allowed = set(get_tool_names())
        extra = def_names - allowed
        assert not extra, f"unexpected tools appeared: {extra}"


# ── FHIR server integration ───────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.skipif(not FHIR_UP, reason="FHIR server not reachable at localhost:8080")
class TestFHIRIntegration:
    """Live tests against the MedAgentBench FHIR server."""

    KNOWN_PATIENT_NAME = "Peter Stafford"
    KNOWN_PATIENT_DOB = "1932-12-29"
    KNOWN_PATIENT_MRN = "S6534835"

    def _call(self, tool_name: str, args: dict, tmp_path: Path) -> dict:
        from src.native.fhir_tools import (
            register_fhir_toolset, set_task_context, clear_task_context,
            _TOOLS, _make_handler, TaskContext,
        )
        register_fhir_toolset()
        ctx = TaskContext(
            fhir_base_url=FHIR_BASE,
            max_calls=20,
            output_dir=tmp_path,
            hermes_task_id="integration-test",
        )
        set_task_context("integration-test", ctx)
        try:
            spec = next(s for s in _TOOLS if s["name"] == tool_name)
            h = _make_handler(spec["name"], spec["counted"], spec["impl"])
            result = h(args, task_id="integration-test")
            return json.loads(result)
        finally:
            clear_task_context("integration-test")

    def test_patient_search_by_name(self, tmp_path):
        # FHIR server matches on individual name tokens, not full "Firstname Lastname" strings.
        result = self._call("patient_search", {"name": "Stafford"}, tmp_path)
        assert "error" not in result, f"unexpected error: {result.get('error')}"
        assert result.get("resourceType") == "Bundle"
        entries = result.get("entry", [])
        assert len(entries) > 0, "Expected at least one patient result"

    def test_patient_search_by_dob(self, tmp_path):
        result = self._call(
            "patient_search",
            {"name": self.KNOWN_PATIENT_NAME, "birthdate": self.KNOWN_PATIENT_DOB},
            tmp_path,
        )
        assert "error" not in result
        assert result.get("resourceType") == "Bundle"

    def test_patient_search_nonexistent(self, tmp_path):
        result = self._call("patient_search", {"name": "ZZZNOTAREALPATIENT99999"}, tmp_path)
        # Should return a Bundle with 0 or no entries, not an error
        assert "error" not in result or result.get("total", 1) == 0

    def test_condition_search(self, tmp_path):
        result = self._call(
            "condition_search",
            {"patient": self.KNOWN_PATIENT_MRN, "category": "problem-list-item"},
            tmp_path,
        )
        assert "error" not in result
        assert result.get("resourceType") == "Bundle"

    def test_observation_labs_search(self, tmp_path):
        result = self._call(
            "observation_search",
            {"patient": self.KNOWN_PATIENT_MRN, "code": "MG"},
            tmp_path,
        )
        assert "error" not in result
        assert result.get("resourceType") == "Bundle"

    def test_observation_vitals_search(self, tmp_path):
        result = self._call(
            "vitals_search",
            {"patient": self.KNOWN_PATIENT_MRN, "category": "vital-signs"},
            tmp_path,
        )
        assert "error" not in result
        assert result.get("resourceType") == "Bundle"

    def test_medication_request_search(self, tmp_path):
        result = self._call(
            "medication_request_search",
            {"patient": self.KNOWN_PATIENT_MRN},
            tmp_path,
        )
        assert "error" not in result
        assert result.get("resourceType") == "Bundle"

    def test_procedure_search(self, tmp_path):
        result = self._call(
            "procedure_search",
            {"patient": self.KNOWN_PATIENT_MRN},
            tmp_path,
        )
        assert "error" not in result
        assert result.get("resourceType") == "Bundle"

    def test_budget_enforced_before_fhir_call(self, tmp_path):
        from src.native.fhir_tools import (
            register_fhir_toolset, set_task_context, clear_task_context,
            _TOOLS, _make_handler, TaskContext,
        )
        register_fhir_toolset()
        ctx = TaskContext(
            fhir_base_url=FHIR_BASE,
            max_calls=2,
            output_dir=tmp_path,
            hermes_task_id="budget-test",
        )
        set_task_context("budget-test", ctx)
        try:
            spec = next(s for s in _TOOLS if s["name"] == "patient_search")
            h = _make_handler("patient_search", True, spec["impl"])
            h({"name": "Alice"}, task_id="budget-test")   # call 1
            h({"name": "Bob"}, task_id="budget-test")     # call 2
            result = json.loads(h({"name": "Carol"}, task_id="budget-test"))  # call 3 → error
            assert "error" in result
            assert ctx.budget_exceeded is True
            assert ctx.call_count == 2
        finally:
            clear_task_context("budget-test")

    def test_call_log_populated(self, tmp_path):
        from src.native.fhir_tools import (
            register_fhir_toolset, set_task_context, clear_task_context,
            _TOOLS, _make_handler, TaskContext,
        )
        register_fhir_toolset()
        ctx = TaskContext(
            fhir_base_url=FHIR_BASE,
            max_calls=5,
            output_dir=tmp_path,
            hermes_task_id="log-test",
        )
        set_task_context("log-test", ctx)
        try:
            spec = next(s for s in _TOOLS if s["name"] == "patient_search")
            h = _make_handler("patient_search", True, spec["impl"])
            h({"name": "Peter Stafford"}, task_id="log-test")
            assert len(ctx.tool_call_log) == 1
            entry = ctx.tool_call_log[0]
            assert entry["tool"] == "patient_search"
            assert entry["counted"] is True
        finally:
            clear_task_context("log-test")


# ── calculator tool ───────────────────────────────────────────────────────────

class TestCalculatorTool:
    def _call_calculator(self, expression: str, tmp_path) -> dict:
        from src.native.fhir_tools import _TOOLS, _make_handler
        spec = next(s for s in _TOOLS if s["name"] == "calculator")
        h = _make_handler("calculator", spec["counted"], spec["impl"])
        return json.loads(h({"expression": expression, "explanation": "test"}, task_id=None))

    def test_basic_arithmetic(self, tmp_path):
        result = self._call_calculator("2 + 2 * 3", tmp_path)
        assert result["result"] == "8"

    def test_float_result(self, tmp_path):
        result = self._call_calculator("10 / 3", tmp_path)
        assert "3.3" in result["result"]

    def test_date_subtraction(self, tmp_path):
        expr = "(datetime(2024, 3, 15) - datetime(2024, 1, 1)).days"
        result = self._call_calculator(expr, tmp_path)
        assert result["result"] == "74"

    def test_invalid_expression_returns_error(self, tmp_path):
        result = self._call_calculator("__import__('os').system('ls')", tmp_path)
        assert "error" in result

    def test_not_counted_against_budget(self, tmp_path):
        from src.native.fhir_tools import _TOOLS
        spec = next(s for s in _TOOLS if s["name"] == "calculator")
        assert spec["counted"] is False


# ── show_plot tool ────────────────────────────────────────────────────────────

class TestShowPlotTool:
    def _call_show_plot(self, args: dict, tmp_path) -> dict:
        from src.native.fhir_tools import _TOOLS, _make_handler
        spec = next(s for s in _TOOLS if s["name"] == "show_plot")
        h = _make_handler("show_plot", spec["counted"], spec["impl"])
        return json.loads(h(args, task_id=None))

    def test_returns_recorded_status(self, tmp_path):
        result = self._call_show_plot({
            "x": [1, 2, 3], "y": [4, 5, 6],
            "x_label": "time", "y_label": "value",
            "explanation": "test plot",
        }, tmp_path)
        assert result["status"] == "recorded"

    def test_not_counted_against_budget(self, tmp_path):
        from src.native.fhir_tools import _TOOLS
        spec = next(s for s in _TOOLS if s["name"] == "show_plot")
        assert spec["counted"] is False
