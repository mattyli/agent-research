"""
Unit tests for NativeBenchDriver (Phase 2).

Run with:
    /opt/anaconda3/envs/medagentbench/bin/python3 -m pytest \
        src/native/tests/test_driver.py -v -m "not integration"
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Ensure repo root is on path.
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.native.interface import NativeHarnessResult, NativeHarnessRunner
from src.native.driver import NativeBenchDriver
from src.native.experiments.config_schema import NativeRunConfig, BenchmarkConfig, RuntimeConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_case(task_id: str = "task1_1") -> dict:
    return {
        "id": task_id,
        "instruction": "What is the MRN of patient John Smith?",
        "context": "FHIR server at localhost",
        "sol": ["S1234567"],
        "eval_MRN": "S1234567",
    }


def _make_func_spec() -> dict:
    return {
        "name": "GET {api_base}/Patient",
        "description": "Search for a patient",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}},
    }


def _make_harness_result(
    success: bool = True,
    timed_out: bool = False,
    fhir_calls: int = 2,
) -> NativeHarnessResult:
    return NativeHarnessResult(
        final_answer='["S1234567"]' if success else None,
        answer_parse_status="ok" if success else "missing",
        transcript=[
            {"role": "user", "content": "Query..."},
            {"role": "assistant", "content": ""},
        ],
        tool_call_log=[
            {"seq": 0, "tool": "fhir_patient_search", "args": {}, "counted": True},
        ],
        raw_harness_logs="",
        fhir_call_count=fhir_calls,
        token_usage={"input": 1000, "output": 200},
        latency_seconds=5.0,
        timed_out=timed_out,
        memory_mode="disabled",
        errors=[],
        artifact_dir="/tmp/test",
    )


_SCORE_STUB = {
    "task_id": "task1_1", "success": True, "error": None,
    "answer_parse_status": "ok", "fhir_call_count": 2,
    "timed_out": False, "latency_seconds": 5.0,
}


class MockRunner(NativeHarnessRunner):
    """Deterministic test runner."""

    def __init__(self, result_factory=None, setup_delay: float = 0):
        self._result_factory = result_factory or (lambda: _make_harness_result())
        self._setup_delay = setup_delay
        self.setup_called = False
        self.run_called = False
        self.teardown_called = False
        self._last_setup_args = {}

    def setup_task(self, task_metadata, task_prompt, tool_specs,
                   model_config, runtime_constraints, memory_config):
        self.setup_called = True
        self._last_setup_args = {
            "task_metadata": task_metadata,
            "task_prompt": task_prompt,
            "model_config": model_config,
        }

    def run(self) -> NativeHarnessResult:
        import time
        if self._setup_delay:
            time.sleep(self._setup_delay)
        self.run_called = True
        return self._result_factory()

    def teardown(self):
        self.teardown_called = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_driver_with_mock_data(
    tmp_path: Path,
    cases: Optional[List[dict]] = None,
    funcs: Optional[List[dict]] = None,
    extra_config: Optional[dict] = None,
) -> NativeBenchDriver:
    cases = cases or [_make_case()]
    funcs = funcs or [_make_func_spec()]

    data_file = tmp_path / "test_data.json"
    func_file = tmp_path / "funcs.json"
    data_file.write_text(json.dumps(cases))
    func_file.write_text(json.dumps(funcs))

    cfg_dict = {
        "benchmark": {
            "medagentbench_path": str(tmp_path),
            "task_split": "all",
            "fhir_base_url": "http://localhost:8080/fhir/",
            "data_file": str(data_file),
            "func_file": str(func_file),
        },
        "harness": {
            "adapter_class": "src.native.driver.NativeBenchDriver",  # unused with mock
        },
        "runtime": {
            "max_fhir_calls": 8,
            "max_iterations": 12,
            "timeout_seconds": 10,
        },
        "logging": {
            "output_dir": str(tmp_path / "outputs"),
            "log_tool_calls": True,
        },
    }
    if extra_config:
        cfg_dict.update(extra_config)

    config = NativeRunConfig.parse_obj(cfg_dict)
    driver = NativeBenchDriver.__new__(NativeBenchDriver)
    driver.config = config
    driver._all_cases = cases
    driver._funcs = funcs
    import time, uuid
    run_id = "test_" + uuid.uuid4().hex[:6]
    driver._run_id = run_id
    driver._output_root = tmp_path / "outputs" / "runs" / run_id
    driver._output_root.mkdir(parents=True, exist_ok=True)
    return driver


# ---------------------------------------------------------------------------
# Tests: task loading
# ---------------------------------------------------------------------------

class TestTaskLoading:
    def test_loads_all_cases(self, tmp_path):
        cases = [_make_case(f"task{i}_{j}") for i in range(1, 3) for j in range(1, 4)]
        driver = _make_driver_with_mock_data(tmp_path, cases=cases)
        assert len(driver._all_cases) == len(cases)

    def test_loads_funcs(self, tmp_path):
        funcs = [_make_func_spec(), _make_func_spec()]
        driver = _make_driver_with_mock_data(tmp_path, funcs=funcs)
        assert len(driver._funcs) == 2

    def test_data_file_missing_raises(self, tmp_path):
        config = NativeRunConfig.parse_obj({
            "benchmark": {
                "medagentbench_path": str(tmp_path),
                "data_file": str(tmp_path / "nonexistent.json"),
                "func_file": str(tmp_path / "funcs.json"),
            }
        })
        with pytest.raises(FileNotFoundError):
            NativeBenchDriver(config)


# ---------------------------------------------------------------------------
# Tests: task filtering
# ---------------------------------------------------------------------------

class TestTaskFiltering:
    def _driver_with_cases(self, tmp_path, task_ids=None, split="all"):
        cases = [
            _make_case(f"task1_{i}") for i in range(1, 31)
        ] + [
            _make_case(f"task2_{i}") for i in range(1, 31)
        ]
        driver = _make_driver_with_mock_data(tmp_path, cases=cases, extra_config={
            "benchmark": {
                "task_split": split,
                "task_ids": task_ids,
                "fhir_base_url": "http://localhost:8080/fhir/",
                "data_file": str(tmp_path / "data.json"),
                "func_file": str(tmp_path / "funcs.json"),
            }
        })
        driver.config.benchmark.task_split = split
        driver.config.benchmark.task_ids = task_ids
        driver._all_cases = cases
        return driver

    def test_all_split_returns_all(self, tmp_path):
        driver = self._driver_with_cases(tmp_path, split="all")
        assert len(driver._filtered_cases()) == 60

    def test_task_ids_filter(self, tmp_path):
        driver = self._driver_with_cases(tmp_path)
        driver.config.benchmark.task_ids = ["task1_1", "task1_2"]
        filtered = driver._filtered_cases()
        assert len(filtered) == 2
        assert all(c["id"] in {"task1_1", "task1_2"} for c in filtered)

    def test_sample_index_extraction(self, tmp_path):
        driver = _make_driver_with_mock_data(tmp_path)
        assert driver._sample_index("task1_1") == 1
        assert driver._sample_index("task10_30") == 30
        assert driver._sample_index("bad") == 0


# ---------------------------------------------------------------------------
# Tests: prompt building
# ---------------------------------------------------------------------------

class TestPromptBuilding:
    def test_prompt_contains_question(self, tmp_path):
        driver = _make_driver_with_mock_data(tmp_path)
        case = _make_case()
        prompt = driver._build_prompt(case)
        assert case["instruction"] in prompt

    def test_prompt_contains_context(self, tmp_path):
        driver = _make_driver_with_mock_data(tmp_path)
        case = _make_case()
        prompt = driver._build_prompt(case)
        assert case["context"] in prompt

    def test_prompt_contains_finish_instruction(self, tmp_path):
        driver = _make_driver_with_mock_data(tmp_path)
        prompt = driver._build_prompt(_make_case())
        assert "fhir_finish" in prompt


# ---------------------------------------------------------------------------
# Tests: run_one_task lifecycle
# ---------------------------------------------------------------------------

class TestRunOneTask:
    def test_runner_lifecycle_called(self, tmp_path):
        driver = _make_driver_with_mock_data(tmp_path)
        runner = MockRunner()
        with patch.object(driver, "_instantiate_runner", return_value=runner):
            with patch("src.native.driver.score_result", return_value={
                "task_id": "task1_1", "success": True, "error": None,
                "answer_parse_status": "ok", "fhir_call_count": 2,
                "timed_out": False, "latency_seconds": 5.0,
            }):
                driver.run_one_task(_make_case())

        assert runner.setup_called
        assert runner.run_called
        assert runner.teardown_called

    def test_teardown_called_even_on_run_error(self, tmp_path):
        driver = _make_driver_with_mock_data(tmp_path)

        def failing_run():
            raise RuntimeError("agent exploded")

        runner = MockRunner()
        runner.run = failing_run

        with patch.object(driver, "_instantiate_runner", return_value=runner):
            with patch("src.native.driver.score_result", return_value={
                "task_id": "task1_1", "success": False, "error": "agent exploded",
                "answer_parse_status": "missing", "fhir_call_count": 0,
                "timed_out": False, "latency_seconds": 0.0,
            }):
                result = driver.run_one_task(_make_case())

        assert runner.teardown_called

    def test_artifacts_written(self, tmp_path):
        driver = _make_driver_with_mock_data(tmp_path)
        runner = MockRunner()
        with patch.object(driver, "_instantiate_runner", return_value=runner):
            with patch("src.native.driver.score_result", return_value={
                "task_id": "task1_1", "success": True, "error": None,
                "answer_parse_status": "ok", "fhir_call_count": 2,
                "timed_out": False, "latency_seconds": 5.0,
            }):
                driver.run_one_task(_make_case())

        task_dir = driver._output_root / "tasks" / "task1_1"
        assert (task_dir / "task_metadata.json").exists()
        assert (task_dir / "score.json").exists()
        assert (task_dir / "task_summary.json").exists()
        assert (task_dir / "normalized_trajectory.jsonl").exists()
        assert (task_dir / "fhir_tool_calls.jsonl").exists()

    def test_model_config_passes_fhir_base_url(self, tmp_path):
        driver = _make_driver_with_mock_data(tmp_path)
        runner = MockRunner()
        with patch.object(driver, "_instantiate_runner", return_value=runner):
            with patch("src.native.driver.score_result", return_value=_SCORE_STUB):
                driver.run_one_task(_make_case())
        assert runner._last_setup_args["model_config"]["fhir_base_url"] == \
            "http://localhost:8080/fhir/"

    def test_model_config_passes_artifact_dir(self, tmp_path):
        driver = _make_driver_with_mock_data(tmp_path)
        runner = MockRunner()
        with patch.object(driver, "_instantiate_runner", return_value=runner):
            with patch("src.native.driver.score_result", return_value=_SCORE_STUB):
                driver.run_one_task(_make_case())
        artifact_dir = runner._last_setup_args["model_config"]["_artifact_dir"]
        assert "task1_1" in artifact_dir


# ---------------------------------------------------------------------------
# Tests: timeout
# ---------------------------------------------------------------------------

class TestTimeout:
    def test_timeout_produces_timed_out_result(self, tmp_path):
        driver = _make_driver_with_mock_data(tmp_path, extra_config={
            "runtime": {"timeout_seconds": 1, "max_fhir_calls": 8, "max_iterations": 12},
        })
        driver.config.runtime.timeout_seconds = 1

        slow_runner = MockRunner(setup_delay=5)  # sleeps 5s, timeout is 1s

        with patch.object(driver, "_instantiate_runner", return_value=slow_runner):
            with patch("src.native.driver.score_result", return_value={
                "task_id": "task1_1", "success": False, "error": None,
                "answer_parse_status": "missing", "fhir_call_count": 0,
                "timed_out": True, "latency_seconds": 1.0,
            }):
                result = driver.run_one_task(_make_case())

        assert slow_runner.teardown_called


# ---------------------------------------------------------------------------
# Tests: aggregate metrics
# ---------------------------------------------------------------------------

class TestAggregateMetrics:
    def _make_result(self, task_id, success, fhir_calls, timed_out=False):
        hr = _make_harness_result(success=success, timed_out=timed_out, fhir_calls=fhir_calls)
        score = {
            "task_id": task_id,
            "success": success,
            "error": None,
            "answer_parse_status": "ok" if success else "missing",
            "fhir_call_count": fhir_calls,
            "timed_out": timed_out,
            "latency_seconds": 3.0,
        }
        return {"task_id": task_id, "score": score, "harness_result": hr}

    def test_success_rate(self, tmp_path):
        driver = _make_driver_with_mock_data(tmp_path)
        results = [
            self._make_result("task1_1", True, 2),
            self._make_result("task1_2", False, 5),
            self._make_result("task1_3", True, 3),
        ]
        summary = driver._aggregate(results)
        assert summary["total"] == 3
        assert summary["successes"] == 2
        assert abs(summary["success_rate"] - 2/3) < 1e-9

    def test_per_task_type_breakdown(self, tmp_path):
        driver = _make_driver_with_mock_data(tmp_path)
        results = [
            self._make_result("task1_1", True, 2),
            self._make_result("task1_2", False, 5),
            self._make_result("task2_1", True, 1),
        ]
        summary = driver._aggregate(results)
        assert "task1" in summary["per_task_type"]
        assert "task2" in summary["per_task_type"]
        assert summary["per_task_type"]["task1"]["successes"] == 1
        assert summary["per_task_type"]["task2"]["successes"] == 1

    def test_empty_results(self, tmp_path):
        driver = _make_driver_with_mock_data(tmp_path)
        summary = driver._aggregate([])
        assert summary["total"] == 0


# ---------------------------------------------------------------------------
# Tests: config schema
# ---------------------------------------------------------------------------

class TestConfigSchema:
    def test_default_config(self):
        config = NativeRunConfig()
        assert config.benchmark.task_split == "all"
        assert config.memory.mode == "disabled"
        assert config.runtime.max_fhir_calls == 8

    def test_parse_from_dict(self):
        config = NativeRunConfig.parse_obj({
            "model": {"model_name": "anthropic/claude-haiku-4-5", "temperature": 0.5},
            "runtime": {"timeout_seconds": 60},
        })
        assert config.model.model_name == "anthropic/claude-haiku-4-5"
        assert config.model.temperature == 0.5
        assert config.runtime.timeout_seconds == 60

    def test_from_yaml(self, tmp_path):
        yaml_content = """
benchmark:
  fhir_base_url: "http://localhost:8080/fhir/"
model:
  model_name: "anthropic/claude-opus-4-5"
memory:
  mode: disabled
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)
        config = NativeRunConfig.from_yaml(str(yaml_file))
        assert config.model.model_name == "anthropic/claude-opus-4-5"
        assert config.memory.mode == "disabled"


# ---------------------------------------------------------------------------
# Tests: scoring bridge (unit — no FHIR server needed)
# ---------------------------------------------------------------------------

class TestScoringBridge:
    def test_build_task_output_completed_status(self, tmp_path):
        from src.native.scoring import build_task_output

        hr = _make_harness_result(success=True)
        task_output = build_task_output(hr, "http://localhost:8080/fhir/")

        from src.typings import SampleStatus
        assert task_output.status == SampleStatus.COMPLETED
        assert task_output.result == '["S1234567"]'

    def test_build_task_output_missing_answer(self):
        from src.native.scoring import build_task_output

        hr = _make_harness_result(success=False)
        task_output = build_task_output(hr, "http://localhost:8080/fhir/")

        from src.typings import SampleStatus
        assert task_output.status == SampleStatus.TASK_LIMIT_REACHED
        assert task_output.result is None

    def test_post_history_injection(self):
        from src.native.scoring import build_task_output

        hr = NativeHarnessResult(
            final_answer='["done"]',
            answer_parse_status="ok",
            transcript=[
                {"role": "user", "content": "Do the task"},
                {"role": "assistant", "content": ""},
                {"role": "tool", "name": "fhir_medication_request_create",
                 "tool_call_id": "tc1", "content": '{"status":"created"}'},
                {"role": "assistant", "content": ""},
                {"role": "tool", "name": "fhir_finish",
                 "tool_call_id": "tc2", "content": '{"status":"finished"}'},
            ],
            tool_call_log=[
                {
                    "seq": 0,
                    "tool": "fhir_medication_request_create",
                    "args": {"resource": {"resourceType": "MedicationRequest"}},
                    "response_preview": '{"status":"created"}',
                    "counted": True,
                }
            ],
            raw_harness_logs="",
            fhir_call_count=1,
            token_usage=None,
            latency_seconds=3.0,
            timed_out=False,
            memory_mode="disabled",
        )
        task_output = build_task_output(hr, "http://localhost:8080/fhir/")

        # Find agent-role entries with POST.
        post_entries = [
            item for item in task_output.history
            if item.role == "agent" and "POST" in item.content
        ]
        assert len(post_entries) == 1
        assert "MedicationRequest" in post_entries[0].content

        # Check acknowledgment follows.
        post_idx = task_output.history.index(post_entries[0])
        ack_entry = task_output.history[post_idx + 1]
        assert "POST request accepted" in ack_entry.content

    def test_timed_out_status(self):
        from src.native.scoring import build_task_output
        from src.typings import SampleStatus

        hr = _make_harness_result(timed_out=True)
        hr.timed_out = True
        task_output = build_task_output(hr, "http://localhost:8080/fhir/")
        assert task_output.status == SampleStatus.TASK_LIMIT_REACHED
