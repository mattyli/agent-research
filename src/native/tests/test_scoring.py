"""Tests for src/native/scoring.py V2 evaluator dispatch."""
import json
from unittest.mock import MagicMock, patch


def _make_harness_result(final_answer=None, tool_call_log=None, timed_out=False, errors=None):
    from src.native.interface import NativeHarnessResult
    return NativeHarnessResult(
        final_answer=final_answer,
        answer_parse_status="ok" if final_answer else "missing",
        transcript=[],
        tool_call_log=tool_call_log or [],
        raw_harness_logs="",
        fhir_call_count=0,
        token_usage=None,
        latency_seconds=1.0,
        timed_out=timed_out,
        memory_mode="disabled",
        errors=errors or [],
        artifact_dir="/tmp",
    )


class TestExtractTaskPrefix:
    def test_simple_task(self):
        from src.native.scoring import _extract_task_prefix
        assert _extract_task_prefix("task1_1") == "task1"

    def test_task10(self):
        from src.native.scoring import _extract_task_prefix
        assert _extract_task_prefix("task10_15") == "task10"

    def test_unknown_returns_empty(self):
        from src.native.scoring import _extract_task_prefix
        assert _extract_task_prefix("unknown_123") == ""


class TestPostToolMap:
    def test_vitals_create_maps_to_observation(self):
        from src.native.scoring import _POST_TOOL_TO_RESOURCE
        assert _POST_TOOL_TO_RESOURCE["vitals_create"] == "Observation"

    def test_medication_request_create_maps_correctly(self):
        from src.native.scoring import _POST_TOOL_TO_RESOURCE
        assert _POST_TOOL_TO_RESOURCE["medication_request_create"] == "MedicationRequest"

    def test_service_request_create_maps_correctly(self):
        from src.native.scoring import _POST_TOOL_TO_RESOURCE
        assert _POST_TOOL_TO_RESOURCE["service_request_create"] == "ServiceRequest"

    def test_old_fhir_prefix_names_not_present(self):
        from src.native.scoring import _POST_TOOL_TO_RESOURCE
        for key in _POST_TOOL_TO_RESOURCE:
            assert not key.startswith("fhir_"), f"Old name still present: {key}"


class TestScoreResult:
    def test_calls_correct_task_function(self):
        from src.native.scoring import score_result

        mock_task_fn = MagicMock(return_value=True)

        with patch("src.native.scoring._get_v2_task_fns", return_value={"task1": mock_task_fn}):
            result = score_result(
                case_data={"id": "task1_1", "eval_MRN": "S123"},
                harness_result=_make_harness_result(final_answer='["some_answer"]'),
                fhir_base_url="http://localhost:8080/fhir/",
            )

        assert mock_task_fn.called
        assert result["success"] is True
        assert result["task_id"] == "task1_1"

    def test_unknown_prefix_sets_error(self):
        from src.native.scoring import score_result

        with patch("src.native.scoring._get_v2_task_fns", return_value={}):
            result = score_result(
                case_data={"id": "taskX_1"},
                harness_result=_make_harness_result(),
                fhir_base_url="http://localhost:8080/fhir/",
            )

        assert result["success"] is False
        assert result["error"] is not None

    def test_scorer_exception_is_caught(self):
        from src.native.scoring import score_result

        mock_task_fn = MagicMock(side_effect=RuntimeError("boom"))

        with patch("src.native.scoring._get_v2_task_fns", return_value={"task2": mock_task_fn}):
            result = score_result(
                case_data={"id": "task2_5"},
                harness_result=_make_harness_result(),
                fhir_base_url="http://localhost:8080/fhir/",
            )

        assert result["success"] is False
        assert "boom" in result["error"]
