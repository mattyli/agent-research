from unittest.mock import MagicMock, patch


def _make_results(result_str=None):
    results = MagicMock()
    results.result = result_str
    results.history = []
    return results


class TestEvalV2Routes:
    def test_routes_to_task1(self):
        from src.server.tasks.medagentbench.eval_v2 import eval as eval_v2

        mock_task_fn = MagicMock(return_value=True)
        mock_nr = MagicMock()
        mock_nr.task1 = mock_task_fn

        with patch("src.server.tasks.medagentbench.eval_v2._new_refsol", mock_nr):
            result = eval_v2(
                {"id": "task1_1"}, _make_results("[1]"), "http://localhost:8080/fhir/"
            )

        assert result is True
        assert mock_task_fn.called

    def test_routes_to_task10(self):
        from src.server.tasks.medagentbench.eval_v2 import eval as eval_v2

        mock_task_fn = MagicMock(return_value=True)
        mock_nr = MagicMock()
        mock_nr.task10 = mock_task_fn

        with patch("src.server.tasks.medagentbench.eval_v2._new_refsol", mock_nr):
            result = eval_v2(
                {"id": "task10_15"}, _make_results("[5.0]"), "http://localhost:8080/fhir/"
            )

        assert result is True
        assert mock_task_fn.called

    def test_returns_false_on_grader_exception(self):
        from src.server.tasks.medagentbench.eval_v2 import eval as eval_v2

        mock_nr = MagicMock()
        mock_nr.task2 = MagicMock(side_effect=RuntimeError("boom"))

        with patch("src.server.tasks.medagentbench.eval_v2._new_refsol", mock_nr):
            result = eval_v2(
                {"id": "task2_5"}, _make_results(), "http://localhost:8080/fhir/"
            )

        assert result is False

    def test_returns_false_when_grader_returns_false(self):
        from src.server.tasks.medagentbench.eval_v2 import eval as eval_v2

        mock_nr = MagicMock()
        mock_nr.task3 = MagicMock(return_value=False)

        with patch("src.server.tasks.medagentbench.eval_v2._new_refsol", mock_nr):
            result = eval_v2(
                {"id": "task3_2"}, _make_results(), "http://localhost:8080/fhir/"
            )

        assert result is False
