import json


class TestSafeEval:
    def test_basic_arithmetic(self):
        from src.server.tasks.medagentbench.medagentbench_v2 import _safe_eval
        result = json.loads(_safe_eval("2 + 2 * 3"))
        assert result["result"] == "8"

    def test_float_division(self):
        from src.server.tasks.medagentbench.medagentbench_v2 import _safe_eval
        result = json.loads(_safe_eval("10 / 4"))
        assert result["result"] == "2.5"

    def test_date_subtraction_days(self):
        from src.server.tasks.medagentbench.medagentbench_v2 import _safe_eval
        result = json.loads(_safe_eval("(datetime(2024, 3, 15) - datetime(2024, 1, 1)).days"))
        assert result["result"] == "74"

    def test_age_calculation(self):
        from src.server.tasks.medagentbench.medagentbench_v2 import _safe_eval
        # Born 1985-04-20, reference date 2023-11-13 → age 38
        result = json.loads(_safe_eval(
            "(datetime(2023, 11, 13) - datetime(1985, 4, 20)).days // 365"
        ))
        assert result["result"] == "38"

    def test_round(self):
        from src.server.tasks.medagentbench.medagentbench_v2 import _safe_eval
        result = json.loads(_safe_eval("round(3.14159, 2)"))
        assert result["result"] == "3.14"

    def test_blocks_import_builtin(self):
        from src.server.tasks.medagentbench.medagentbench_v2 import _safe_eval
        result = json.loads(_safe_eval("__import__('os').system('ls')"))
        assert "error" in result

    def test_blocks_undefined_name(self):
        from src.server.tasks.medagentbench.medagentbench_v2 import _safe_eval
        result = json.loads(_safe_eval("not_defined_fn()"))
        assert "error" in result

    def test_empty_expression_returns_error(self):
        from src.server.tasks.medagentbench.medagentbench_v2 import _safe_eval
        result = json.loads(_safe_eval(""))
        assert "error" in result
