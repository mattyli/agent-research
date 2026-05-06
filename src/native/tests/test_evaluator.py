def test_new_refsol_imports():
    from src.server.tasks.medagentbench import new_refsol  # noqa: F401

def test_new_refsol_has_all_task_functions():
    from src.server.tasks.medagentbench import new_refsol
    for i in range(1, 11):
        assert hasattr(new_refsol, f"task{i}"), f"new_refsol missing task{i}"

def test_v2_utils_imports():
    from src.server.tasks.medagentbench import v2_utils  # noqa: F401

def test_v2_utils_has_send_get_request():
    from src.server.tasks.medagentbench.v2_utils import send_get_request
    assert callable(send_get_request)
