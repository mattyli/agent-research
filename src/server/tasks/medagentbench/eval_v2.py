import importlib

_new_refsol = None


def _get_new_refsol():
    global _new_refsol
    if _new_refsol is None:
        _new_refsol = importlib.import_module(
            "src.server.tasks.medagentbench.new_refsol"
        )
    return _new_refsol


def eval(case_data, results, fhir_api_base):
    task_id = case_data["id"].split("_")[0]
    grader_func = getattr(_get_new_refsol(), task_id)
    try:
        if grader_func(case_data, results, fhir_api_base) is True:
            return True
    except Exception as e:
        print(e)
    return False
