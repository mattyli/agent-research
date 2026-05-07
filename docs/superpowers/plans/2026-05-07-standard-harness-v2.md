# Standard Harness V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `MedAgentBenchV2` subclass so gpt-5.4-nano can run through the standard assigner harness with V2 tools (10 FHIR + calculator) and V2 evaluation (`new_refsol.py`), while leaving all V1 files untouched.

**Architecture:** New subclass `MedAgentBenchV2` inherits from `MedAgentBench`, overrides `start_sample` to intercept `GET .../calculator` calls with safe local eval, and overrides `calculate_overall` to use `eval_v2.eval()`. A new `funcs_v2.json` provides V2 tool descriptions. A dedicated `configs/assignments/v2.yaml` runs the V2 agent without touching the V1 assignment config.

**Tech Stack:** Python 3.9, pytest, Pydantic v1, asyncio, PyYAML, requests, FHIR R4 (Hapi FHIR on port 8080), OpenAI-compatible HTTP API

---

## File Map

| Action | File | Responsibility |
|---|---|---|
| Create | `data/medagentbench/funcs_v2.json` | 11 V2 tool descriptions (10 FHIR + calculator) |
| Create | `src/server/tasks/medagentbench/eval_v2.py` | Eval shim routing to `new_refsol.py` |
| Create | `src/server/tasks/medagentbench/tests/__init__.py` | Test package marker |
| Create | `src/server/tasks/medagentbench/tests/test_eval_v2.py` | Tests for eval_v2 routing |
| Create | `src/server/tasks/medagentbench/tests/test_medagentbench_v2.py` | Tests for _safe_eval |
| Create | `src/server/tasks/medagentbench/medagentbench_v2.py` | MedAgentBenchV2 subclass |
| Modify | `configs/tasks/medagentbench.yaml` | Add medagentbench-v2-std task variant |
| Modify | `configs/agents/api_agents.yaml` | Add gpt-5.4-nano agent entry |
| Create | `configs/assignments/v2.yaml` | V2-specific assignment config |

---

## Task 1: Create funcs_v2.json

**Files:**
- Create: `data/medagentbench/funcs_v2.json`

- [ ] **Step 1: Write funcs_v2.json**

Create `data/medagentbench/funcs_v2.json` with this exact content:

```json
[
  {
    "name": "patient_search (GET {api_base}/Patient)",
    "description": "Search for a FHIR Patient record. Returns matching patient resources including patient ID (used as the 'patient' parameter in other tools). Provide at least one of identifier, name, or birthdate.",
    "parameters": {
      "type": "object",
      "properties": {
        "identifier": {"type": "string", "description": "Patient MRN or other identifier."},
        "name": {"type": "string", "description": "Patient name token (family or given). Use a single token (e.g. 'Stafford'), not a full name."},
        "birthdate": {"type": "string", "description": "Patient date of birth in YYYY-MM-DD format."},
        "explanation": {"type": "string", "description": "One sentence justifying this tool call before executing it."}
      },
      "required": ["explanation"]
    }
  },
  {
    "name": "condition_search (GET {api_base}/Condition)",
    "description": "Search the patient's problem list (Condition resources). Use category='problem-list-item' for the standard problem list.",
    "parameters": {
      "type": "object",
      "properties": {
        "patient": {"type": "string", "description": "FHIR Patient resource ID."},
        "category": {"type": "string", "description": "Condition category, typically 'problem-list-item'."},
        "explanation": {"type": "string", "description": "One sentence justifying this tool call before executing it."}
      },
      "required": ["patient", "explanation"]
    }
  },
  {
    "name": "observation_search (GET {api_base}/Observation)",
    "description": "Search lab results (Observation resources). Provide a friendly lab code such as 'K', 'A1C', 'MG', 'GLU', 'NA' — not a LOINC string.",
    "parameters": {
      "type": "object",
      "properties": {
        "patient": {"type": "string", "description": "FHIR Patient resource ID."},
        "code": {"type": "string", "description": "Lab code (e.g. 'K', 'A1C', 'MG', 'GLU', 'NA')."},
        "date": {"type": "string", "description": "Filter by date (YYYY-MM-DD) or date range (e.g. ge2023-01-01)."},
        "explanation": {"type": "string", "description": "One sentence justifying this tool call before executing it."}
      },
      "required": ["patient", "code", "explanation"]
    }
  },
  {
    "name": "vitals_search (GET {api_base}/Observation)",
    "description": "Search vital sign observations (Observation resources with category vital-signs).",
    "parameters": {
      "type": "object",
      "properties": {
        "patient": {"type": "string", "description": "FHIR Patient resource ID."},
        "category": {"type": "string", "description": "Must be 'vital-signs'."},
        "date": {"type": "string", "description": "Filter by date or date range (e.g. ge2023-11-12)."},
        "explanation": {"type": "string", "description": "One sentence justifying this tool call before executing it."}
      },
      "required": ["patient", "category", "explanation"]
    }
  },
  {
    "name": "vitals_create (POST {api_base}/Observation)",
    "description": "Create (POST) a vital signs Observation resource. The payload must be valid FHIR Observation JSON with resourceType='Observation', category vital-signs, effectiveDateTime, status='final', code with text set to the flowsheet ID (e.g. 'BP'), valueString, and subject reference.",
    "parameters": {
      "type": "object",
      "properties": {
        "resourceType": {"type": "string", "description": "Use 'Observation'."},
        "category": {"type": "array", "description": "Use [{\"coding\": [{\"system\": \"http://hl7.org/fhir/observation-category\", \"code\": \"vital-signs\", \"display\": \"Vital Signs\"}]}]"},
        "code": {"type": "object", "description": "The flowsheet ID as text, e.g. {\"text\": \"BP\"}."},
        "effectiveDateTime": {"type": "string", "description": "ISO datetime of the observation."},
        "status": {"type": "string", "description": "Use 'final'."},
        "valueString": {"type": "string", "description": "Measurement value as a string, e.g. '120/80 mmHg'."},
        "subject": {"type": "object", "description": "Patient reference, e.g. {\"reference\": \"Patient/123\"}."},
        "explanation": {"type": "string", "description": "One sentence justifying this tool call before executing it."}
      },
      "required": ["resourceType", "category", "code", "effectiveDateTime", "status", "valueString", "subject", "explanation"]
    }
  },
  {
    "name": "medication_request_search (GET {api_base}/MedicationRequest)",
    "description": "Search MedicationRequest resources for a patient.",
    "parameters": {
      "type": "object",
      "properties": {
        "patient": {"type": "string", "description": "FHIR Patient resource ID."},
        "explanation": {"type": "string", "description": "One sentence justifying this tool call before executing it."}
      },
      "required": ["patient", "explanation"]
    }
  },
  {
    "name": "medication_request_create (POST {api_base}/MedicationRequest)",
    "description": "Create (POST) a MedicationRequest resource to order a medication. Must include resourceType, medicationCodeableConcept with NDC code, subject reference, status='active', intent='order', authoredOn, and dosageInstruction.",
    "parameters": {
      "type": "object",
      "properties": {
        "resourceType": {"type": "string", "description": "Use 'MedicationRequest'."},
        "medicationCodeableConcept": {"type": "object", "description": "Medication with coding array (system, code, display) and text."},
        "authoredOn": {"type": "string", "description": "Date the prescription was written (ISO format)."},
        "dosageInstruction": {"type": "array", "description": "Dosage instructions array."},
        "status": {"type": "string", "description": "Use 'active'."},
        "intent": {"type": "string", "description": "Use 'order'."},
        "subject": {"type": "object", "description": "Patient reference, e.g. {\"reference\": \"Patient/123\"}."},
        "explanation": {"type": "string", "description": "One sentence justifying this tool call before executing it."}
      },
      "required": ["resourceType", "medicationCodeableConcept", "authoredOn", "dosageInstruction", "status", "intent", "subject", "explanation"]
    }
  },
  {
    "name": "procedure_search (GET {api_base}/Procedure)",
    "description": "Search Procedure resources for a patient.",
    "parameters": {
      "type": "object",
      "properties": {
        "patient": {"type": "string", "description": "FHIR Patient resource ID."},
        "date": {"type": "string", "description": "Date or period the procedure was performed."},
        "explanation": {"type": "string", "description": "One sentence justifying this tool call before executing it."}
      },
      "required": ["patient", "explanation"]
    }
  },
  {
    "name": "service_request_create (POST {api_base}/ServiceRequest)",
    "description": "Create (POST) a ServiceRequest resource to order a procedure or consult. Must include resourceType='ServiceRequest', code with SNOMED or LOINC coding, subject reference, status='active', intent='order', priority='stat', authoredOn. For referral orders include note.text with the free-text narrative.",
    "parameters": {
      "type": "object",
      "properties": {
        "resourceType": {"type": "string", "description": "Use 'ServiceRequest'."},
        "code": {"type": "object", "description": "Procedure code with coding array (system, code, display)."},
        "authoredOn": {"type": "string", "description": "Order datetime in ISO format."},
        "status": {"type": "string", "description": "Use 'active'."},
        "intent": {"type": "string", "description": "Use 'order'."},
        "priority": {"type": "string", "description": "Use 'stat'."},
        "subject": {"type": "object", "description": "Patient reference, e.g. {\"reference\": \"Patient/123\"}."},
        "note": {"type": "object", "description": "Free-text narrative, e.g. {\"text\": \"SBAR narrative here\"}."},
        "occurrenceDateTime": {"type": "string", "description": "Scheduled datetime in ISO format."},
        "explanation": {"type": "string", "description": "One sentence justifying this tool call before executing it."}
      },
      "required": ["resourceType", "code", "authoredOn", "status", "intent", "priority", "subject", "explanation"]
    }
  },
  {
    "name": "calculator (GET {api_base}/calculator)",
    "description": "Evaluate a Python arithmetic expression safely. Supports math operators, datetime, timedelta, and Decimal. Use for date arithmetic (e.g. days since last procedure) and numeric calculations before calling FINISH.",
    "parameters": {
      "type": "object",
      "properties": {
        "expression": {"type": "string", "description": "A Python expression to evaluate. May use: arithmetic operators, datetime(), timedelta(), math.*, abs(), round(), int(), float(), len(), min(), max(), sum(). Example: (datetime(2023,11,13) - datetime(1985,4,20)).days // 365"},
        "explanation": {"type": "string", "description": "One sentence justifying this tool call before executing it."}
      },
      "required": ["expression", "explanation"]
    }
  }
]
```

- [ ] **Step 2: Verify the JSON is valid and has 11 entries with explanation fields**

```bash
python3 -c "
import json
with open('data/medagentbench/funcs_v2.json') as f:
    funcs = json.load(f)
assert len(funcs) == 11, f'Expected 11, got {len(funcs)}'
for fn in funcs:
    props = fn['parameters']['properties']
    assert 'explanation' in props, f\"{fn['name']} missing explanation\"
    assert 'explanation' in fn['parameters']['required'], f\"{fn['name']} explanation not required\"
names = [fn['name'].split(' ')[0] for fn in funcs]
assert 'calculator' in names, 'calculator missing'
print('OK:', names)
"
```

Expected: `OK: ['patient_search', 'condition_search', 'observation_search', ...]`

- [ ] **Step 3: Commit**

```bash
git add data/medagentbench/funcs_v2.json
git commit -m "data: add funcs_v2.json with V2 tool descriptions and calculator"
```

---

## Task 2: Create eval_v2.py (TDD)

**Files:**
- Create: `src/server/tasks/medagentbench/tests/__init__.py`
- Create: `src/server/tasks/medagentbench/tests/test_eval_v2.py`
- Create: `src/server/tasks/medagentbench/eval_v2.py`

- [ ] **Step 1: Create the test package**

```bash
mkdir -p src/server/tasks/medagentbench/tests
touch src/server/tasks/medagentbench/tests/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `src/server/tasks/medagentbench/tests/test_eval_v2.py`:

```python
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
```

- [ ] **Step 3: Run the tests to confirm they fail**

```bash
python -m pytest src/server/tasks/medagentbench/tests/test_eval_v2.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.server.tasks.medagentbench.eval_v2'`

- [ ] **Step 4: Implement eval_v2.py**

Create `src/server/tasks/medagentbench/eval_v2.py`:

```python
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
```

- [ ] **Step 5: Run the tests to confirm they pass**

```bash
python -m pytest src/server/tasks/medagentbench/tests/test_eval_v2.py -v
```

Expected: 4 tests PASSED

- [ ] **Step 6: Commit**

```bash
git add src/server/tasks/medagentbench/eval_v2.py \
        src/server/tasks/medagentbench/tests/__init__.py \
        src/server/tasks/medagentbench/tests/test_eval_v2.py
git commit -m "feat: add eval_v2.py routing to new_refsol"
```

---

## Task 3: Create medagentbench_v2.py (TDD)

**Files:**
- Create: `src/server/tasks/medagentbench/tests/test_medagentbench_v2.py`
- Create: `src/server/tasks/medagentbench/medagentbench_v2.py`

- [ ] **Step 1: Write the failing tests**

Create `src/server/tasks/medagentbench/tests/test_medagentbench_v2.py`:

```python
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest src/server/tasks/medagentbench/tests/test_medagentbench_v2.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.server.tasks.medagentbench.medagentbench_v2'`

- [ ] **Step 3: Implement medagentbench_v2.py**

Create `src/server/tasks/medagentbench/medagentbench_v2.py`:

```python
import decimal
import importlib
import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from urllib.parse import parse_qs, unquote_plus, urlparse

from src.server.task import Session
from src.typings import AgentOutputStatus, SampleStatus, TaskOutput

from . import MedAgentBench
from .utils import send_get_request

MedAgentBenchV2_prompt = """You are an expert in using FHIR functions to assist medical professionals. You are given an instruction and a set of possible functions. Based on the instruction, you will need to make one or more function/tool calls to achieve the purpose.

1. If you decide to invoke a GET function, you MUST put it in the format of
GET url?param_name1=param_value1&param_name2=param_value2...

2. If you decide to invoke a POST function, you MUST put it in the format of
POST url
[your payload data in JSON format]

3. If you have got answers for all the questions and finished all the requested tasks, you MUST call to finish the conversation in the format of (make sure the list is JSON loadable.)
FINISH([answer1, answer2, ...])

Your response must be in the format of one of the three cases, and you can call only one function each time. You SHOULD NOT include any other text in the response.

Include an 'explanation' parameter on every tool call explaining why you are making it. Use the calculator tool for date and numeric arithmetic.

Here is a list of functions in JSON format that you can invoke. Note that you should use {api_base} as the api_base.
{functions}

Context: {context}
Instruction: {question}"""

_SAFE_CALC_GLOBALS = {
    "__builtins__": {},
    "math": math,
    "datetime": datetime,
    "timedelta": timedelta,
    "timezone": timezone,
    "Decimal": decimal.Decimal,
    "abs": abs,
    "round": round,
    "int": int,
    "float": float,
    "str": str,
    "len": len,
    "min": min,
    "max": max,
    "sum": sum,
}


def _safe_eval(expression: str) -> str:
    if not expression:
        return json.dumps({"error": "empty expression"})
    try:
        result = eval(expression, _SAFE_CALC_GLOBALS, {})  # noqa: S307
        return json.dumps({"result": str(result)})
    except Exception as exc:
        return json.dumps({"error": f"calculator error: {exc}"})


class MedAgentBenchV2(MedAgentBench):
    def __init__(self, **configs):
        super().__init__(**configs)
        try:
            importlib.import_module("src.server.tasks.medagentbench.new_refsol")
        except ModuleNotFoundError:
            print("new_refsol.py not found at src/server/tasks/medagentbench/new_refsol.py")
            exit()

    async def start_sample(self, index, session: Session):
        print(f"task start {index}")
        case = self.data[index]
        session.inject({
            "role": "user",
            "content": MedAgentBenchV2_prompt.format(
                api_base=self.fhir_api_base,
                functions=json.dumps(self.funcs),
                context=case["context"],
                question=case["instruction"],
            ),
        })
        try:
            for _ in range(self.max_round):
                res = await session.action()
                if res.status == AgentOutputStatus.AGENT_CONTEXT_LIMIT:
                    return TaskOutput(
                        status=SampleStatus.AGENT_CONTEXT_LIMIT,
                        history=session.history,
                    )
                r = res.content.strip().replace("```tool_code", "").replace("```", "").strip()

                if r.startswith("GET") and "/calculator" in r.split("?")[0]:
                    parsed = urlparse(r[3:].strip())
                    params = parse_qs(parsed.query)
                    expression = unquote_plus(params.get("expression", [""])[0])
                    result = _safe_eval(expression)
                    session.inject({
                        "role": "user",
                        "content": (
                            f"Calculator result: {result}. "
                            "Please call FINISH if you have got answers for all the "
                            "questions and finished all the requested tasks"
                        ),
                    })

                elif r.startswith("GET"):
                    url = r[3:].strip() + "&_format=json"
                    get_res = send_get_request(url)
                    if "data" in get_res:
                        session.inject({
                            "role": "user",
                            "content": (
                                f"Here is the response from the GET request:\n{get_res['data']}. "
                                "Please call FINISH if you have got answers for all the "
                                "questions and finished all the requested tasks"
                            ),
                        })
                    else:
                        session.inject({
                            "role": "user",
                            "content": f"Error in sending the GET request: {get_res['error']}",
                        })

                elif r.startswith("POST"):
                    try:
                        json.loads("\n".join(r.split("\n")[1:]))
                    except Exception:
                        session.inject({"role": "user", "content": "Invalid POST request"})
                    else:
                        session.inject({
                            "role": "user",
                            "content": (
                                "POST request accepted and executed successfully. "
                                "Please call FINISH if you have got answers for all the "
                                "questions and finished all the requested tasks"
                            ),
                        })

                elif r.startswith("FINISH("):
                    return TaskOutput(
                        status=SampleStatus.COMPLETED,
                        result=r[len("FINISH("):-1],
                        history=session.history,
                    )

                else:
                    return TaskOutput(
                        status=SampleStatus.AGENT_INVALID_ACTION,
                        history=session.history,
                    )

        except Exception as e:
            return TaskOutput(
                status=SampleStatus.TASK_ERROR,
                result={"error": str(e)},
                history=session.history,
            )

        return TaskOutput(
            status=SampleStatus.TASK_LIMIT_REACHED,
            history=session.history,
        )

    def calculate_overall(self, results: List[TaskOutput]) -> Dict[str, Any]:
        from .eval_v2 import eval as eval_v2

        total_task = len(results)
        assert len(self.get_indices()) == total_task
        correct_count = 0
        for i in range(total_task):
            if getattr(results[i], "result") is not None:
                index = results[i].index
                if eval_v2(self.data[index], results[i], self.fhir_api_base) is True:
                    correct_count += 1
                    results[i].status += "Correct"
                else:
                    results[i].status += "Incorrect"
        return {"success rate": correct_count / total_task, "raw_results": results}
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
python -m pytest src/server/tasks/medagentbench/tests/test_medagentbench_v2.py -v
```

Expected: 8 tests PASSED

- [ ] **Step 5: Run the full test suite to check for regressions**

```bash
python -m pytest src/server/tasks/medagentbench/tests/ -v
```

Expected: All 12 tests PASSED (4 eval_v2 + 8 safe_eval)

- [ ] **Step 6: Commit**

```bash
git add src/server/tasks/medagentbench/medagentbench_v2.py \
        src/server/tasks/medagentbench/tests/test_medagentbench_v2.py
git commit -m "feat: add MedAgentBenchV2 subclass with calculator intercept"
```

---

## Task 4: Update Configs

**Files:**
- Modify: `configs/tasks/medagentbench.yaml`
- Modify: `configs/agents/api_agents.yaml`
- Create: `configs/assignments/v2.yaml`

- [ ] **Step 1: Add medagentbench-v2-std to configs/tasks/medagentbench.yaml**

Open `configs/tasks/medagentbench.yaml`. The current content is:

```yaml
default:
  module: src.server.tasks.medagentbench.MedAgentBench
  parameters:
    concurrency: 1
    max_round: 8
    fhir_api_base: "http://localhost:8080/fhir/"

medagentbench-std:
  parameters:
    name: medagentbench-std
    data_file: "data/medagentbench/test_data_v2.json"
    func_file: "data/medagentbench/funcs_v1.json"
```

Append the V2 task variant. The `module` key at the named-entry level overrides the `default.module` via the config system's deep-merge:

```yaml
default:
  module: src.server.tasks.medagentbench.MedAgentBench
  parameters:
    concurrency: 1
    max_round: 8
    fhir_api_base: "http://localhost:8080/fhir/"

medagentbench-std:
  parameters:
    name: medagentbench-std
    data_file: "data/medagentbench/test_data_v2.json"
    func_file: "data/medagentbench/funcs_v1.json"

medagentbench-v2-std:
  module: src.server.tasks.medagentbench.medagentbench_v2.MedAgentBenchV2
  parameters:
    name: medagentbench-v2-std
    data_file: "data/medagentbench/test_data_v2.json"
    func_file: "data/medagentbench/funcs_v2.json"
```

- [ ] **Step 2: Add gpt-5.4-nano to configs/agents/api_agents.yaml**

Open `configs/agents/api_agents.yaml`. Add at the top (after `gpt-4o-mini`):

```yaml
gpt-5.4-nano:
    import: "./openai-chat.yaml"
    parameters:
        name: "gpt-5.4-nano"
        body:
            model: "gpt-5.4-nano"
```

- [ ] **Step 3: Create configs/assignments/v2.yaml**

Create `configs/assignments/v2.yaml`:

```yaml
import: definition.yaml

concurrency:
  task:
    medagentbench-v2-std: 20
  agent:
    gpt-5.4-nano: 10

assignments:
  - agent:
      - gpt-5.4-nano
    task:
      - medagentbench-v2-std

output: "outputs/MedAgentBenchv2"
```

- [ ] **Step 4: Verify the config system can parse the new task**

```bash
python3 -c "
from src.configs import ConfigLoader
cfg = ConfigLoader('configs/assignments/v2.yaml').config
print('assignments:', [(a.agent, a.task) for a in cfg.assignments])
print('output:', cfg.output)
task_def = cfg.definition.task['medagentbench-v2-std']
print('task module:', task_def.module)
print('func_file:', task_def.parameters.get('func_file'))
assert task_def.module == 'src.server.tasks.medagentbench.medagentbench_v2.MedAgentBenchV2'
assert task_def.parameters['func_file'] == 'data/medagentbench/funcs_v2.json'
print('OK')
"
```

Expected: prints assignments, output path, and `OK`.

- [ ] **Step 5: Verify the agent config**

```bash
python -m src.client.agent_test --config configs/agents/api_agents.yaml --agent gpt-5.4-nano
```

Expected: model responds without error.

- [ ] **Step 6: Commit**

```bash
git add configs/tasks/medagentbench.yaml \
        configs/agents/api_agents.yaml \
        configs/assignments/v2.yaml
git commit -m "config: add MedAgentBenchV2 task variant, gpt-5.4-nano agent, v2 assignment config"
```

---

## Task 5: Smoke Test End-to-End

Prerequisites: Docker running, FHIR server up, `OPENAI_API_KEY` set.

- [ ] **Step 1: Confirm FHIR server is running**

```bash
curl -s http://localhost:8080/fhir/metadata | python3 -c "import sys,json; d=json.load(sys.stdin); print('FHIR OK:', d.get('resourceType'))"
```

Expected: `FHIR OK: CapabilityStatement`

If not running: `docker run -p 8080:8080 medagentbench`

- [ ] **Step 2: Start the task server**

In a dedicated terminal:

```bash
python -m src.start_task -a
```

Wait ~1 minute until you see worker initialization messages. Leave this terminal running.

- [ ] **Step 3: Edit v2.yaml to run a single task (task2_1) for the smoke test**

Temporarily limit `test_data_v2.json` isn't easily filtered by index via the assigner, so instead edit `MedAgentBenchV2.get_indices()` isn't accessible without code changes. Use an alternative: edit `configs/assignments/v2.yaml` to cap concurrency at 1 and run, then check the first result. The assigner will run all 300 tasks, but you only need to wait for one output to appear.

Alternatively, test the class instantiation and calculator path directly:

```bash
python3 -c "
import json, asyncio
from src.server.tasks.medagentbench.medagentbench_v2 import MedAgentBenchV2, _safe_eval

# Test calculator
result = json.loads(_safe_eval('(datetime(2023,11,13) - datetime(1985,4,20)).days // 365'))
print('Age test:', result)
assert result['result'] == '38', f'Expected 38, got {result}'

# Test class instantiation (requires FHIR server)
task = MedAgentBenchV2(
    name='test',
    data_file='data/medagentbench/test_data_v2.json',
    func_file='data/medagentbench/funcs_v2.json',
    fhir_api_base='http://localhost:8080/fhir/',
    concurrency=1,
    max_round=8,
)
print('Loaded', len(task.data), 'cases and', len(task.funcs), 'functions')
assert len(task.funcs) == 11
assert len(task.data) == 300
print('OK')
"
```

Expected:
```
Age test: {'result': '38'}
Loaded 300 cases and 11 functions
OK
```

- [ ] **Step 4: Run the full benchmark**

In a second terminal (task server must be running from Step 2):

```bash
python -m src.assigner --config configs/assignments/v2.yaml
```

- [ ] **Step 5: Check results**

```bash
cat outputs/MedAgentBenchv2/gpt-5.4-nano/medagentbench-v2-std/overall.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('Success rate:', d.get('success rate'))
"
```

Expected: a numeric success rate (any value confirms the pipeline ran end-to-end without crashing).

- [ ] **Step 6: Final commit**

```bash
git add -u
git commit -m "feat: MedAgentBenchV2 standard harness complete"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| Create MedAgentBenchV2 subclass | Task 3 |
| Calculator intercept in start_sample | Task 3 |
| Safe eval with correct whitelist | Task 3 |
| V2 prompt with explanation guidance | Task 3 |
| calculate_overall uses eval_v2 | Task 3 |
| eval_v2.py routes to new_refsol | Task 2 |
| funcs_v2.json with 11 V2 tools | Task 1 |
| explanation field on all tools | Task 1 |
| medagentbench-v2-std task config | Task 4 |
| gpt-5.4-nano agent config | Task 4 |
| configs/assignments/v2.yaml | Task 4 |
| output path outputs/MedAgentBenchv2 | Task 4 |
| V1 files untouched | All tasks |

All spec requirements covered. ✓
