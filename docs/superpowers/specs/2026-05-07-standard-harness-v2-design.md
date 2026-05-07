# Standard Harness V2 Design

**Date:** 2026-05-07
**Status:** Approved
**Branch:** main

---

## Context

The native Hermes harness runs MedAgentBenchV2 tasks with 12 typed tool calls and evaluates via `new_refsol.py`. The standard MedAgentBench harness (assigner → task workers) uses a text-based `GET/POST/FINISH` protocol with 9 tool descriptions from `funcs_v1.json` and evaluates via `refsol.py` (V1). This design adds a `MedAgentBenchV2` subclass so gpt-5.4-nano can be benchmarked through the standard harness with V2 tools and V2 evaluation, while keeping V1 completely untouched.

**Target agent:** gpt-5.4-nano via the existing OpenAI-compatible `HTTPAgent`.

---

## What Changes

Six changes total. V1 (`MedAgentBench`, `eval.py`, `funcs_v1.json`, `refsol.py`) is untouched.

| Action | File | What |
|---|---|---|
| Create | `data/medagentbench/funcs_v2.json` | 11-tool V2 descriptions (10 FHIR + calculator) |
| Create | `src/server/tasks/medagentbench/eval_v2.py` | Eval shim routing to `new_refsol.py` |
| Create | `src/server/tasks/medagentbench/medagentbench_v2.py` | `MedAgentBenchV2` subclass |
| Modify | `configs/tasks/medagentbench.yaml` | Add `medagentbench-v2-std` task variant |
| Modify | `configs/agents/api_agents.yaml` | Add `gpt-5.4-nano` entry |
| Modify | `configs/assignments/default.yaml` | Wire `gpt-5.4-nano` → `medagentbench-v2-std` |

---

## 1. Tool File — `data/medagentbench/funcs_v2.json`

11 tool entries. Each has the same JSON schema shape as `funcs_v1.json` (`name`, `description`, `parameters`). Changes from V1:

- **Names** use the format `"tool_name (METHOD {api_base}/Resource)"` so the model sees both the logical V2 name and the URL to emit.
- **Descriptions** updated to V2 wording (e.g. observation_search tells the agent to use friendly lab codes like `K`, `A1C`, not LOINC strings).
- **`explanation` field** added to every tool's `properties` and `required` list, matching the Hermes schema and reinforcing chain-of-thought.
- **`calculator` tool** added: `name: "calculator (GET {api_base}/calculator)"`, single `expression` parameter. The task class intercepts calls to this URL and evaluates them locally.
- `show_plot` omitted — never called in the most recent run, has no useful return value for text-based protocol.

The 10 FHIR tools map to the same FHIR endpoints as V1:

| V2 tool name | URL |
|---|---|
| patient_search | GET {api_base}/Patient |
| condition_search | GET {api_base}/Condition |
| observation_search | GET {api_base}/Observation |
| vitals_search | GET {api_base}/Observation |
| medication_request_search | GET {api_base}/MedicationRequest |
| procedure_search | GET {api_base}/Procedure |
| vitals_create | POST {api_base}/Observation |
| medication_request_create | POST {api_base}/MedicationRequest |
| service_request_create | POST {api_base}/ServiceRequest |
| calculator | GET {api_base}/calculator (intercepted locally) |

`finish` is not in the JSON — it is described in the prompt text as `FINISH([...])`, same as V1.

---

## 2. Evaluator Shim — `src/server/tasks/medagentbench/eval_v2.py`

Identical structure to `eval.py` but imports and calls `new_refsol` instead of `refsol`:

```python
import importlib

_new_refsol = None

def _get_new_refsol():
    global _new_refsol
    if _new_refsol is None:
        _new_refsol = importlib.import_module('src.server.tasks.medagentbench.new_refsol')
    return _new_refsol

def eval(case_data, results, fhir_api_base):
    task_id = case_data['id'].split('_')[0]
    grader_func = getattr(_get_new_refsol(), task_id)
    try:
        if grader_func(case_data, results, fhir_api_base) is True:
            return True
    except Exception as e:
        print(e)
        return False
```

`new_refsol` functions access `results.history` (list of `ChatHistoryItem` objects with `.role` and `.content`) and `results.result` (JSON string). The standard harness already stores history as `ChatHistoryItem` objects (`role="user"` for injected messages, `role="agent"` for model turns), so no adapter is needed.

---

## 3. Task Class — `src/server/tasks/medagentbench/medagentbench_v2.py`

Subclasses `MedAgentBench` from `__init__.py`. Three overrides:

### `__init__`
Calls `super().__init__()`, then replaces the startup `refsol` import check with a `new_refsol` import check. No other constructor changes.

### `start_sample`
Identical to parent except one intercept before `send_get_request`:

```python
if r.startswith('GET') and '/calculator' in r.split('?')[0]:
    expr = urllib.parse.parse_qs(urllib.parse.urlparse(r[3:].strip()).query).get('expression', [''])[0]
    result = _safe_eval(expr)
    session.inject({"role": "user", "content": f"Calculator result: {result}. Please call FINISH if you have completed all tasks."})
```

`_safe_eval` uses the same whitelist namespace as the Hermes `calculator` tool: `math`, `datetime`, `timedelta`, `timezone`, `Decimal`, `abs`, `round`, `int`, `float`, `str`, `len`, `min`, `max`, `sum`. Returns a JSON string `{"result": "..."}` or `{"error": "..."}`.

The prompt uses a V2-aligned variant:
- Replaces `Question:` with `Instruction:` in the label shown to the model
- Adds: *"Include an 'explanation' field on every tool call explaining why you are making it."*
- Adds: *"Use the calculator tool for date and numeric arithmetic."*

Everything else (GET handling, POST handling, FINISH parsing, round limits, error returns) is inherited unchanged from `MedAgentBench`.

### `calculate_overall`
Same structure as parent, calls `eval_v2.eval()` instead of `eval.eval()`.

---

## 4. Config Changes

### `configs/tasks/medagentbench.yaml`
Add a `medagentbench-v2-std` entry pointing to `MedAgentBenchV2` and `funcs_v2.json`:

```yaml
medagentbench-v2-std:
  module: src.server.tasks.medagentbench.medagentbench_v2.MedAgentBenchV2
  parameters:
    name: medagentbench-v2-std
    data_file: "data/medagentbench/test_data_v2.json"
    func_file: "data/medagentbench/funcs_v2.json"
```

### `configs/agents/api_agents.yaml`
Add `gpt-5.4-nano` following the same pattern as `gpt-4o-mini`:

```yaml
gpt-5.4-nano:
    import: "./openai-chat.yaml"
    parameters:
        name: "gpt-5.4-nano"
        body:
            model: "gpt-5.4-nano"
```

### `configs/assignments/default.yaml`
Add `gpt-5.4-nano` to concurrency and assignments:

```yaml
concurrency:
  agent:
    gpt-5.4-nano: 10
assignments:
  - agent:
      - gpt-5.4-nano
    task:
      - medagentbench-v2-std
```

---

## Data Flow

```
Assigner → TaskClient → Controller (port 5000) → Worker (ports 5001–5020)
                                                        ↕
                                              HTTPAgent → gpt-5.4-nano (OpenAI)
                                                        ↕
                                              MedAgentBenchV2.start_sample()
                                                  ├── GET /calculator → _safe_eval()
                                                  ├── GET /fhir/* → FHIR server (port 8080)
                                                  ├── POST /fhir/* → acknowledged in history
                                                  └── FINISH([...]) → TaskOutput
                                                        ↕
                                              eval_v2.eval() → new_refsol.task1–10()
```

---

## Verification

### Unit test
```bash
python -m pytest src/server/tasks/medagentbench/tests/ -v
```
Tests: eval_v2 imports correctly, eval_v2 routes to the right task function, MedAgentBenchV2 safe_eval handles arithmetic/date expressions/invalid input.

### Config parse test
```bash
python -m src.client.agent_test --config configs/agents/api_agents.yaml --agent gpt-5.4-nano
```

### Single-task smoke test (task2_1 — no POST, uses calculator)
```bash
python -m src.start_task -a  # terminal 1
python -m src.assigner        # terminal 2 (after editing default.yaml to limit to task2_1)
```
Check `outputs/MedAgentBenchv2/<agent>/medagentbench-v2-std/overall.json` for `success rate`.

---

## Out of Scope

- Any changes to V1 (`MedAgentBench`, `eval.py`, `funcs_v1.json`, `refsol.py`)
- Prompt tuning to fix specific failure modes (separate effort)
- `show_plot` support in the standard harness
- Parallelism or concurrency changes
