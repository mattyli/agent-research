# Hermes Native Harness — Smoke Test Design

**Date:** 2026-05-05  
**Status:** Approved  
**Scope:** Single-task end-to-end run (task1_1) through the full native stack

---

## Goal

Verify that the Hermes harness can run a complete MedAgentBench task end-to-end:
driver → runner → Hermes AIAgent → FHIR tool calls → fhir_finish → scoring → artifacts.

This is a smoke test first; full 300-task benchmark runs come after the loop is confirmed working.

---

## Changes

### 1. `src/native/driver.py` — inject `fhir_base_url` and `_artifact_dir` into model_config

**Problem:** `run_one_task` passes `self.config.model.dict()` as `model_config`, but
`ModelConfig` has no `fhir_base_url` or `_artifact_dir`. The runner reads both via
`.get()` with silent fallbacks, so the configured FHIR URL is ignored and tool-call
JSONL files land in `/tmp` instead of the run's output directory.

**Fix:** In `run_one_task`, build a merged dict before calling `setup_task`:

```python
model_config = {
    **self.config.model.dict(),
    "fhir_base_url": self.config.benchmark.fhir_base_url,
    "_artifact_dir": str(artifact_dir),
}
```

No interface changes; the runner already reads these keys.

### 2. `src/native/__main__.py` — CLI entrypoint

**Problem:** There is no way to invoke the driver without writing a custom script.

**New file** that accepts:
- `--config <path>` — path to a NativeRunConfig YAML (default: `configs/native/hermes_disabled_memory.yaml`)
- `--task-id <id>` — restrict the run to a single task (optional; omit for all tasks)
- `--log-level <level>` — logging verbosity (default: `INFO`)

Invocation:
```bash
/Users/02matt/.hermes/hermes-agent/venv/bin/python3 -m src.native \
  --config configs/native/hermes_disabled_memory.yaml \
  --task-id task1_1
```

### 3. `src/native/driver.py` — refsol presence check at startup

**Problem:** If `refsol.py` is missing, the driver crashes mid-run inside `score_result`
with an opaque `ImportError`.

**Fix:** In `NativeBenchDriver.__init__`, attempt to import `refsol` and raise a clear
`RuntimeError` immediately if it is absent, pointing to the Box download.

---

## Data Flow (smoke test)

```
python -m src.native --config ... --task-id task1_1
  → NativeBenchDriver.__init__
      → refsol presence check (fail fast if missing)
      → load test_data_v2.json, funcs_v1.json
  → run_one_task(case_data)
      → artifact_dir = outputs/native/runs/{run_id}/tasks/task1_1/
      → merged model_config (adds fhir_base_url + _artifact_dir)
      → HermesNativeRunner.setup_task(...)
      → HermesNativeRunner.run()
          → register_fhir_toolset()             # idempotent
          → TaskContext registered by hermes_task_id
          → AIAgent(enabled_toolsets=["fhir-medagent"], skip_memory=True, ...)
          → agent.run_conversation(prompt, task_id=hermes_task_id)
              ↕  fhir_* tool calls → TaskContext → FHIR HTTP → localhost:8080
              →  fhir_finish sets ctx.finished + ctx.final_answer
          → returns NativeHarnessResult
      → teardown() — clears TaskContext
      → score_result(case_data, result, fhir_base_url)
          → build_task_output() → refsol grader
      → write artifacts:
          task_metadata.json, score.json, task_summary.json,
          normalized_trajectory.jsonl, fhir_tool_calls.jsonl
  → summary.json + aggregate_metrics.csv at run root
```

---

## Error Handling

| Failure | Behaviour |
|---|---|
| `refsol.py` missing | `RuntimeError` at driver init with download instructions |
| FHIR server down | Tool handlers return JSON error; agent sees it as a tool response; harness times out if agent loops |
| Hermes API error | `run_conversation` returns `completed=False` + `error` key; captured in `result.errors` |
| Agent never calls `fhir_finish` | `ctx.finished` stays False; `answer_parse_status = "missing"`; timeout enforced by driver's `ThreadPoolExecutor` |

---

## Prerequisites

- FHIR server running: `docker run -p 8080:8080 medagentbench`
- `refsol.py` present at `src/server/tasks/medagentbench/refsol.py`
- Hermes Python interpreter: `/Users/02matt/.hermes/hermes-agent/venv/bin/python3`
- `ANTHROPIC_API_KEY` set in environment (used by Hermes for `anthropic/claude-opus-4-5`)

---

## Files Changed

| File | Change |
|---|---|
| `src/native/driver.py` | Merge `fhir_base_url` + `_artifact_dir` into model_config; add refsol check |
| `src/native/__main__.py` | New — CLI entrypoint with `--config`, `--task-id`, `--log-level` |

---

## Out of Scope

- Memory modes (`task_local`, `warmup_frozen`, `persistent_eval`) — addressed in a later phase
- Full 300-task benchmark parallelism
- Any changes to the Hermes AIAgent or FHIR server
