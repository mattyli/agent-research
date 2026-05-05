# Hermes Native Harness — Smoke Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two silent bugs in the driver and add a CLI entrypoint so the Hermes harness can be invoked for a single-task end-to-end smoke test.

**Architecture:** Two targeted edits to `src/native/driver.py` (inject missing config keys into model_config; add fail-fast refsol check), plus a new `src/native/__main__.py` that wraps the driver with argparse. No interface changes — the runner's `.get()` calls already handle the injected keys.

**Tech Stack:** Python 3.11 (Hermes venv), pydantic v2, argparse, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/native/driver.py` | Modify | Inject `fhir_base_url` + `_artifact_dir` into model_config; add refsol check |
| `src/native/__main__.py` | Create | CLI entrypoint (`--config`, `--task-id`, `--log-level`) |
| `src/native/tests/test_driver.py` | Modify | Add tests for model_config injection and refsol check |

---

## Task 1: Fix model_config injection — `fhir_base_url` and `_artifact_dir`

**Files:**
- Modify: `src/native/driver.py` (method `run_one_task`, lines ~151–166)
- Modify: `src/native/tests/test_driver.py` (class `TestRunOneTask`)

The driver currently passes `self.config.model.dict()` as `model_config`. `ModelConfig` has no
`fhir_base_url` or `_artifact_dir`, so the runner silently falls back to hardcoded values.
Fix: build a merged dict before calling `setup_task`.

Also extend `MockRunner.setup_task` to capture `model_config` in `_last_setup_args` so the
new tests can inspect it. Existing tests only read `task_metadata` and `task_prompt` from
`_last_setup_args`, so this is additive and safe.

- [ ] **Step 1: Write failing tests**

Open `src/native/tests/test_driver.py`. Inside `TestRunOneTask`, add two new test methods.
Also update `MockRunner.setup_task` (in the same file) to capture `model_config`:

```python
# In MockRunner.setup_task, add model_config to _last_setup_args:
def setup_task(self, task_metadata, task_prompt, tool_specs,
               model_config, runtime_constraints, memory_config):
    self.setup_called = True
    self._last_setup_args = {
        "task_metadata": task_metadata,
        "task_prompt": task_prompt,
        "model_config": model_config,   # ADD THIS LINE
    }
```

Add a module-level constant near the top of `src/native/tests/test_driver.py`
(after the existing `_make_harness_result` helper, before any class definitions):

```python
_SCORE_STUB = {
    "task_id": "task1_1", "success": True, "error": None,
    "answer_parse_status": "ok", "fhir_call_count": 2,
    "timed_out": False, "latency_seconds": 5.0,
}
```

Then add these two tests inside `TestRunOneTask`:

```python
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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
/opt/anaconda3/envs/medagentbench/bin/python3 -m pytest \
    src/native/tests/test_driver.py::TestRunOneTask::test_model_config_passes_fhir_base_url \
    src/native/tests/test_driver.py::TestRunOneTask::test_model_config_passes_artifact_dir \
    -v
```

Expected: **FAIL** — `KeyError: 'fhir_base_url'` (key not present in model_config yet).

- [ ] **Step 3: Fix `run_one_task` in `src/native/driver.py`**

Find the `run_one_task` method. Replace the `runner.setup_task(...)` call so it uses a merged
model_config. The full updated block (from `runner = self._instantiate_runner()` through the
`setup_task` call) should read:

```python
runner = self._instantiate_runner()

model_config = {
    **self.config.model.dict(),
    "fhir_base_url": self.config.benchmark.fhir_base_url,
    "_artifact_dir": str(artifact_dir),
}

runner.setup_task(
    task_metadata=case_data,
    task_prompt=self._build_prompt(case_data),
    tool_specs=self._funcs,
    model_config=model_config,
    runtime_constraints=self.config.runtime.dict(),
    memory_config=self.config.memory.dict(),
)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
/opt/anaconda3/envs/medagentbench/bin/python3 -m pytest \
    src/native/tests/test_driver.py::TestRunOneTask::test_model_config_passes_fhir_base_url \
    src/native/tests/test_driver.py::TestRunOneTask::test_model_config_passes_artifact_dir \
    -v
```

Expected: **PASS** both tests.

- [ ] **Step 5: Run full unit suite — verify no regressions**

```bash
/opt/anaconda3/envs/medagentbench/bin/python3 -m pytest \
    src/native/tests/ -v -m "not integration"
```

Expected: **61 passed, 21 deselected** (59 existing + 2 new).

- [ ] **Step 6: Commit**

```bash
git add src/native/driver.py src/native/tests/test_driver.py
git commit -m "fix: inject fhir_base_url and _artifact_dir into model_config in driver"
```

---

## Task 2: Add fail-fast refsol check at driver startup

**Files:**
- Modify: `src/native/driver.py` (method `__init__`, new private method `_check_refsol`)
- Modify: `src/native/tests/test_driver.py` (new test class `TestRefsol`)

If `refsol.py` is missing the driver currently crashes mid-run with an opaque `ModuleNotFoundError`
inside `eval.py`. Add a check in `__init__` that fails immediately with an actionable message.

- [ ] **Step 1: Write failing test**

Add a new test class at the bottom of `src/native/tests/test_driver.py`:

```python
class TestRefsol:
    def test_missing_refsol_raises_runtime_error(self, tmp_path):
        data_file = tmp_path / "test_data.json"
        func_file = tmp_path / "funcs.json"
        data_file.write_text(json.dumps([_make_case()]))
        func_file.write_text(json.dumps([_make_func_spec()]))

        config = NativeRunConfig.parse_obj({
            "benchmark": {
                "medagentbench_path": str(tmp_path),
                "data_file": str(data_file),
                "func_file": str(func_file),
                "fhir_base_url": "http://localhost:8080/fhir/",
            },
            "logging": {"output_dir": str(tmp_path / "outputs")},
        })

        with patch(
            "src.native.driver.importlib.import_module",
            side_effect=ModuleNotFoundError("refsol"),
        ):
            with pytest.raises(RuntimeError, match="refsol.py not found"):
                NativeBenchDriver(config)
```

No new imports needed in the test file — the patch target is a string path.

- [ ] **Step 2: Run test — verify it fails**

```bash
/opt/anaconda3/envs/medagentbench/bin/python3 -m pytest \
    src/native/tests/test_driver.py::TestRefsol::test_missing_refsol_raises_runtime_error \
    -v
```

Expected: **FAIL** — `Failed: DID NOT RAISE <class 'RuntimeError'>`.

- [ ] **Step 3: Add `_check_refsol` to driver**

In `src/native/driver.py`, add `import importlib` to the top-level imports (after the existing
stdlib imports). Then add `_check_refsol` as a private method and call it from `__init__`:

```python
# Add to top-level imports:
import importlib

# Add to NativeBenchDriver.__init__, before _load_data():
def __init__(self, config: "NativeRunConfig"):
    self.config = config
    self._check_refsol()       # ADD THIS LINE
    self._load_data()
    self._setup_output_dir()
    self._runner: Optional[NativeHarnessRunner] = None

# Add as a new private method (place after _setup_output_dir):
def _check_refsol(self) -> None:
    try:
        importlib.import_module("src.server.tasks.medagentbench.refsol")
    except ModuleNotFoundError:
        raise RuntimeError(
            "refsol.py not found at src/server/tasks/medagentbench/refsol.py. "
            "Download it from the Stanford Medicine Box link in the README and "
            "place it at that path before running the benchmark."
        )
```

- [ ] **Step 4: Run test — verify it passes**

```bash
/opt/anaconda3/envs/medagentbench/bin/python3 -m pytest \
    src/native/tests/test_driver.py::TestRefsol::test_missing_refsol_raises_runtime_error \
    -v
```

Expected: **PASS**.

- [ ] **Step 5: Run full unit suite — verify no regressions**

```bash
/opt/anaconda3/envs/medagentbench/bin/python3 -m pytest \
    src/native/tests/ -v -m "not integration"
```

Expected: **62 passed, 21 deselected**.

Note: `_make_driver_with_mock_data` uses `NativeBenchDriver.__new__` (bypasses `__init__`),
so existing tests are unaffected by the new check.

- [ ] **Step 6: Commit**

```bash
git add src/native/driver.py src/native/tests/test_driver.py
git commit -m "fix: fail fast with clear error when refsol.py is missing"
```

---

## Task 3: Create `src/native/__main__.py`

**Files:**
- Create: `src/native/__main__.py`

This is the CLI entrypoint. No unit tests — argument parsing is trivially verified by running
the module with `--help`. The real verification is the smoke test in Task 4.

- [ ] **Step 1: Create `src/native/__main__.py`**

```python
"""
CLI entrypoint for the native benchmark harness.

Usage:
    /Users/02matt/.hermes/hermes-agent/venv/bin/python3 -m src.native \
        --config configs/native/hermes_disabled_memory.yaml \
        --task-id task1_1
"""
import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.native.experiments.config_schema import NativeRunConfig
from src.native.driver import NativeBenchDriver


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MedAgentBench tasks through the native Hermes harness.",
    )
    parser.add_argument(
        "--config",
        default="configs/native/hermes_disabled_memory.yaml",
        help="Path to NativeRunConfig YAML (default: configs/native/hermes_disabled_memory.yaml).",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        metavar="ID",
        help="Run a single task by ID, e.g. task1_1. Omit to run all tasks in the config.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = NativeRunConfig.from_yaml(args.config)

    if args.task_id:
        config.benchmark.task_ids = [args.task_id]

    driver = NativeBenchDriver(config)
    summary = driver.run()

    print(
        f"\nRun complete: {summary['success_rate']:.1%} "
        f"({summary['successes']}/{summary['total']}) tasks succeeded"
    )
    print(f"Output directory: {driver._output_root}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify `--help` works**

```bash
/Users/02matt/.hermes/hermes-agent/venv/bin/python3 -m src.native --help
```

Expected output (approximately):

```
usage: __main__.py [-h] [--config CONFIG] [--task-id ID] [--log-level {DEBUG,INFO,WARNING,ERROR}]

Run MedAgentBench tasks through the native Hermes harness.

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to NativeRunConfig YAML ...
  --task-id ID          Run a single task by ID ...
  --log-level {DEBUG,INFO,WARNING,ERROR}
```

- [ ] **Step 3: Commit**

```bash
git add src/native/__main__.py
git commit -m "feat: add src/native/__main__.py CLI entrypoint"
```

---

## Task 4: Smoke test — run task1_1 end-to-end

This task is manual verification. No code changes.

**Prerequisites (do these before running):**

1. Start the FHIR server:
   ```bash
   docker run -p 8080:8080 medagentbench
   ```
   Wait ~30 seconds for it to be ready, then verify:
   ```bash
   curl -s "http://localhost:8080/fhir/Patient?_count=1&_format=json" | python3 -m json.tool | head -5
   ```
   Expected: JSON with `"resourceType": "Bundle"`.

2. Verify `refsol.py` is present:
   ```bash
   ls src/server/tasks/medagentbench/refsol.py
   ```
   Expected: file listed. If missing, download from Stanford Medicine Box (see README).

3. Set the Anthropic API key (used by Hermes for `anthropic/claude-opus-4-5`):
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

- [ ] **Step 1: Run the smoke test**

Run from the repo root:

```bash
/Users/02matt/.hermes/hermes-agent/venv/bin/python3 -m src.native \
    --config configs/native/hermes_disabled_memory.yaml \
    --task-id task1_1 \
    --log-level DEBUG
```

- [ ] **Step 2: Verify artifacts were written**

```bash
ls outputs/native/runs/
# Note the run_id directory (timestamp + hex suffix)

RUN_DIR=$(ls -td outputs/native/runs/*/ | head -1)
ls "$RUN_DIR/tasks/task1_1/"
```

Expected files:
```
task_metadata.json
score.json
task_summary.json
normalized_trajectory.jsonl
fhir_tool_calls.jsonl
```

- [ ] **Step 3: Check the score**

```bash
cat "$RUN_DIR/tasks/task1_1/score.json"
```

Expected: JSON with a `"success"` key (true or false) and no Python traceback in the terminal.
A `false` score is acceptable for the smoke test — the goal is confirming the full loop runs
without crashing, not that the agent gets the right answer.

- [ ] **Step 4: Check the tool call log**

```bash
cat "$RUN_DIR/tasks/task1_1/fhir_tool_calls.jsonl"
```

Expected: one or more JSONL lines, each with `"tool"` (e.g. `fhir_patient_search`) and
`"response_preview"`. If this file is empty, the agent made no FHIR calls — check the
transcript for errors.

- [ ] **Step 5: Check the transcript**

```bash
cat "$RUN_DIR/tasks/task1_1/normalized_trajectory.jsonl"
```

Expected: alternating user/assistant/tool role entries showing the agent's reasoning and
tool calls. If the agent called `fhir_finish`, there should be a tool entry with
`"name": "fhir_finish"`.

- [ ] **Step 6: If anything fails, check errors.log**

```bash
cat "$RUN_DIR/tasks/task1_1/errors.log" 2>/dev/null || echo "(no errors.log — no errors captured)"
```

- [ ] **Step 7: Commit summary**

No code changes in this task. If you had to fix a bug discovered during the smoke test,
commit that fix with a descriptive message before marking this task done.
