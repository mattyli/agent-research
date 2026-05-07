# Standard Harness V2 Baseline Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align `medagentbench-v2-std` / `gpt-5.4-nano` config with the Hermes run `20260506_214159_361342` so results are directly comparable.

**Architecture:** Two config-only edits — `max_round: 12` on the task variant and `max_completion_tokens: 4096` on the agent. No code changes.

**Tech Stack:** Python 3.9, PyYAML, existing assigner harness

---

## File Map

| Action | File | Change |
|---|---|---|
| Modify | `configs/tasks/medagentbench.yaml` | Add `max_round: 12` under `medagentbench-v2-std.parameters` |
| Modify | `configs/agents/api_agents.yaml` | Add `max_completion_tokens: 4096` under `gpt-5.4-nano.parameters.body` |

---

## Task 1: Set max_round: 12 on medagentbench-v2-std

**Files:**
- Modify: `configs/tasks/medagentbench.yaml`

- [ ] **Step 1: Edit the task config**

Open `configs/tasks/medagentbench.yaml`. The current `medagentbench-v2-std` block is:

```yaml
medagentbench-v2-std:
  module: src.server.tasks.medagentbench.medagentbench_v2.MedAgentBenchV2
  parameters:
    name: medagentbench-v2-std
    data_file: "data/medagentbench/test_data_v2.json"
    func_file: "data/medagentbench/funcs_v2.json"
```

Add `max_round: 12` so it becomes:

```yaml
medagentbench-v2-std:
  module: src.server.tasks.medagentbench.medagentbench_v2.MedAgentBenchV2
  parameters:
    name: medagentbench-v2-std
    data_file: "data/medagentbench/test_data_v2.json"
    func_file: "data/medagentbench/funcs_v2.json"
    max_round: 12
```

- [ ] **Step 2: Verify the config parses and the value is picked up**

```bash
python3 -c "
from src.configs import ConfigLoader
cfg = ConfigLoader('configs/assignments/v2.yaml').config
task = cfg.definition.task['medagentbench-v2-std']
assert task.parameters['max_round'] == 12, f\"Got {task.parameters.get('max_round')}\"
print('max_round OK:', task.parameters['max_round'])
"
```

Expected: `max_round OK: 12`

- [ ] **Step 3: Commit**

```bash
git add configs/tasks/medagentbench.yaml
git commit -m "config: set max_round 12 on medagentbench-v2-std for Hermes parity"
```

---

## Task 2: Set max_completion_tokens: 4096 on gpt-5.4-nano

**Files:**
- Modify: `configs/agents/api_agents.yaml`

- [ ] **Step 1: Edit the agent config**

Open `configs/agents/api_agents.yaml`. The current `gpt-5.4-nano` block is:

```yaml
gpt-5.4-nano:
    import: "./openai-chat.yaml"
    parameters:
        name: "gpt-5.4-nano"
        body:
            model: "gpt-5.4-nano"
```

Add `max_completion_tokens: 4096` so it becomes:

```yaml
gpt-5.4-nano:
    import: "./openai-chat.yaml"
    parameters:
        name: "gpt-5.4-nano"
        body:
            model: "gpt-5.4-nano"
            max_completion_tokens: 4096
```

- [ ] **Step 2: Verify the config parses and the value is present**

```bash
python3 -c "
from src.configs import ConfigLoader
cfg = ConfigLoader('configs/agents/api_agents.yaml').config
agent = cfg['gpt-5.4-nano']
body = agent.parameters['body']
assert body.get('max_completion_tokens') == 4096, f\"Got {body.get('max_completion_tokens')}\"
print('max_completion_tokens OK:', body['max_completion_tokens'])
"
```

Expected: `max_completion_tokens OK: 4096`

> If the ConfigLoader interface doesn't support loading agent YAMLs directly like this, verify by running the agent test instead (Step 3 covers this).

- [ ] **Step 3: Test the agent connects successfully**

```bash
python -m src.client.agent_test --config configs/agents/api_agents.yaml --agent gpt-5.4-nano
```

Expected: model responds without error (confirms the merged body is valid).

- [ ] **Step 4: Commit**

```bash
git add configs/agents/api_agents.yaml
git commit -m "config: set max_completion_tokens 4096 on gpt-5.4-nano for Hermes parity"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| `max_round: 12` on `medagentbench-v2-std` | Task 1 |
| `max_completion_tokens: 4096` on `gpt-5.4-nano` | Task 2 |
| V1 and other agents untouched | Both tasks (edits are scoped to named entries only) |

All requirements covered. ✓

**Placeholder scan:** No TBDs, no vague steps. ✓

**Type consistency:** Config-only plan, no type surface. ✓
