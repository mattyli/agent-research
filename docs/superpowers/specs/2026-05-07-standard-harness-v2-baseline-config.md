# Standard Harness V2 — Comparable Baseline Config

**Date:** 2026-05-07
**Status:** Approved

---

## Goal

Align the standard assigner harness (`MedAgentBenchV2`) with the Hermes run `20260506_214159_361342` so results are directly comparable. Two config values differ from the Hermes run; both are corrected here.

---

## Changes

| File | Field | Before | After | Reason |
|---|---|---|---|---|
| `configs/tasks/medagentbench.yaml` | `max_round` on `medagentbench-v2-std` | inherited 8 | 12 | match Hermes `max_iterations: 12` |
| `configs/agents/api_agents.yaml` | `max_completion_tokens` on `gpt-5.4-nano` | inherited 2048 | 4096 | match Hermes `max_tokens: 4096` |

---

## Known Methodological Difference

The Hermes run used `memory_mode: persistent_eval` — the model had a populated memory store from prior warmup runs. The standard harness has no memory. This difference cannot be eliminated and should be noted when comparing success rates.

---

## Out of Scope

- Any changes to `MedAgentBenchV2`, `eval_v2.py`, `funcs_v2.json`, or `new_refsol.py`
- Changing V1 config
- Adding memory to the standard harness
