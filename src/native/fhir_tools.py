"""
Restricted FHIR tool layer for MedAgentBench V2 native harness.

Registers 12 tools under the "fhir-medagent" Hermes toolset:
  - patient_search
  - condition_search
  - observation_search
  - vitals_search
  - vitals_create
  - medication_request_search
  - medication_request_create
  - procedure_search
  - service_request_create
  - finish
  - calculator
  - show_plot

Call register_fhir_toolset() once per process before creating any AIAgent.
Wrap each task with set_task_context / clear_task_context.
"""

import json
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import decimal
import math
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

HERMES_AGENT_PATH = Path.home() / ".hermes" / "hermes-agent"
TOOLSET_NAME = "fhir-medagent"

_registered = False
_register_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Per-task context
# ---------------------------------------------------------------------------

@dataclass
class TaskContext:
    """
    Shared mutable state for one benchmark task execution.
    Passed by reference; tool handlers update it through the module-level
    registry keyed by Hermes task_id.
    """
    fhir_base_url: str
    max_calls: int
    output_dir: Path          # directory where fhir_tool_calls.jsonl is written
    hermes_task_id: str       # matches the task_id Hermes passes to handlers

    call_count: int = 0
    finished: bool = False
    final_answer: Optional[str] = None  # JSON-serialized list from fhir_finish
    budget_exceeded: bool = False
    tool_call_log: List[Dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def try_consume(self) -> bool:
        """Consume one call slot. Returns False (and sets budget_exceeded) if at limit."""
        with self._lock:
            if self.call_count >= self.max_calls:
                self.budget_exceeded = True
                return False
            self.call_count += 1
            return True

    def log_call(self, tool: str, args: dict, response: str, counted: bool) -> None:
        with self._lock:
            seq = len(self.tool_call_log)
            entry = {
                "seq": seq,
                "timestamp": time.time(),
                "tool": tool,
                "args": args,
                "response_preview": response[:500],
                "counted": counted,
            }
            self.tool_call_log.append(entry)
        log_path = self.output_dir / "fhir_tool_calls.jsonl"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.warning("fhir_tools: failed to write tool call log: %s", exc)


_ctx_lock = threading.Lock()
_active_contexts: Dict[str, TaskContext] = {}


def set_task_context(task_id: str, ctx: TaskContext) -> None:
    with _ctx_lock:
        _active_contexts[task_id] = ctx


def get_task_context(task_id: str) -> Optional[TaskContext]:
    with _ctx_lock:
        return _active_contexts.get(task_id)


def clear_task_context(task_id: str) -> None:
    with _ctx_lock:
        _active_contexts.pop(task_id, None)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ctx_for(kwargs: dict) -> Optional[TaskContext]:
    """Look up the active TaskContext from the Hermes task_id kwarg."""
    task_id = kwargs.get("task_id")
    if not task_id:
        return None
    return get_task_context(task_id)


def _fhir_get(base_url: str, resource: str, params: Dict[str, Any]) -> str:
    """Execute a FHIR GET and return a JSON string (data or error)."""
    params = {k: v for k, v in params.items() if v is not None}
    params.setdefault("_format", "json")
    params.setdefault("_count", 5000)
    url = base_url.rstrip("/") + "/" + resource
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return json.dumps(resp.json())
    except requests.HTTPError as exc:
        return json.dumps({"error": f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _fhir_post(base_url: str, resource: str, payload: dict) -> str:
    """Execute a FHIR POST and return a JSON string (data or error)."""
    url = base_url.rstrip("/") + "/" + resource
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return json.dumps({"status": "created", "response": resp.json()})
    except requests.HTTPError as exc:
        return json.dumps({"error": f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


_BUDGET_ERROR = json.dumps({
    "error": (
        "FHIR call budget exhausted. No more FHIR calls are allowed. "
        "Call finish immediately with your best current answer."
    )
})


def _guard(ctx: Optional[TaskContext], tool: str, args: dict, counted: bool = True):
    """
    Check budget and return (allowed: bool, error_json: str | None).
    Always logs the attempt regardless of whether it is allowed.
    """
    if ctx is None:
        return True, None  # no context → allow (test/standalone mode)
    if counted and not ctx.try_consume():
        ctx.log_call(tool, args, _BUDGET_ERROR, counted=False)
        return False, _BUDGET_ERROR
    return True, None


def _make_handler(tool_name: str, counted: bool, impl):
    """
    Return a Hermes-compatible handler: f(args: dict, **kw) -> str.
    Handles budget check, logging, and delegates to impl(ctx, args) -> str.
    """
    def handler(args: dict, **kw) -> str:
        ctx = _ctx_for(kw)
        allowed, err = _guard(ctx, tool_name, args, counted=counted)
        if not allowed:
            return err
        result = impl(ctx, args)
        if ctx is not None:
            ctx.log_call(tool_name, args, result, counted=counted)
        return result
    handler.__name__ = f"handle_{tool_name}"
    return handler


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _impl_patient_search(ctx: Optional[TaskContext], args: dict) -> str:
    base = ctx.fhir_base_url if ctx else args.get("_fhir_base_url", "")
    return _fhir_get(base, "Patient", {
        "identifier": args.get("identifier"),
        "name": args.get("name"),
        "birthdate": args.get("birthdate"),
    })


def _impl_condition_search(ctx: Optional[TaskContext], args: dict) -> str:
    base = ctx.fhir_base_url if ctx else ""
    return _fhir_get(base, "Condition", {
        "patient": args.get("patient"),
        "category": args.get("category"),
    })


def _impl_observation_labs_search(ctx: Optional[TaskContext], args: dict) -> str:
    base = ctx.fhir_base_url if ctx else ""
    return _fhir_get(base, "Observation", {
        "patient": args.get("patient"),
        "code": args.get("code"),
        "date": args.get("date"),
    })


def _impl_observation_vitals_search(ctx: Optional[TaskContext], args: dict) -> str:
    base = ctx.fhir_base_url if ctx else ""
    return _fhir_get(base, "Observation", {
        "patient": args.get("patient"),
        "category": args.get("category", "vital-signs"),
        "date": args.get("date"),
    })


def _impl_observation_vitals_create(ctx: Optional[TaskContext], args: dict) -> str:
    base = ctx.fhir_base_url if ctx else ""
    resource = args.get("resource", args)
    return _fhir_post(base, "Observation", resource)


def _impl_medication_request_search(ctx: Optional[TaskContext], args: dict) -> str:
    base = ctx.fhir_base_url if ctx else ""
    return _fhir_get(base, "MedicationRequest", {
        "patient": args.get("patient"),
    })


def _impl_medication_request_create(ctx: Optional[TaskContext], args: dict) -> str:
    base = ctx.fhir_base_url if ctx else ""
    resource = args.get("resource", args)
    return _fhir_post(base, "MedicationRequest", resource)


def _impl_procedure_search(ctx: Optional[TaskContext], args: dict) -> str:
    base = ctx.fhir_base_url if ctx else ""
    return _fhir_get(base, "Procedure", {
        "patient": args.get("patient"),
    })


def _impl_service_request_create(ctx: Optional[TaskContext], args: dict) -> str:
    base = ctx.fhir_base_url if ctx else ""
    resource = args.get("resource", args)
    return _fhir_post(base, "ServiceRequest", resource)


def _impl_finish(ctx: Optional[TaskContext], args: dict) -> str:
    answer = args.get("answer", [])
    serialized = json.dumps(answer)
    if ctx is not None:
        with ctx._lock:
            ctx.finished = True
            ctx.final_answer = serialized
    return json.dumps({"status": "finished", "answer": answer})


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


def _impl_calculator(ctx: Optional[TaskContext], args: dict) -> str:
    expr = str(args.get("expression", ""))
    try:
        result = eval(expr, _SAFE_CALC_GLOBALS, {})  # noqa: S307
        return json.dumps({"result": str(result)})
    except Exception as exc:
        return json.dumps({"error": f"calculator error: {exc}"})


def _impl_show_plot(ctx: Optional[TaskContext], args: dict) -> str:
    return json.dumps({
        "status": "recorded",
        "x": args.get("x", []),
        "y": args.get("y", []),
        "x_label": args.get("x_label", ""),
        "y_label": args.get("y_label", ""),
    })


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_FHIR_RESOURCE_SCHEMA = {
    "type": "object",
    "description": "Full FHIR resource JSON object to POST.",
    "additionalProperties": True,
}

_EXPLANATION_FIELD = {
    "explanation": {
        "type": "string",
        "description": "One sentence justifying this tool call before executing it.",
    }
}

_TOOLS: List[Dict] = [
    {
        "name": "patient_search",
        "counted": True,
        "impl": _impl_patient_search,
        "schema": {
            "description": (
                "Search for a FHIR Patient record. Returns matching patient resources "
                "including patient ID (used as the 'patient' parameter in other tools). "
                "Provide at least one of identifier, name, or birthdate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": "Patient MRN or other identifier.",
                    },
                    "name": {
                        "type": "string",
                        "description": (
                            "Patient name token (family or given). "
                            "Use a single token (e.g. 'Stafford'), not a full name."
                        ),
                    },
                    "birthdate": {
                        "type": "string",
                        "description": "Patient date of birth in YYYY-MM-DD format.",
                    },
                    **_EXPLANATION_FIELD,
                },
                "required": ["explanation"],
            },
        },
    },
    {
        "name": "condition_search",
        "counted": True,
        "impl": _impl_condition_search,
        "schema": {
            "description": (
                "Search the patient's problem list (Condition resources). "
                "Use category='problem-list-item' for the standard problem list."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient": {
                        "type": "string",
                        "description": "FHIR Patient resource ID.",
                    },
                    "category": {
                        "type": "string",
                        "description": "Condition category, typically 'problem-list-item'.",
                    },
                    **_EXPLANATION_FIELD,
                },
                "required": ["patient", "explanation"],
            },
        },
    },
    {
        "name": "observation_search",
        "counted": True,
        "impl": _impl_observation_labs_search,
        "schema": {
            "description": (
                "Search lab results (Observation resources). "
                "Provide a friendly lab code such as 'K', 'A1C', 'MG', 'GLU', 'NA' — "
                "not a LOINC string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient": {
                        "type": "string",
                        "description": "FHIR Patient resource ID.",
                    },
                    "code": {
                        "type": "string",
                        "description": "Lab code (e.g. 'K', 'A1C', 'MG', 'GLU', 'NA').",
                    },
                    "date": {
                        "type": "string",
                        "description": "Filter by date (YYYY-MM-DD) or date range (ge2023-01-01).",
                    },
                    **_EXPLANATION_FIELD,
                },
                "required": ["patient", "code", "explanation"],
            },
        },
    },
    {
        "name": "vitals_search",
        "counted": True,
        "impl": _impl_observation_vitals_search,
        "schema": {
            "description": (
                "Search vital sign observations (Observation resources with category vital-signs)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient": {
                        "type": "string",
                        "description": "FHIR Patient resource ID.",
                    },
                    "category": {
                        "type": "string",
                        "description": "Must be 'vital-signs'.",
                        "enum": ["vital-signs"],
                    },
                    "date": {
                        "type": "string",
                        "description": "Filter by date or date range.",
                    },
                    **_EXPLANATION_FIELD,
                },
                "required": ["patient", "category", "explanation"],
            },
        },
    },
    {
        "name": "vitals_create",
        "counted": True,
        "impl": _impl_observation_vitals_create,
        "schema": {
            "description": (
                "Create (POST) a vital signs Observation resource. "
                "The 'resource' field must be a valid FHIR Observation JSON with "
                "resourceType='Observation', category vital-signs, effectiveDateTime, "
                "status, code, valueString or valueQuantity, and subject reference."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "resource": {**_FHIR_RESOURCE_SCHEMA,
                                 "description": "FHIR Observation resource body."},
                    **_EXPLANATION_FIELD,
                },
                "required": ["resource", "explanation"],
            },
        },
    },
    {
        "name": "medication_request_search",
        "counted": True,
        "impl": _impl_medication_request_search,
        "schema": {
            "description": "Search MedicationRequest resources for a patient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient": {
                        "type": "string",
                        "description": "FHIR Patient resource ID.",
                    },
                    **_EXPLANATION_FIELD,
                },
                "required": ["patient", "explanation"],
            },
        },
    },
    {
        "name": "medication_request_create",
        "counted": True,
        "impl": _impl_medication_request_create,
        "schema": {
            "description": (
                "Create (POST) a MedicationRequest resource to order a medication. "
                "Must include resourceType, medicationCodeableConcept with NDC code, "
                "subject reference, status, intent, authoredOn, and dosageInstruction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "resource": {**_FHIR_RESOURCE_SCHEMA,
                                 "description": "FHIR MedicationRequest resource body."},
                    **_EXPLANATION_FIELD,
                },
                "required": ["resource", "explanation"],
            },
        },
    },
    {
        "name": "procedure_search",
        "counted": True,
        "impl": _impl_procedure_search,
        "schema": {
            "description": "Search Procedure resources for a patient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient": {
                        "type": "string",
                        "description": "FHIR Patient resource ID.",
                    },
                    **_EXPLANATION_FIELD,
                },
                "required": ["patient", "explanation"],
            },
        },
    },
    {
        "name": "service_request_create",
        "counted": True,
        "impl": _impl_service_request_create,
        "schema": {
            "description": (
                "Create (POST) a ServiceRequest resource to order a procedure or consult. "
                "Must include resourceType='ServiceRequest', code with SNOMED or LOINC coding, "
                "subject reference, status, intent, priority, and authoredOn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "resource": {**_FHIR_RESOURCE_SCHEMA,
                                 "description": "FHIR ServiceRequest resource body."},
                    **_EXPLANATION_FIELD,
                },
                "required": ["resource", "explanation"],
            },
        },
    },
    {
        "name": "finish",
        "counted": False,
        "impl": _impl_finish,
        "schema": {
            "description": (
                "Signal task completion and provide the final answer. "
                "Call this ONLY when you have gathered all information needed to answer "
                "every part of the question. The answer must be a JSON-serializable list "
                "matching the expected output format. Use the correct Python type for each "
                "element: int, float, str, or None."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "array",
                        "description": (
                            "Final answer list. Each element corresponds to one "
                            "part of the question."
                        ),
                        "items": {},
                    },
                    **_EXPLANATION_FIELD,
                },
                "required": ["answer", "explanation"],
            },
        },
    },
    {
        "name": "calculator",
        "counted": False,
        "impl": _impl_calculator,
        "schema": {
            "description": (
                "Evaluate a Python arithmetic expression safely. "
                "Supports math operators, datetime, timedelta, and Decimal. "
                "Use for date arithmetic (e.g. days since last procedure) and "
                "numeric calculations before calling finish."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "A Python expression to evaluate. "
                            "May use: arithmetic operators, datetime(), timedelta(), "
                            "math.*, abs(), round(), int(), float(), len(), min(), max(), sum()."
                        ),
                    },
                    **_EXPLANATION_FIELD,
                },
                "required": ["expression", "explanation"],
            },
        },
    },
    {
        "name": "show_plot",
        "counted": False,
        "impl": _impl_show_plot,
        "schema": {
            "description": (
                "Record a data series for visualization. "
                "Returns confirmation only — no rendering during benchmarking."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "array",
                        "description": "X-axis values.",
                        "items": {},
                    },
                    "y": {
                        "type": "array",
                        "description": "Y-axis values.",
                        "items": {},
                    },
                    "x_label": {"type": "string", "description": "X-axis label."},
                    "y_label": {"type": "string", "description": "Y-axis label."},
                    **_EXPLANATION_FIELD,
                },
                "required": ["x", "y", "x_label", "y_label", "explanation"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_fhir_toolset() -> None:
    """
    Register all FHIR tools with the Hermes tool registry under the
    'fhir-medagent' toolset. Safe to call multiple times (idempotent).

    Must be called before any AIAgent is instantiated.
    Adds HERMES_AGENT_PATH to sys.path if not already present.
    """
    global _registered
    with _register_lock:
        if _registered:
            return

        hermes_path = str(HERMES_AGENT_PATH)
        if hermes_path not in sys.path:
            sys.path.insert(0, hermes_path)

        try:
            from tools.registry import registry
            from toolsets import create_custom_toolset, TOOLSETS
        except ImportError as exc:
            raise RuntimeError(
                f"Cannot import Hermes registry from {HERMES_AGENT_PATH}. "
                f"Check that HERMES_AGENT_PATH is correct. Original error: {exc}"
            ) from exc

        tool_names = [t["name"] for t in _TOOLS]

        # Add to TOOLSETS so validate_toolset / resolve_toolset can find it
        if TOOLSET_NAME not in TOOLSETS:
            create_custom_toolset(
                name=TOOLSET_NAME,
                description="MedAgentBench restricted FHIR tools for native harness evaluation",
                tools=tool_names,
                includes=[],
            )

        for spec in _TOOLS:
            name = spec["name"]
            counted = spec["counted"]
            impl = spec["impl"]
            schema = spec["schema"]
            handler = _make_handler(name, counted, impl)
            registry.register(
                name=name,
                toolset=TOOLSET_NAME,
                schema=schema,
                handler=handler,
                emoji="🏥",
            )

        _registered = True
        logger.info(
            "fhir_tools: registered %d tools under toolset '%s'",
            len(_TOOLS),
            TOOLSET_NAME,
        )


def is_registered() -> bool:
    return _registered


def get_tool_names() -> List[str]:
    return [t["name"] for t in _TOOLS]
