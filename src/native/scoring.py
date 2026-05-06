"""
Scoring bridge: construct a MedAgentBench TaskOutput from a NativeHarnessResult
and invoke the V2 refsol grader.

POST scoring strategy
---------------------
new_refsol.extract_posts() scans results.history for agent-role entries that
contain 'POST' followed by a 'POST request accepted' user entry. In native mode,
POST operations are executed as typed tool calls. This module synthesizes the
matching history entries from the tool_call_log so the scorer sees the expected
format.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_POST_TOOL_TO_RESOURCE: Dict[str, str] = {
    "vitals_create": "Observation",
    "medication_request_create": "MedicationRequest",
    "service_request_create": "ServiceRequest",
}

_POST_ACCEPTED_MSG = (
    "POST request accepted and executed successfully. "
    "Please call FINISH if you have got answers for all the questions "
    "and finished all the requested tasks"
)

_V2_TASK_FNS = None


def _get_v2_task_fns() -> Dict[str, Any]:
    global _V2_TASK_FNS
    if _V2_TASK_FNS is None:
        from src.server.tasks.medagentbench import new_refsol as nr
        _V2_TASK_FNS = {f"task{i}": getattr(nr, f"task{i}") for i in range(1, 11)}
    return _V2_TASK_FNS


def _extract_task_prefix(task_id: str) -> str:
    m = re.match(r"(task\d+)", task_id)
    return m.group(1) if m else ""


def build_task_output(
    harness_result: "NativeHarnessResult",
    fhir_base_url: str,
) -> "TaskOutput":
    from src.typings import TaskOutput, SampleStatus
    from src.typings.general import ChatHistoryItem

    if harness_result.timed_out:
        status = SampleStatus.TASK_LIMIT_REACHED
    elif harness_result.final_answer is not None:
        status = SampleStatus.COMPLETED
    elif harness_result.errors:
        status = SampleStatus.TASK_ERROR
    else:
        status = SampleStatus.TASK_LIMIT_REACHED

    history = _build_history(harness_result, fhir_base_url)

    return TaskOutput(
        status=status,
        result=harness_result.final_answer,
        history=history,
    )


def _build_history(
    harness_result: "NativeHarnessResult",
    fhir_base_url: str,
) -> List["ChatHistoryItem"]:
    from src.typings.general import ChatHistoryItem

    history: List[ChatHistoryItem] = []

    post_calls = [
        entry for entry in harness_result.tool_call_log
        if entry.get("tool") in _POST_TOOL_TO_RESOURCE
    ]
    post_idx = 0

    for msg in harness_result.transcript:
        role = msg.get("role", "")
        content = msg.get("content") or ""

        if role == "assistant":
            history.append(ChatHistoryItem(role="agent", content=content))
        elif role == "user":
            history.append(ChatHistoryItem(role="user", content=content))
        elif role == "tool":
            tool_name = msg.get("name", "")
            if tool_name in _POST_TOOL_TO_RESOURCE and post_idx < len(post_calls):
                entry = post_calls[post_idx]
                post_idx += 1
                _inject_post_pair(history, entry, fhir_base_url)

    while post_idx < len(post_calls):
        entry = post_calls[post_idx]
        post_idx += 1
        _inject_post_pair(history, entry, fhir_base_url)

    return history


def _inject_post_pair(
    history: List["ChatHistoryItem"],
    entry: Dict[str, Any],
    fhir_base_url: str,
) -> None:
    from src.typings.general import ChatHistoryItem

    tool_name = entry.get("tool", "")
    resource_type = _POST_TOOL_TO_RESOURCE.get(tool_name, "Unknown")
    url = fhir_base_url.rstrip("/") + "/" + resource_type

    args = entry.get("args", {})
    payload = args.get("resource", args)

    agent_content = f"POST {url}\n{json.dumps(payload)}"
    history.append(ChatHistoryItem(role="agent", content=agent_content))
    history.append(ChatHistoryItem(role="user", content=_POST_ACCEPTED_MSG))


def score_result(
    case_data: Dict[str, Any],
    harness_result: "NativeHarnessResult",
    fhir_base_url: str,
) -> Dict[str, Any]:
    task_id = case_data.get("id", "unknown")
    task_output = build_task_output(harness_result, fhir_base_url)

    success = False
    error_msg = None
    try:
        prefix = _extract_task_prefix(task_id)
        task_fn = _get_v2_task_fns().get(prefix)
        if task_fn is None:
            raise ValueError(f"No V2 evaluator for task prefix '{prefix}'")
        success = bool(task_fn(case_data, task_output, fhir_base_url))
    except Exception as exc:
        error_msg = f"scorer error: {exc}"
        logger.warning("scoring failed for %s: %s", task_id, exc)

    return {
        "task_id": task_id,
        "success": success,
        "error": error_msg,
        "answer_parse_status": harness_result.answer_parse_status,
        "fhir_call_count": harness_result.fhir_call_count,
        "timed_out": harness_result.timed_out,
        "latency_seconds": round(harness_result.latency_seconds, 3),
    }
