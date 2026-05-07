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
