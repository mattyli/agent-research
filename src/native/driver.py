"""
NativeBenchDriver: framework-agnostic orchestrator for native harness evaluation.

Usage:
    config = NativeRunConfig.from_yaml("configs/native/hermes_disabled_memory.yaml")
    driver = NativeBenchDriver(config)
    driver.run()
"""
import csv
import importlib
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.native.interface import NativeHarnessResult, NativeHarnessRunner
from src.native.scoring import score_result

logger = logging.getLogger(__name__)

# Native mode system prompt — no text-protocol instructions;
# the agent uses typed FHIR tools registered with Hermes.
_NATIVE_PROMPT = (
    "You are an expert clinician's assistant using FHIR APIs to answer medical questions.\n\n"
    "You have access to typed FHIR tools to query and update patient records. "
    "Use them to gather all information needed to answer the question.\n\n"
    "When you have all answers, call fhir_finish with a JSON-serializable list of answers "
    "matching the number of sub-questions asked. "
    "Do not call fhir_finish until you have confirmed every part of the answer.\n\n"
    "Context: {context}\n\n"
    "Question: {question}"
)


def _load_adapter_class(dotted_path: str):
    parts = dotted_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"adapter_class must be 'module.ClassName', got: {dotted_path}")
    module = importlib.import_module(parts[0])
    return getattr(module, parts[1])


class NativeBenchDriver:
    """
    Orchestrates a native benchmark run:
      1. Loads task data from test_data_v2.json and funcs_v1.json.
      2. Instantiates the configured NativeHarnessRunner.
      3. Calls setup → run → teardown → score for each task.
      4. Writes per-task artifacts and aggregate metrics.
    """

    def __init__(self, config: "NativeRunConfig"):
        self.config = config
        self._load_data()
        self._setup_output_dir()
        self._runner: Optional[NativeHarnessRunner] = None

    def _load_data(self) -> None:
        base = Path(self.config.benchmark.medagentbench_path)
        data_path = base / self.config.benchmark.data_file
        func_path = base / self.config.benchmark.func_file

        with open(data_path) as fh:
            self._all_cases: List[Dict] = json.load(fh)
        with open(func_path) as fh:
            self._funcs: List[Dict] = json.load(fh)

        logger.info("Loaded %d cases and %d functions", len(self._all_cases), len(self._funcs))

    def _setup_output_dir(self) -> None:
        run_id = self.config.logging.run_id or (
            time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        )
        self._run_id = run_id
        self._output_root = Path(self.config.logging.output_dir) / "runs" / run_id
        self._output_root.mkdir(parents=True, exist_ok=True)

        # Persist config snapshot.
        try:
            config_snapshot = self.config.dict()
            with open(self._output_root / "config.json", "w") as fh:
                json.dump(config_snapshot, fh, indent=2)
        except Exception as exc:
            logger.warning("Could not write config snapshot: %s", exc)

    def _filtered_cases(self) -> List[Dict]:
        cases = self._all_cases

        # Filter by task_ids if provided.
        if self.config.benchmark.task_ids:
            ids = set(self.config.benchmark.task_ids)
            cases = [c for c in cases if c["id"] in ids]

        # Filter by task_split.
        split = self.config.benchmark.task_split
        if split == "train":
            # First 15 samples per task type (indices 0-14 within each type).
            cases = [c for c in cases if self._sample_index(c["id"]) <= 15]
        elif split == "test":
            cases = [c for c in cases if self._sample_index(c["id"]) > 15]
        # "all" = no filter

        return cases

    @staticmethod
    def _sample_index(task_id: str) -> int:
        """Extract the numeric sample index from 'task1_23' → 23."""
        try:
            return int(task_id.split("_")[-1])
        except (ValueError, IndexError):
            return 0

    def _build_prompt(self, case_data: Dict) -> str:
        return _NATIVE_PROMPT.format(
            context=case_data.get("context", ""),
            question=case_data.get("instruction", ""),
        )

    def _instantiate_runner(self) -> NativeHarnessRunner:
        cls = _load_adapter_class(self.config.harness.adapter_class)
        return cls()

    def run(self) -> Dict[str, Any]:
        """Run all filtered tasks and return aggregate summary."""
        cases = self._filtered_cases()
        logger.info(
            "Starting run %s: %d tasks, harness=%s, model=%s",
            self._run_id, len(cases),
            self.config.harness.name, self.config.model.model_name,
        )

        results = []
        for case_data in cases:
            result = self.run_one_task(case_data)
            results.append(result)
            logger.info(
                "Task %s: success=%s, fhir_calls=%d, latency=%.1fs",
                case_data["id"], result["score"]["success"],
                result["score"]["fhir_call_count"],
                result["score"]["latency_seconds"],
            )

        summary = self._aggregate(results)
        self._write_summary(summary, results)
        return summary

    def run_one_task(self, case_data: Dict) -> Dict[str, Any]:
        """Run a single task: setup → run → teardown → score → write artifacts."""
        task_id = case_data["id"]
        artifact_dir = self._output_root / "tasks" / task_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

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

        harness_result: Optional[NativeHarnessResult] = None
        timed_out = False

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(runner.run)
                try:
                    harness_result = future.result(
                        timeout=self.config.runtime.timeout_seconds
                    )
                except FuturesTimeout:
                    timed_out = True
                    future.cancel()
                    logger.warning("Task %s timed out after %ds", task_id,
                                   self.config.runtime.timeout_seconds)
                except Exception as exc:
                    logger.exception("runner.run() raised for task %s", task_id)
                    harness_result = NativeHarnessResult(
                        final_answer=None,
                        answer_parse_status="missing",
                        transcript=[],
                        tool_call_log=[],
                        raw_harness_logs="",
                        fhir_call_count=0,
                        token_usage=None,
                        latency_seconds=0.0,
                        timed_out=False,
                        memory_mode=self.config.memory.mode,
                        errors=[f"runner error: {exc}"],
                        artifact_dir=str(artifact_dir),
                    )
        finally:
            try:
                runner.teardown()
            except Exception as exc:
                logger.warning("teardown error for %s: %s", task_id, exc)

        if harness_result is None:
            harness_result = NativeHarnessResult(
                final_answer=None,
                answer_parse_status="missing",
                transcript=[],
                tool_call_log=[],
                raw_harness_logs="",
                fhir_call_count=0,
                token_usage=None,
                latency_seconds=self.config.runtime.timeout_seconds,
                timed_out=True,
                memory_mode=self.config.memory.mode,
                errors=[f"timeout after {self.config.runtime.timeout_seconds}s"],
                artifact_dir=str(artifact_dir),
            )

        score = score_result(
            case_data, harness_result, self.config.benchmark.fhir_base_url
        )

        self._write_task_artifacts(artifact_dir, case_data, harness_result, score)

        return {"task_id": task_id, "score": score, "harness_result": harness_result}

    def _write_task_artifacts(
        self,
        artifact_dir: Path,
        case_data: Dict,
        result: NativeHarnessResult,
        score: Dict,
    ) -> None:
        # Task metadata.
        with open(artifact_dir / "task_metadata.json", "w") as fh:
            json.dump(case_data, fh, indent=2)

        # Score.
        with open(artifact_dir / "score.json", "w") as fh:
            json.dump(score, fh, indent=2)

        # Normalized trajectory (transcript).
        with open(artifact_dir / "normalized_trajectory.jsonl", "w") as fh:
            for msg in result.transcript:
                fh.write(json.dumps(msg) + "\n")

        # FHIR tool call log.
        if self.config.logging.log_tool_calls:
            with open(artifact_dir / "fhir_tool_calls.jsonl", "w") as fh:
                for entry in result.tool_call_log:
                    fh.write(json.dumps(entry) + "\n")

        # Error log.
        if result.errors:
            with open(artifact_dir / "errors.log", "w") as fh:
                fh.write("\n".join(result.errors) + "\n")

        # Summary for this task.
        summary = {
            "task_id": case_data["id"],
            "score": score,
            "fhir_call_count": result.fhir_call_count,
            "latency_seconds": result.latency_seconds,
            "timed_out": result.timed_out,
            "answer_parse_status": result.answer_parse_status,
            "token_usage": result.token_usage,
            "memory_mode": result.memory_mode,
        }
        with open(artifact_dir / "task_summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)

    def _aggregate(self, results: List[Dict]) -> Dict[str, Any]:
        if not results:
            return {"total": 0, "success_rate": 0.0}

        scores = [r["score"] for r in results]
        task_types = list({r["task_id"].split("_")[0] for r in results})

        total = len(scores)
        successes = sum(1 for s in scores if s["success"])
        timeouts = sum(1 for s in scores if s["timed_out"])
        budget_exhausted = sum(
            1 for r in results
            if r["harness_result"].fhir_call_count >= self.config.runtime.max_fhir_calls
        )

        per_type: Dict[str, Dict] = {}
        for task_type in task_types:
            type_scores = [s for s in scores if s["task_id"].startswith(task_type + "_")]
            per_type[task_type] = {
                "total": len(type_scores),
                "successes": sum(1 for s in type_scores if s["success"]),
                "success_rate": (
                    sum(1 for s in type_scores if s["success"]) / len(type_scores)
                    if type_scores else 0.0
                ),
            }

        latencies = [r["harness_result"].latency_seconds for r in results]
        fhir_counts = [r["harness_result"].fhir_call_count for r in results]

        return {
            "run_id": self._run_id,
            "total": total,
            "successes": successes,
            "success_rate": successes / total,
            "timeouts": timeouts,
            "budget_exhausted": budget_exhausted,
            "mean_latency_seconds": sum(latencies) / len(latencies),
            "mean_fhir_calls": sum(fhir_counts) / len(fhir_counts),
            "per_task_type": per_type,
            "harness": self.config.harness.name,
            "model": self.config.model.model_name,
            "memory_mode": self.config.memory.mode,
        }

    def _write_summary(self, summary: Dict, results: List[Dict]) -> None:
        # Summary JSON.
        with open(self._output_root / "summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)

        # Aggregate CSV.
        csv_path = self._output_root / "aggregate_metrics.csv"
        rows = []
        for r in results:
            s = r["score"]
            hr = r["harness_result"]
            rows.append({
                "task_id": r["task_id"],
                "success": s["success"],
                "fhir_call_count": hr.fhir_call_count,
                "latency_seconds": hr.latency_seconds,
                "timed_out": hr.timed_out,
                "answer_parse_status": hr.answer_parse_status,
                "memory_mode": hr.memory_mode,
                "errors": "; ".join(hr.errors) if hr.errors else "",
            })
        if rows:
            with open(csv_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        logger.info(
            "Run complete: success_rate=%.1f%% (%d/%d), output=%s",
            summary["success_rate"] * 100,
            summary["successes"],
            summary["total"],
            self._output_root,
        )
