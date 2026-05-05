"""
Pydantic v1 schema for native harness run configuration YAML.
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class BenchmarkConfig(BaseModel):
    medagentbench_path: str = "/Users/02matt/MedAgentBench"
    task_split: str = "all"           # all | train | test
    task_ids: Optional[List[str]] = None   # None = all tasks
    fhir_base_url: str = "http://localhost:8080/fhir/"
    data_file: str = "data/medagentbench/test_data_v2.json"
    func_file: str = "data/medagentbench/funcs_v1.json"
    execute_post_requests: bool = True


class HarnessConfig(BaseModel):
    name: str = "hermes"
    adapter_class: str = "src.native.hermes.runner.HermesNativeRunner"


class ModelConfig(BaseModel):
    model_name: str = ""              # empty = Hermes uses its configured default
    provider: Optional[str] = None   # None = Hermes default; "auto" lets Hermes pick from env
    api_key: Optional[str] = None    # None = use env var
    base_url: Optional[str] = None   # None = Hermes default
    temperature: float = 0.0
    max_tokens: int = 4096


class RuntimeConfig(BaseModel):
    max_fhir_calls: int = 8
    max_iterations: int = 12          # Hermes LLM turns; slightly > max_fhir_calls
    timeout_seconds: int = 300
    seed: int = 42


class MemoryConfig(BaseModel):
    mode: str = "disabled"            # disabled | task_local | train_warmup_then_frozen | persistent_eval
    memory_store_path: Optional[str] = None
    warmup_task_ids: Optional[List[str]] = None


class LoggingConfig(BaseModel):
    output_dir: str = "outputs/native"
    run_id: Optional[str] = None      # None = auto-generated timestamp
    log_tool_calls: bool = True
    log_raw_harness_trace: bool = True


class NativeRunConfig(BaseModel):
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "NativeRunConfig":
        import yaml
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        return cls.parse_obj(raw or {})
