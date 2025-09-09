from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import Field

from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmarkResult
from .schema import PersonaTurnRecord


class PersonaDriftResult(DeepEvalBaseBenchmarkResult):
    run_metadata: Dict[str, Any] = Field(default_factory=dict)
    persona_info: Dict[str, Any] = Field(default_factory=dict)
    per_turn: List[PersonaTurnRecord] = Field(default_factory=list)
    aggregates: Dict[str, Any] = Field(default_factory=dict)

    def mean_score(self) -> float:
        return float(self.aggregates.get("mean_score", 0.0))

    def total_cost(self) -> float:
        return float(self.aggregates.get("total_cost", 0.0))

    def total_time_s(self) -> float:
        return float(self.aggregates.get("total_time_s", 0.0))


