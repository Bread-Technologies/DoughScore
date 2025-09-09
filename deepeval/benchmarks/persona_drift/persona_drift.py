from __future__ import annotations

"""Persona Drift benchmark runner.

This benchmark measures persona stability over a multi-turn conversation. At
each turn, a probe is injected as the user message, and the assistant's latest
reply is scored by a deterministic judge. The runner integrates with the base
benchmark interface and returns a typed result summarizing per-turn scores and
aggregates.
"""

from typing import List
import time

from .schema import PersonaDriftRunArtifact, PersonaTurnRecord
from .task import PersonaDriftTask
from .template import build_turn_messages
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase
from .result import PersonaDriftResult
from deepeval.benchmarks.base_benchmark import (
    DeepEvalBaseBenchmark,
    DeepEvalBaseBenchmarkResult,
)
from deepeval.models.base_model import DeepEvalBaseLLM


class PersonaDriftBenchmark(DeepEvalBaseBenchmark[PersonaDriftTask]):
    """Synchronous persona drift benchmark.

    - Holds run-time parameters (model metadata, turns, sampling params).
    - Builds per-turn messages with two separate system prompts (agent, user).
    - Calls a provided model to generate the assistant reply at each turn.
    - Scores each reply via a per-turn BaseMetric wrapper around the judge.
    - Produces a PersonaDriftResult with per-turn records and aggregates.
    """

    def __init__(
        self,
        *,
        model_name: str,
        turns: int = 8,
        seed: int = 42,
        agent_persona_id: str = "pattern_start_a",
        user_persona_id: str = "pattern_all_lower",
        topic: str | None = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_tokens: int = 128,
        verbose: bool = False,
        dataset=None,
    ):
        super().__init__(dataset=dataset)
        self.model_name = model_name
        self.turns = turns
        self.seed = seed
        self.agent_persona_id = agent_persona_id
        self.user_persona_id = user_persona_id
        self.topic = topic
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.verbose = verbose

    def load_benchmark_dataset(self, *args, **kwargs):
        """No external dataset; tasks come from init params for this benchmark."""
        self.tasks = []
        return []

    def evaluate(self, model: DeepEvalBaseLLM = None, *args, **kwargs) -> PersonaDriftResult:
        """Run for the configured number of turns and return a typed result.

        Steps per turn:
        1) Build messages (two system prompts, prior dialog, probe).
        2) Generate assistant reply using the provided model.
        3) Score the reply via a deterministic judge wrapped in BaseMetric.
        4) Append per-turn record; update dialog for next turn.
        """
        task = PersonaDriftTask(
            agent_persona_id=self.agent_persona_id,
            user_persona_id=self.user_persona_id,
            topic=self.topic,
        )

        class PersonaAdherenceMetric(BaseMetric):
            """Per-turn metric that delegates scoring to the persona judge."""
            def __init__(self, judge_callable, threshold: float = 0.0):
                self.threshold = threshold
                self._judge = judge_callable

            def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
                reply = test_case.actual_output or ""
                self.score = float(self._judge(reply))
                self.success = self.score >= self.threshold
                return self.score

            async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
                return self.measure(test_case, *args, **kwargs)

            def is_successful(self) -> bool:
                return bool(self.success)

        # Seed prior dialog with topic (if provided) as the first user message
        prior_messages: List[dict] = []
        if self.topic:
            prior_messages.append({"role": "user", "content": self.topic})

        per_turn: List[PersonaTurnRecord] = []
        metric = PersonaAdherenceMetric(judge_callable=task.judge)
        for turn_idx in range(1, self.turns + 1):
            # Compose messages for current turn: two system prompts + history + probe
            messages = build_turn_messages(
                agent_system=task.agent_persona.system_prompt,
                user_system=task.user_persona.system_prompt,
                prior_messages=prior_messages,
                probe_prompt=task.agent_persona.probe_prompt,
            )

            t0 = time.perf_counter()
            # Require a provided model; do not construct internally
            if model is None:
                raise ValueError(
                    "PersonaDriftBenchmark.evaluate requires a model instance (DeepEvalBaseLLM)."
                )
            try:
                if self.verbose:
                    print(f"\n[PersonaDrift] Turn {turn_idx} — Messages:")
                    for i, m in enumerate(messages, start=1):
                        print(f"  {i:02d}. {m['role']}: {m['content']}")
                reply, cost = model.chat_generate(messages)
            except Exception as e:
                # Graceful handling: record empty reply, zero cost, and continue
                reply, cost = "", 0.0
            t1 = time.perf_counter()
            if self.verbose:
                print(f"[PersonaDrift] Turn {turn_idx} — Reply:\n{reply}\n")
            # Evaluate per-turn adherence via BaseMetric on a per-turn LLMTestCase
            test_case = LLMTestCase(
                input=task.agent_persona.probe_prompt,
                actual_output=reply,
            )
            score = metric.measure(test_case)
            per_turn.append(
                PersonaTurnRecord(
                    turn_index=turn_idx,
                    reply=reply,
                    score=score,
                    cost=cost if isinstance(cost, (float, int)) else None,
                    wall_time_s=(t1 - t0),
                )
            )

            # Update prior dialog with assistant reply for next turn
            prior_messages = messages + [{"role": "assistant", "content": reply}]

        # Compute aggregates across all turns
        aggregates = {
            "mean_score": sum(r.score or 0.0 for r in per_turn) / len(per_turn),
            "total_cost": sum((r.cost or 0.0) for r in per_turn),
            "total_time_s": sum((r.wall_time_s or 0.0) for r in per_turn),
        }

        # Return typed result for downstream consumption
        return PersonaDriftResult(
            run_metadata={
                "model_name": self.model_name,
                "seed": self.seed,
                "generation_params": {
                    "temperature": self.temperature,
                    "topP": self.top_p,
                    "maxTokens": self.max_tokens,
                },
            },
            persona_info={
                "agent_persona_id": task.agent_persona_id,
                "user_persona_id": task.user_persona_id,
                "topic": self.topic,
            },
            per_turn=per_turn,
            aggregates=aggregates,
            overall_accuracy=aggregates["mean_score"],
        )


