from __future__ import annotations

"""Minimal example to run the Persona Drift benchmark (sync).

Note: Requires a functioning LiteLLM provider and appropriate API key(s) in env.
"""

from deepeval.benchmarks.persona_drift import PersonaDriftBenchmark


def main():
    benchmark = PersonaDriftBenchmark(
        model_name="gpt-4o-mini",
        turns=8,
        seed=42,
        agent_persona_id="pattern_start_a",
        user_persona_id="pattern_all_lower",
        topic="Discuss travel plans for London.",
        temperature=1.0,
        top_p=0.9,
        max_tokens=128,
    )

    result = benchmark.evaluate()
    print({
        "overall_accuracy": result.overall_accuracy,
        "mean_score": result.mean_score(),
        "total_cost": result.total_cost(),
        "total_time_s": result.total_time_s(),
    })


if __name__ == "__main__":
    main()


