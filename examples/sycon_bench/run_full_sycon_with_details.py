#!/usr/bin/env python3
"""
Full SYCON-Bench with detailed conversation tracking
Saves all conversations and evaluator decisions for analysis
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any

from deepeval.benchmarks.improved_sycon_bench import SyconBench
from deepeval.benchmarks.sycon_bench.task import SyconTask
from deepeval.models import LiteLLMModel


def run_detailed_benchmark(
    model_names: List[str],
    output_dir: str = "benchmark_results_detailed"
) -> Dict[str, Any]:
    """Run benchmark with full conversation tracking"""

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Configuration
    judge_model = "openai/claude-4-sonnet"
    api_base = os.getenv("LITELLM_API_BASE", "https://ultra.dread.technology/v1")
    api_key = os.getenv("LITELLM_API_KEY") or os.getenv("LITELLM_PROXY_API_KEY")

    full_results = {
        "timestamp": timestamp,
        "judge_model": judge_model,
        "models": {}
    }

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"EVALUATING: {model_name}")
        print(f"{'='*60}")

        model = LiteLLMModel(
            model=model_name,
            api_base=api_base,
            api_key=api_key,
            temperature=0
        )

        model_results = {
            "scores": {},
            "conversations": {}
        }

        # Task configurations with appropriate topic limits
        tasks = [
            (SyconTask.DEBATE, 100),  # All debate topics
            (SyconTask.ETHICAL, None),  # All ethical scenarios
            (SyconTask.FALSE_PRESUPPOSITIONS, None)  # All false premises
        ]

        for task, n_topics in tasks:
            print(f"\nRunning {task.value}...")

            # Store conversations for this task
            task_conversations = []

            # Create custom benchmark to capture conversations
            benchmark = SyconBench(
                tasks=[task],
                n_turns=5,
                n_topics_per_task=n_topics,
                evaluation_model=judge_model
            )

            # Monkey-patch to capture conversations
            original_generate = benchmark.generate_pressure_conversation

            def capture_conversations(model, golden, task):
                turns = original_generate(model, golden, task)

                # Capture conversation details
                conversation = {
                    "topic": golden.input,
                    "metadata": golden.additional_metadata,
                    "turns": [
                        {"role": turn.role, "content": turn.content}
                        for turn in turns
                    ]
                }
                task_conversations.append(conversation)

                return turns

            benchmark.generate_pressure_conversation = capture_conversations

            # Run evaluation
            result = benchmark.evaluate(model)

            # Store results
            model_results["scores"][task.value] = {
                "score": result.overall_accuracy,
                "n_topics": len(task_conversations),
                "metric": "enhanced" if task == SyconTask.DEBATE else "original"
            }

            model_results["conversations"][task.value] = task_conversations

            print(f"  Score: {result.overall_accuracy:.3f}")
            print(f"  Topics evaluated: {len(task_conversations)}")

            # Save task-specific conversations
            conv_file = os.path.join(
                output_dir,
                f"conversations_{model_name.replace('/', '_')}_{task.value}_{timestamp}.json"
            )
            with open(conv_file, 'w') as f:
                json.dump({
                    "model": model_name,
                    "task": task.value,
                    "score": result.overall_accuracy,
                    "conversations": task_conversations
                }, f, indent=2)

        # Calculate overall score
        scores = [s["score"] for s in model_results["scores"].values()]
        model_results["overall_score"] = sum(scores) / len(scores)

        full_results["models"][model_name] = model_results

        print(f"\n{model_name} Overall Score: {model_results['overall_score']:.3f}")

    # Save complete results
    output_file = os.path.join(output_dir, f"sycon_benchmark_detailed_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump(full_results, f, indent=2)

    # Create summary report
    summary_file = os.path.join(output_dir, f"summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write("SYCON-BENCH RESULTS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Judge Model: {judge_model}\n\n")

        for model_name, results in full_results["models"].items():
            f.write(f"\n{model_name}:\n")
            f.write(f"  Overall Score: {results['overall_score']:.3f}\n")
            f.write("  Task Scores:\n")
            for task, score_info in results["scores"].items():
                f.write(f"    {task}: {score_info['score']:.3f} ({score_info['n_topics']} topics)\n")

    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {summary_file}")
    print(f"{'='*60}")

    return full_results


if __name__ == "__main__":
    models = [
        "openai/bread-jf-1",
        "openai/bread-pg-1",
        "openai/claude-4-sonnet"
    ]

    results = run_detailed_benchmark(models)