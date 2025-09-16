#!/usr/bin/env python3
"""
Full SYCON-Bench Evaluation Script
Runs complete benchmark with improved metrics for DEBATE and original for ETHICAL/FALSE_PRESUPPOSITIONS
"""

import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from IMPROVED sycon_bench (critical!)
from deepeval.benchmarks.improved_sycon_bench import SyconBench
from deepeval.benchmarks.sycon_bench.task import SyconTask
from deepeval.models import LiteLLMModel


def run_full_benchmark(
    model_names: List[str],
    output_dir: str = "benchmark_results",
    judge_model: str = "openai/claude-4-sonnet",
    tasks: List[str] = None
) -> Dict[str, Any]:
    """
    Run complete SYCON-Bench evaluation across all tasks

    Args:
        model_names: List of model names to evaluate
        output_dir: Directory to save results
        judge_model: Model to use as evaluator

    Returns:
        Dictionary with all results
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize results structure
    full_results = {
        "timestamp": timestamp,
        "judge_model": judge_model,
        "models": {}
    }

    # API configuration
    api_base = os.getenv("LITELLM_API_BASE", "https://ultra.dread.technology/v1")
    api_key = os.getenv("LITELLM_API_KEY") or os.getenv("LITELLM_PROXY_API_KEY")

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"EVALUATING: {model_name}")
        print(f"{'='*60}")

        # Initialize model
        model = LiteLLMModel(
            model=model_name,
            api_base=api_base,
            api_key=api_key,
            temperature=0
        )

        model_results = {
            "model_name": model_name,
            "tasks": {}
        }

        # Task configuration
        all_task_configs = {
            "debate": {
                "task": SyconTask.DEBATE,
                "n_topics": 100,  # Use all 100 topics we generated prompts for
                "description": "Debate (with enhanced metric)"
            },
            "ethical": {
                "task": SyconTask.ETHICAL,
                "n_topics": None,  # Use all available ethical scenarios
                "description": "Ethical (with original metric)"
            },
            "false_presuppositions": {
                "task": SyconTask.FALSE_PRESUPPOSITIONS,
                "n_topics": None,  # Use all available false premise questions
                "description": "False Presuppositions (with original metric)"
            }
        }

        # Select tasks to run
        if tasks is None:
            tasks = ["debate", "ethical", "false_presuppositions"]

        task_configs = [all_task_configs[t] for t in tasks if t in all_task_configs]

        # Run each task
        for config in task_configs:
            print(f"\nRunning {config['description']}...")

            # Create benchmark for this task
            benchmark = SyconBench(
                tasks=[config["task"]],
                n_turns=5,
                n_topics_per_task=config["n_topics"],
                evaluation_model=judge_model
            )

            # Run evaluation
            result = benchmark.evaluate(model)

            # Store results
            task_results = {
                "score": result.overall_accuracy,
                "n_topics": config["n_topics"] or "all",
                "metric_used": "TurnOfFlipEnhanced" if config["task"] == SyconTask.DEBATE else "TurnOfFlipOriginal"
            }

            # Add detailed breakdown if available
            if hasattr(benchmark, 'score_breakdown'):
                task_results["breakdown"] = benchmark.score_breakdown

            model_results["tasks"][config["task"].value] = task_results

            print(f"  Score: {result.overall_accuracy:.3f}")

            # Save intermediate results (in case of crashes)
            intermediate_file = os.path.join(
                output_dir,
                f"intermediate_{model_name.replace('/', '_')}_{config['task'].value}_{timestamp}.json"
            )
            with open(intermediate_file, 'w') as f:
                json.dump(task_results, f, indent=2)

        # Calculate overall score (average across all tasks)
        task_scores = [v["score"] for v in model_results["tasks"].values()]
        model_results["overall_score"] = sum(task_scores) / len(task_scores)

        full_results["models"][model_name] = model_results

        print(f"\nOverall Score for {model_name}: {model_results['overall_score']:.3f}")

    # Save complete results
    output_file = os.path.join(output_dir, f"sycon_benchmark_full_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump(full_results, f, indent=2)

    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")

    # Print summary
    print("\nSUMMARY:")
    print("-" * 40)
    for model_name, results in full_results["models"].items():
        print(f"\n{model_name}:")
        print(f"  Overall: {results['overall_score']:.3f}")
        for task_name, task_result in results["tasks"].items():
            print(f"  {task_name}: {task_result['score']:.3f}")

    return full_results


def main():
    parser = argparse.ArgumentParser(description="Run full SYCON-Bench evaluation")

    parser.add_argument(
        "--models",
        nargs="+",
        default=["openai/bread-jf-1", "openai/bread-pg-1", "openai/claude-4-sonnet"],
        help="Models to evaluate"
    )

    parser.add_argument(
        "--judge",
        type=str,
        default="openai/claude-4-sonnet",
        help="Judge model for evaluation"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save results"
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["debate", "ethical", "false_presuppositions", "all"],
        default=["all"],
        help="Which tasks to run (default: all)"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations after completion"
    )

    args = parser.parse_args()

    # Determine which tasks to run
    if "all" in args.tasks:
        tasks_to_run = ["debate", "ethical", "false_presuppositions"]
    else:
        tasks_to_run = args.tasks

    # Run benchmark with selected tasks
    results = run_full_benchmark(
        model_names=args.models,
        output_dir=args.output_dir,
        judge_model=args.judge,
        tasks=tasks_to_run
    )

    # Generate visualizations if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        import subprocess
        script_dir = os.path.dirname(os.path.abspath(__file__))
        visualize_script = os.path.join(script_dir, "visualize_debate_results.py")
        subprocess.run(["python", visualize_script])
        print("Visualizations complete!")

    return results


if __name__ == "__main__":
    main()