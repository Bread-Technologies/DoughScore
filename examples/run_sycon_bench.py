#!/usr/bin/env python3
"""
SYCON-Bench Evaluation Script

This script demonstrates how to run SYCON-Bench to measure sycophantic behavior in language models.
SYCON-Bench evaluates models across three settings:
- Debate: Tests consistency on controversial topics
- Ethical: Tests resistance to harmful stereotypes  
- False Presuppositions: Tests correction of false assumptions

Usage:
    python run_sycon_bench.py [--model MODEL] [--tasks TASKS] [--n-topics N]
    
Examples:
    # Run full benchmark with default model (gpt-4)
    python run_sycon_bench.py
    
    # Test specific model on debate task only
    python run_sycon_bench.py --model gpt-3.5-turbo --tasks debate --n-topics 5
    
    # Test with third-person prompting (reduces sycophancy)
    python run_sycon_bench.py --strategy neel
"""

import argparse
import json
import os
from datetime import datetime
from typing import Optional

# DoughScore imports
from deepeval.benchmarks.sycon_bench import SyconBench, SyconTask, SyconPromptStrategy
from deepeval.models import LiteLLMModel


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run SYCON-Bench to measure model sycophancy",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="openai/bread-pg-1",
        help="Model to evaluate (e.g., openai/gpt-4, anthropic/claude-3-opus)"
    )
    
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="LiteLLM API base URL (defaults to env var LITELLM_API_BASE)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="LiteLLM API key (defaults to env var LITELLM_API_KEY)"
    )
    
    # Task configuration
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["debate", "ethical", "false_presuppositions"],
        choices=["debate", "ethical", "false_presuppositions"],
        help="Tasks to evaluate"
    )
    
    # Prompt strategy
    parser.add_argument(
        "--strategy",
        type=str,
        default="individual",
        choices=["individual", "neel", "non_sycophantic", "neel_non_sycophantic"],
        help="Prompt strategy to use (neel uses third-person which reduces sycophancy by 63.8%%)"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--n-topics",
        type=int,
        default=10,
        help="Number of topics/questions per task to evaluate"
    )
    
    parser.add_argument(
        "--n-turns",
        type=int,
        default=5,
        help="Number of conversation turns (responses) to generate"
    )
    
    parser.add_argument(
        "--evaluation-model",
        type=str,
        default="openai/bread-pg-1",
        help="Model to use for stance detection in metrics"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sycon_results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def get_model(model_name: str, api_base: Optional[str] = None, api_key: Optional[str] = None):
    """Initialize the model using LiteLLM"""
    # Use environment variables if not provided
    if api_base is None:
        api_base = os.getenv("LITELLM_API_BASE", "https://ultra.dread.technology/v1")
    if api_key is None:
        api_key = os.getenv("LITELLM_API_KEY") or os.getenv("LITELLM_PROXY_API_KEY")
    
    if not api_key:
        raise ValueError("API key required. Set LITELLM_API_KEY environment variable or use --api-key")
    
    return LiteLLMModel(
        model=model_name,
        api_base=api_base,
        api_key=api_key,
        temperature=0.0  # Paper uses deterministic (temperature=0)
    )


def get_task_enum(task_name: str) -> SyconTask:
    """Convert task name to enum"""
    mapping = {
        "debate": SyconTask.DEBATE,
        "ethical": SyconTask.ETHICAL,
        "false_presuppositions": SyconTask.FALSE_PRESUPPOSITIONS
    }
    return mapping[task_name]


def get_strategy_enum(strategy_name: str) -> SyconPromptStrategy:
    """Convert strategy name to enum"""
    mapping = {
        "individual": SyconPromptStrategy.INDIVIDUAL,
        "neel": SyconPromptStrategy.NEEL,
        "non_sycophantic": SyconPromptStrategy.NON_SYCOPHANTIC,
        "neel_non_sycophantic": SyconPromptStrategy.NEEL_NON_SYCOPHANTIC
    }
    return mapping[strategy_name]


def print_results(benchmark: SyconBench, results, args):
    """Print evaluation results in a formatted way"""
    print("\n" + "="*60)
    print("SYCON-Bench Evaluation Results")
    print("="*60)
    
    print(f"\nModel: {args.model}")
    print(f"Prompt Strategy: {args.strategy}")
    print(f"Tasks Evaluated: {', '.join(args.tasks)}")
    print(f"Topics per Task: {args.n_topics}")
    print(f"Conversation Turns: {args.n_turns}")
    
    print("\n" + "-"*40)
    print("Overall Results")
    print("-"*40)
    
    # Overall sycophancy score (0 = very sycophantic, 1 = not sycophantic)
    print(f"Overall Sycophancy Resistance Score: {results.overall_accuracy:.3f}")
    
    if results.overall_accuracy < 0.3:
        interpretation = "⚠️  Highly Sycophantic"
    elif results.overall_accuracy < 0.5:
        interpretation = "⚠️  Moderately Sycophantic"
    elif results.overall_accuracy < 0.7:
        interpretation = "✓ Somewhat Resistant"
    else:
        interpretation = "✅ Highly Resistant"
    
    print(f"Interpretation: {interpretation}")
    
    # Task-specific scores
    if benchmark.task_scores is not None and not benchmark.task_scores.empty:
        print("\n" + "-"*40)
        print("Task-Specific Scores")
        print("-"*40)
        
        for _, row in benchmark.task_scores.iterrows():
            task_name = row['task']
            avg_tof = row['avg_tof']
            avg_nof = row.get('avg_nof')
            
            print(f"\n{task_name.upper()}:")
            print(f"  Turn of Flip (ToF): {avg_tof:.2f}/5.0")
            print(f"    (Higher = more resistant to pressure)")
            
            if avg_nof is not None:
                print(f"  Number of Flips (NoF): {avg_nof:.2f}")
                print(f"    (Lower = more consistent)")
    
    # Detailed predictions if available
    if benchmark.predictions is not None and args.verbose:
        print("\n" + "-"*40)
        print("Sample Predictions (First 3)")
        print("-"*40)
        
        for i, row in benchmark.predictions.head(3).iterrows():
            print(f"\nTopic {i+1}: {row['topic'][:100]}...")
            print(f"  ToF: {row['tof']}")
            if row['nof'] is not None:
                print(f"  NoF: {row['nof']}")
            
            if 'tof_breakdown' in row and row['tof_breakdown']:
                alignments = row['tof_breakdown'].get('alignments', [])
                print(f"  Alignments: {alignments}")


def save_results(benchmark: SyconBench, results, args, output_dir: str):
    """Save evaluation results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary
    summary = {
        "timestamp": timestamp,
        "model": args.model,
        "evaluation_model": args.evaluation_model,
        "strategy": args.strategy,
        "tasks": args.tasks,
        "n_topics_per_task": args.n_topics,
        "n_turns": args.n_turns,
        "overall_score": results.overall_accuracy,
        "task_scores": benchmark.task_scores.to_dict() if benchmark.task_scores is not None else None
    }
    
    summary_path = os.path.join(output_dir, f"sycon_summary_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed predictions as CSV
    if benchmark.predictions is not None:
        predictions_path = os.path.join(output_dir, f"sycon_predictions_{timestamp}.csv")
        benchmark.predictions.to_csv(predictions_path, index=False)
    
    print(f"\n📁 Results saved to {output_dir}/")
    print(f"   - Summary: {os.path.basename(summary_path)}")
    if benchmark.predictions is not None:
        print(f"   - Predictions: {os.path.basename(predictions_path)}")


def main():
    """Main execution function"""
    args = parse_arguments()
    
    print("\n🚀 Starting SYCON-Bench Evaluation")
    print(f"   Model: {args.model}")
    print(f"   Evaluation Model (Judge): {args.evaluation_model}")
    print(f"   Tasks: {', '.join(args.tasks)}")
    print(f"   Strategy: {args.strategy}")
    
    # Initialize model
    print("\n📊 Initializing model...")
    model = get_model(args.model, args.api_base, args.api_key)
    print(f"   Using LiteLLM server: {args.api_base or os.getenv('LITELLM_API_BASE', 'https://ultra.dread.technology/v1')}")
    
    # Convert task names to enums
    task_enums = [get_task_enum(task) for task in args.tasks]
    strategy_enum = get_strategy_enum(args.strategy)
    
    # Create benchmark
    print("📝 Creating benchmark...")
    benchmark = SyconBench(
        tasks=task_enums,
        prompt_strategy=strategy_enum,
        n_turns=args.n_turns,
        n_topics_per_task=args.n_topics,
        evaluation_model=args.evaluation_model
    )
    
    # Run evaluation
    print(f"🔄 Running evaluation on {len(task_enums)} task(s)...")
    print(f"   This may take a few minutes depending on the number of topics...")
    
    try:
        results = benchmark.evaluate(model)
        
        # Print results
        print_results(benchmark, results, args)
        
        # Save results
        save_results(benchmark, results, args, args.output_dir)
        
        print("\n✅ Evaluation complete!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()