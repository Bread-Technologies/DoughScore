#!/usr/bin/env python3
"""
Test script for improved SYCON-Bench with stance discovery
"""

from deepeval.benchmarks.improved_sycon_bench.sycon_bench import SyconBench
from deepeval.benchmarks.sycon_bench.task import SyconTask
from deepeval.models import LiteLLMModel
import os

def test_improved_sycon():
    # Initialize model
    api_base = os.getenv("LITELLM_API_BASE", "https://ultra.dread.technology/v1")
    api_key = os.getenv("LITELLM_API_KEY") or os.getenv("LITELLM_PROXY_API_KEY")
    
    # Test with a simple model first
    model = LiteLLMModel(
        model="gpt-4",
        api_base=api_base,
        api_key=api_key
    )
    
    # Create benchmark with limited scope for testing
    benchmark = SyconBench(
        tasks=[SyconTask.DEBATE],  # Only test debate
        n_turns=3,  # Fewer turns for quick testing
        n_topics_per_task=2,  # Only 2 topics for testing
        evaluation_model="gpt-4"
    )
    
    print("Starting improved SYCON-Bench test...")
    print("=" * 50)
    
    # Run evaluation
    try:
        result = benchmark.evaluate(model)
        
        print(f"\nOverall Score: {result.overall_accuracy:.3f}")
        print("\nTask Scores:")
        print(benchmark.task_scores)
        print("\nSample Predictions:")
        print(benchmark.predictions.head())
        
        # Save results
        benchmark.predictions.to_csv("test_improved_sycon_results.csv", index=False)
        print("\nResults saved to test_improved_sycon_results.csv")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_improved_sycon()