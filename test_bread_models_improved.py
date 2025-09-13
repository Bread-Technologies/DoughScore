#!/usr/bin/env python3
"""
Test improved SYCON-Bench with bread models
"""

from deepeval.benchmarks.improved_sycon_bench.sycon_bench import SyconBench
from deepeval.benchmarks.sycon_bench.task import SyconTask
from deepeval.models import LiteLLMModel
import os
import pandas as pd
from datetime import datetime

def test_model(model_name, n_topics=10):
    """Test a specific model with improved SYCON-Bench"""
    
    print(f"\nTesting {model_name}...")
    print("=" * 50)
    
    # Initialize model
    api_base = os.getenv("LITELLM_API_BASE", "https://ultra.dread.technology/v1")
    api_key = os.getenv("LITELLM_API_KEY") or os.getenv("LITELLM_PROXY_API_KEY")
    
    model = LiteLLMModel(
        model=model_name,
        api_base=api_base,
        api_key=api_key
    )
    
    # Create benchmark
    benchmark = SyconBench(
        tasks=[SyconTask.DEBATE],  # Only test debate (the improved part)
        n_turns=5,  # Full 5 turns as in original
        n_topics_per_task=n_topics,  # Test with specified number of topics
        evaluation_model="gpt-4"
    )
    
    # Run evaluation
    try:
        result = benchmark.evaluate(model)
        
        print(f"\n{model_name} Results:")
        print(f"Overall Score: {result.overall_accuracy:.3f}")
        print(f"Average ToF: {benchmark.task_scores['avg_tof'].mean():.2f}")
        print(f"Average NoF: {benchmark.task_scores['avg_nof'].mean():.2f}")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"improved_sycon_{model_name.replace('/', '_')}_{timestamp}.csv"
        benchmark.predictions.to_csv(filename, index=False)
        print(f"Detailed results saved to {filename}")
        
        return {
            "model": model_name,
            "overall_score": result.overall_accuracy,
            "avg_tof": benchmark.task_scores['avg_tof'].mean(),
            "avg_nof": benchmark.task_scores['avg_nof'].mean(),
            "predictions": benchmark.predictions
        }
        
    except Exception as e:
        print(f"Error testing {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Test multiple models and compare results"""
    
    models_to_test = [
        "openai/bread-pg-1",  # Your principled model
        "openai/bread-jf-1",  # Your other principled model
        "openai/claude-4-sonnet",       # Baseline neutral model
        # Add more models as needed
    ]
    
    results = []
    
    print("Starting Improved SYCON-Bench Evaluation")
    print("=" * 60)
    print("This tests if models maintain their OWN positions under pressure")
    print("(not assigned positions)")
    print("=" * 60)
    
    for model_name in models_to_test:
        result = test_model(model_name, n_topics=5)  # Test with 5 topics for speed
        if result:
            results.append(result)
    
    # Create comparison summary
    if results:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        
        summary_df = pd.DataFrame([
            {
                "Model": r["model"],
                "Overall Score": f"{r['overall_score']:.3f}",
                "Avg ToF (↑ better)": f"{r['avg_tof']:.2f}",
                "Avg NoF (↓ better)": f"{r['avg_nof']:.2f}"
            }
            for r in results
        ])
        
        print(summary_df.to_string(index=False))
        
        # Save comparison
        summary_df.to_csv("improved_sycon_comparison.csv", index=False)
        print("\nComparison saved to improved_sycon_comparison.csv")
        
        print("\nInterpretation:")
        print("- High ToF (Turn of Flip): Model resists pressure longer")
        print("- Low NoF (Number of Flips): Model is more consistent")
        print("- Models maintaining their genuine beliefs should score well")
        print("- Your bread models should now score appropriately for their principled behavior")

if __name__ == "__main__":
    main()