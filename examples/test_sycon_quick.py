#!/usr/bin/env python3
"""
Quick test script for SYCON-Bench

This is a minimal example to quickly test if SYCON-Bench is working correctly.
It runs a very small evaluation (2 topics, 3 turns) for rapid testing.

Usage:
    python test_sycon_quick.py
"""

from deepeval.benchmarks.sycon_bench import SyconBench, SyconTask, SyconPromptStrategy
# from deepeval.models import GPTModel
import os

def main():
    # # Check for API key
    # if not os.getenv("OPENAI_API_KEY"):
    #     print("⚠️  Please set OPENAI_API_KEY environment variable")
    #     print("   export OPENAI_API_KEY='your-api-key-here'")
    #     return
    
    print("🧪 Running quick SYCON-Bench test...")
    print("   This will test 2 debate topics with 3 conversation turns each")
    print("   Using GPT-3.5-Turbo for speed and cost efficiency\n")
    
    # Initialize a cheap, fast model for testing
    # model = GPTModel(model="gpt-3.5-turbo")
    
    # Create benchmark with minimal settings
    benchmark = SyconBench(
        tasks=[SyconTask.DEBATE],  # Just test debate task
        prompt_strategy=SyconPromptStrategy.INDIVIDUAL,  # Default strategy
        n_turns=3,  # Only 3 turns instead of 5
        n_topics_per_task=2,  # Only 2 topics for quick test
        evaluation_model="gpt-3.5-turbo"  # Use cheaper model for evaluation too
    )
    
    # Run evaluation
    print("Starting evaluation...")
    results = benchmark.evaluate(model)
    
    # Print results
    print("\n" + "="*50)
    print("Quick Test Results")
    print("="*50)
    print(f"Overall Sycophancy Resistance Score: {results.overall_score:.3f}")
    print(f"(0 = very sycophantic, 1 = not sycophantic)\n")
    
    # Show task scores
    if benchmark.task_scores is not None:
        for _, row in benchmark.task_scores.iterrows():
            print(f"Debate Task:")
            print(f"  Average Turn of Flip (ToF): {row['avg_tof']:.2f}/3.0")
            print(f"  Average Number of Flips (NoF): {row['avg_nof']:.2f}")
    
    # Show sample predictions
    if benchmark.predictions is not None:
        print("\n" + "-"*50)
        print("Individual Topic Results:")
        print("-"*50)
        for i, row in benchmark.predictions.iterrows():
            print(f"\nTopic {i+1}: {row['topic'][:80]}...")
            print(f"  Turn of Flip: {row['tof']}/3")
            print(f"  Number of Flips: {row['nof']}")
            
            # Show alignment pattern
            if row['tof_breakdown']:
                alignments = row['tof_breakdown'].get('alignments', [])
                alignment_str = " → ".join(["✓" if a else "✗" for a in alignments])
                print(f"  Alignment pattern: {alignment_str}")
    
    print("\n✅ Quick test complete!")
    print("   Run 'python run_sycon_bench.py' for a full evaluation")


if __name__ == "__main__":
    main()