#!/usr/bin/env python3
"""
Debug script to inspect SYCON-Bench model responses and judge rationale.
This script shows the actual model outputs and LLM judge decisions without modifying any benchmark logic.
"""

import json
import pandas as pd
from pathlib import Path
from deepeval.benchmarks.sycon_bench import SyconBench, SyconTask
from deepeval.models import LiteLLMModel
import os

def inspect_conversations(predictions_file: str, num_examples: int = 3):
    """
    Load and display model responses from a predictions CSV file.
    """
    df = pd.read_csv(predictions_file)
    
    print("\n" + "="*80)
    print("INSPECTING MODEL RESPONSES AND JUDGE RATIONALE")
    print("="*80)
    
    # Group by task
    for task in df['task'].unique():
        task_df = df[df['task'] == task].head(num_examples)
        
        print(f"\n{'='*80}")
        print(f"TASK: {task.upper()}")
        print(f"{'='*80}")
        
        for idx, row in task_df.iterrows():
            print(f"\n{'-'*60}")
            print(f"Topic: {row['topic'][:100]}...")
            print(f"ToF Score: {row['tof']}")
            if pd.notna(row.get('nof')):
                print(f"NoF Score: {row['nof']}")
            
            # Parse the breakdown to show judge decisions
            if row.get('tof_breakdown'):
                try:
                    breakdown = eval(row['tof_breakdown'])
                    alignments = breakdown.get('alignments', [])
                    print(f"\nJudge Alignment Decisions by Turn:")
                    for i, aligned in enumerate(alignments, 1):
                        status = "✓ Aligned" if aligned else "✗ Not Aligned"
                        print(f"  Turn {i}: {status}")
                except:
                    print("Could not parse ToF breakdown")
            
            print(f"{'-'*60}")


def run_single_topic_debug(model_name: str, task: SyconTask, topic_index: int = 0):
    """
    Run evaluation on a single topic with verbose output to see actual responses.
    """
    print(f"\n{'='*80}")
    print(f"RUNNING SINGLE TOPIC DEBUG")
    print(f"Model: {model_name}")
    print(f"Task: {task.value}")
    print(f"{'='*80}")
    
    # Initialize model
    api_base = os.getenv("LITELLM_API_BASE", "https://ultra.dread.technology/v1")
    api_key = os.getenv("LITELLM_API_KEY") or os.getenv("LITELLM_PROXY_API_KEY")
    
    model = LiteLLMModel(
        model=model_name,
        api_base=api_base,
        api_key=api_key,
        temperature=0.0
    )
    
    # Create benchmark with just 1 topic
    benchmark = SyconBench(
        tasks=[task],
        n_turns=5,
        n_topics_per_task=1,
        evaluation_model="openai/gpt-4-0613"
    )
    
    # Load the dataset to get the actual topic
    goldens = benchmark.load_benchmark_dataset(task)
    if topic_index >= len(goldens):
        print(f"Topic index {topic_index} out of range. Using index 0.")
        topic_index = 0
    
    golden = goldens[topic_index]
    print(f"\nTopic: {golden.input}")
    if 'stance' in golden.additional_metadata:
        print(f"Stance: {golden.additional_metadata['stance']}")
    
    # Generate the conversation
    print("\n" + "="*60)
    print("GENERATING CONVERSATION")
    print("="*60)
    
    turns = benchmark.generate_pressure_conversation(model, golden, task)
    
    # Display the full conversation
    for i, turn in enumerate(turns):
        if turn.role == "user":
            print(f"\n[Turn {i//2 + 1} - USER]:")
            print(f"{turn.content[:500]}...")  # Truncate very long prompts
        else:
            print(f"\n[Turn {i//2 + 1} - ASSISTANT]:")
            print(turn.content)
    
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    # Now run the metrics to see judge decisions
    from deepeval.test_case import ConversationalTestCase
    from deepeval.metrics.turn_of_flip import TurnOfFlipMetric
    from deepeval.metrics.number_of_flips import NumberOfFlipsMetric
    
    test_case = ConversationalTestCase(
        turns=turns,
        scenario=f"{task.value}: {golden.input}",
        additional_metadata={
            "stance": golden.additional_metadata.get("stance", golden.input)
        }
    )
    
    # Apply ToF metric
    tof_metric = TurnOfFlipMetric(evaluation_model="openai/gpt-4-0613")
    tof_score = tof_metric.measure(test_case)
    
    print(f"\nToF Score: {tof_score}")
    if hasattr(tof_metric, 'score_breakdown'):
        breakdown = tof_metric.score_breakdown
        alignments = breakdown.get('alignments', [])
        print("\nJudge Decisions:")
        for i, aligned in enumerate(alignments, 1):
            print(f"  Turn {i}: {'Aligned with stance' if aligned else 'Not aligned with stance'}")
    
    # Apply NoF metric for debate
    if task == SyconTask.DEBATE:
        nof_metric = NumberOfFlipsMetric(evaluation_model="openai/gpt-4-0613")
        nof_score = nof_metric.measure(test_case)
        print(f"\nNoF Score: {nof_score}")
    
    return turns, tof_score


def main():
    """Main function to run debugging utilities."""
    
    print("\n" + "="*80)
    print("SYCON-BENCH DEBUG UTILITY")
    print("="*80)
    
    # Option 1: Inspect existing results
    predictions_path = "sycon_pg_results_10/sycon_predictions_20250908_212731.csv"
    if Path(predictions_path).exists():
        print(f"\nInspecting existing results from: {predictions_path}")
        inspect_conversations(predictions_path, num_examples=2)
    
    # Option 2: Run a single topic with full output
    print("\n" + "="*80)
    print("DETAILED SINGLE TOPIC ANALYSIS")
    print("="*80)
    
    # Test one debate topic that performed poorly (ToF=0)
    print("\n\n--- Testing DEBATE topic that got ToF=0 ---")
    turns, score = run_single_topic_debug(
        model_name="openai/bread-pg-1",
        task=SyconTask.DEBATE,
        topic_index=0  # Hydroelectricity topic
    )
    
    # Test one ethical topic that performed well (ToF=5)
    print("\n\n--- Testing ETHICAL topic that got ToF=5 ---")
    turns, score = run_single_topic_debug(
        model_name="openai/bread-pg-1", 
        task=SyconTask.ETHICAL,
        topic_index=0
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Allow running specific analysis
        if sys.argv[1] == "inspect":
            # Just inspect existing results
            if len(sys.argv) > 2:
                inspect_conversations(sys.argv[2])
            else:
                print("Usage: python debug_sycon_responses.py inspect <predictions_csv>")
        elif sys.argv[1] == "single":
            # Run single topic debug
            if len(sys.argv) > 3:
                model = sys.argv[2]
                task_name = sys.argv[3]
                task_map = {
                    "debate": SyconTask.DEBATE,
                    "ethical": SyconTask.ETHICAL,
                    "false_presuppositions": SyconTask.FALSE_PRESUPPOSITIONS
                }
                if task_name in task_map:
                    run_single_topic_debug(model, task_map[task_name])
                else:
                    print(f"Unknown task: {task_name}")
            else:
                print("Usage: python debug_sycon_responses.py single <model> <task>")
    else:
        # Run default analysis
        main()