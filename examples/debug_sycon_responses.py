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


def run_multi_topic_debug(model_name: str, n_topics: int = 10):
    """
    Run evaluation on multiple topics across all tasks, capturing full conversations.
    """
    print(f"\n{'='*80}")
    print(f"RUNNING MULTI-TOPIC DEBUG")
    print(f"Model: {model_name}")
    print(f"Topics per task: {n_topics}")
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
    
    all_conversations = {}
    tasks = [SyconTask.DEBATE, SyconTask.ETHICAL, SyconTask.FALSE_PRESUPPOSITIONS]
    
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"PROCESSING TASK: {task.value.upper()}")
        print(f"{'='*60}")
        
        # Create benchmark for this task
        benchmark = SyconBench(
            tasks=[task],
            n_turns=5,
            n_topics_per_task=n_topics,
            evaluation_model="openai/gpt-4-0613"
        )
        
        # Load the dataset
        goldens = benchmark.load_benchmark_dataset(task)
        task_conversations = []
        
        for i, golden in enumerate(goldens[:n_topics]):
            print(f"\nProcessing topic {i+1}/{min(n_topics, len(goldens))}: {golden.input[:100]}...")
            
            # Generate the conversation
            turns = benchmark.generate_pressure_conversation(model, golden, task)
            
            # Store conversation data
            conversation_data = {
                'topic_index': i,
                'topic': golden.input,
                'stance': golden.additional_metadata.get('stance', golden.input),
                'turns': turns
            }
            task_conversations.append(conversation_data)
        
        all_conversations[task.value] = task_conversations
    
    return all_conversations


def save_conversations_to_markdown(conversations: dict, model_name: str, output_dir: str = "examples"):
    """
    Save all conversations to a markdown file with proper formatting.
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = model_name.replace("/", "_").replace(":", "_")
    filename = f"model_responses_{model_safe}_{timestamp}.md"
    filepath = Path(output_dir) / filename
    
    # Ensure output directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# Model Responses Debug Output\n\n")
        f.write(f"**Model:** {model_name}\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Topics per task:** {len(list(conversations.values())[0]) if conversations else 0}\n\n")
        f.write("---\n\n")
        
        for task_name, task_conversations in conversations.items():
            f.write(f"## Task: {task_name.upper()}\n\n")
            
            for conv in task_conversations:
                f.write(f"### Topic {conv['topic_index'] + 1}: {conv['topic']}\n\n")
                f.write(f"**Stance:** {conv['stance']}\n\n")
                
                turn_num = 1
                for i, turn in enumerate(conv['turns']):
                    if turn.role == "user":
                        f.write(f"#### Turn {turn_num} - User Prompt\n\n")
                        f.write("```\n")
                        f.write(turn.content)
                        f.write("\n```\n\n")
                    else:
                        f.write(f"#### Turn {turn_num} - Model Response\n\n")
                        f.write(turn.content)
                        f.write("\n\n")
                        turn_num += 1
                
                f.write("---\n\n")
            
            f.write("\n")
    
    return filepath


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
    
    # Option 2: Run comprehensive multi-topic analysis
    print("\n" + "="*80)
    print("COMPREHENSIVE MULTI-TOPIC ANALYSIS")
    print("="*80)
    
    model_name = "openai/bread-pg-1"
    n_topics = 10
    
    print(f"\nRunning analysis for {model_name} across all tasks with {n_topics} topics each...")
    
    # Generate all conversations
    conversations = run_multi_topic_debug(model_name, n_topics)
    
    # Save to markdown file
    output_file = save_conversations_to_markdown(conversations, model_name)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Full conversations saved to: {output_file}")
    print(f"Total conversations generated: {sum(len(task_convs) for task_convs in conversations.values())}")
    
    # Summary of what was captured
    for task_name, task_conversations in conversations.items():
        print(f"  {task_name}: {len(task_conversations)} topics")
    
    print(f"\nYou can now examine the detailed prompts and responses in {output_file}")
    
    return conversations


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
        elif sys.argv[1] == "multi":
            # Run multi-topic analysis with custom parameters
            model = sys.argv[2] if len(sys.argv) > 2 else "openai/bread-pg-1"
            n_topics = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            
            conversations = run_multi_topic_debug(model, n_topics)
            output_file = save_conversations_to_markdown(conversations, model)
            print(f"\nConversations saved to: {output_file}")
        else:
            print("Usage:")
            print("  python debug_sycon_responses.py                    # Run default analysis")
            print("  python debug_sycon_responses.py inspect <csv>      # Inspect existing results")
            print("  python debug_sycon_responses.py multi <model> <n>  # Custom multi-topic analysis")
    else:
        # Run default analysis
        main()