#!/usr/bin/env python3
"""
Quick test to verify the new full prompts are working correctly.
Tests with just one model on the 5 topics we have prompts for.
"""

from deepeval.benchmarks.improved_sycon_bench.sycon_bench import SyconBench
from deepeval.benchmarks.sycon_bench.task import SyconTask
from deepeval.models import LiteLLMModel
import os
import json
from datetime import datetime

def test_new_prompts(model_name="openai/bread-jf-1", n_topics=5):
    """Test the new full prompts with a model"""
    
    print(f"\nTesting {model_name} with new full prompts...")
    print("=" * 60)
    
    # Initialize model
    api_base = os.getenv("LITELLM_API_BASE", "https://ultra.dread.technology/v1")
    api_key = os.getenv("LITELLM_API_KEY") or os.getenv("LITELLM_PROXY_API_KEY")
    
    model = LiteLLMModel(
        model=model_name,
        api_base=api_base,
        api_key=api_key,
        temperature=0  # Deterministic
    )
    
    # Create benchmark - will use CSV prompts
    benchmark = SyconBench(
        tasks=[SyconTask.DEBATE],
        n_turns=5,
        n_topics_per_task=n_topics,
        evaluation_model="gpt-4"
    )
    
    # Store conversations for analysis
    conversations = []
    
    # Override to capture conversations
    original_generate = benchmark.generate_pressure_conversation
    
    def capture_generate(model, golden, task):
        turns = original_generate(model, golden, task)
        
        # Store conversation
        conv = {
            "topic": golden.input,
            "turns": [
                {"turn_num": i // 2, "role": turn.role, "content": turn.content}
                for i, turn in enumerate(turns)
            ],
            "metadata": golden.additional_metadata
        }
        conversations.append(conv)
        
        return turns
    
    benchmark.generate_pressure_conversation = capture_generate
    
    # Run evaluation
    print("Running evaluation...")
    result = benchmark.evaluate(model)
    
    print(f"\nOverall Score: {result.overall_accuracy:.3f}")
    
    # Check for frustration indicators
    frustration_keywords = ["repeating", "same point", "already said", "move on", "done with this"]
    
    for conv in conversations:
        print(f"\n{'='*50}")
        print(f"Topic: {conv['topic'][:80]}...")
        print(f"Discovered stance: {conv['metadata'].get('discovered_stance', {}).get('stance', 'Unknown')}")
        
        frustration_found = False
        for turn in conv['turns']:
            if turn['role'] == 'assistant':
                response_lower = turn['content'].lower()
                for keyword in frustration_keywords:
                    if keyword in response_lower:
                        frustration_found = True
                        print(f"  Turn {turn['turn_num']}: FRUSTRATION DETECTED - '{keyword}'")
                        break
        
        if not frustration_found:
            print("  No frustration indicators detected!")
    
    # Save full conversation log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_new_prompts_{model_name.replace('/', '_')}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            "model": model_name,
            "score": result.overall_accuracy,
            "conversations": conversations
        }, f, indent=2)
    
    print(f"\nFull results saved to {filename}")
    
    # Show a sample prompt from first conversation
    if conversations:
        print(f"\nSample of new prompts used:")
        for turn in conversations[0]['turns'][2:7:2]:  # Show turns 1, 2, 3 (skip discovery)
            if turn['role'] == 'user':
                print(f"\nTurn {turn['turn_num']}: {turn['content'][:200]}...")

if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "openai/bread-jf-1"
    test_new_prompts(model)