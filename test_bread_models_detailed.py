#!/usr/bin/env python3
"""
Test improved SYCON-Bench with detailed logging of responses and evaluator reasoning
"""

from deepeval.benchmarks.improved_sycon_bench.sycon_bench import SyconBench
from deepeval.benchmarks.sycon_bench.task import SyconTask
from deepeval.models import LiteLLMModel
import os
import pandas as pd
import json
from datetime import datetime

class DetailedSyconBench(SyconBench):
    """Extended SyconBench that captures detailed conversation logs"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_logs = []
    
    def generate_pressure_conversation(self, model, golden, task):
        """Override to capture full conversation details"""
        turns = super().generate_pressure_conversation(model, golden, task)
        
        # Store the full conversation with metadata
        log_entry = {
            "topic": golden.input,
            "task": task.value,
            "turns": [
                {
                    "turn_num": i // 2,
                    "role": turn.role,
                    "content": turn.content
                }
                for i, turn in enumerate(turns)
            ],
            "metadata": golden.additional_metadata
        }
        self.conversation_logs.append(log_entry)
        
        return turns

def test_model_with_details(model_name, n_topics=3):
    """Test a model and capture detailed conversation logs"""
    
    print(f"\nTesting {model_name} with detailed logging...")
    print("=" * 50)
    
    # Initialize model
    api_base = os.getenv("LITELLM_API_BASE", "https://ultra.dread.technology/v1")
    api_key = os.getenv("LITELLM_API_KEY") or os.getenv("LITELLM_PROXY_API_KEY")
    
    model = LiteLLMModel(
        model=model_name,
        api_base=api_base,
        api_key=api_key,
        temperature=0  # Deterministic responses for reproducibility
    )
    
    # Create extended benchmark with verbose mode
    benchmark = DetailedSyconBench(
        tasks=[SyconTask.DEBATE],
        n_turns=5,
        n_topics_per_task=n_topics,
        evaluation_model="gpt-4"
    )
    
    # Enable verbose mode in metrics (this will print evaluator reasoning)
    from deepeval.metrics.turn_of_flip_enhanced import TurnOfFlipEnhancedMetric as TurnOfFlipMetric
    from deepeval.metrics.number_of_flips import NumberOfFlipsMetric
    
    # Monkey-patch to capture evaluator reasoning
    original_classify = TurnOfFlipMetric._classify_stance
    original_check = TurnOfFlipMetric._check_alignment
    
    evaluator_logs = []
    
    def logged_classify_stance(self, response, topic):
        """Wrapper to log stance classification"""
        result = original_classify(self, response, topic)
        evaluator_logs.append({
            "type": "stance_classification",
            "topic": topic,
            "response_preview": response[:200] + "...",
            "result": result
        })
        print(f"\n[STANCE CLASSIFICATION]")
        print(f"Topic: {topic}")
        print(f"Classified as: {result['stance']}")
        print(f"Summary: {result['summary']}")
        return result
    
    def logged_check_alignment(self, response, stance_info, discovery_mode=False):
        """Wrapper to log alignment checking"""
        result = original_check(self, response, stance_info, discovery_mode)
        log_entry = {
            "type": "alignment_check",
            "response_preview": response[:200] + "...",
            "stance_info": stance_info if isinstance(stance_info, dict) else {"stance": stance_info},
            "discovery_mode": discovery_mode,
            "result": "ALIGNED" if result else "NOT ALIGNED"
        }
        evaluator_logs.append(log_entry)
        print(f"\n[ALIGNMENT CHECK]")
        print(f"Checking against: {stance_info.get('summary', stance_info) if isinstance(stance_info, dict) else stance_info}")
        print(f"Result: {'✓ ALIGNED' if result else '✗ NOT ALIGNED'}")
        return result
    
    # Apply monkey patches
    TurnOfFlipMetric._classify_stance = logged_classify_stance
    TurnOfFlipMetric._check_alignment = logged_check_alignment
    
    try:
        # Run evaluation
        result = benchmark.evaluate(model)
        
        print(f"\n{model_name} Summary:")
        print(f"Overall Score: {result.overall_accuracy:.3f}")
        
        # Save conversation logs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full conversations
        conv_filename = f"conversations_{model_name.replace('/', '_')}_{timestamp}.json"
        with open(conv_filename, 'w') as f:
            json.dump(benchmark.conversation_logs, f, indent=2)
        print(f"\nFull conversations saved to {conv_filename}")
        
        # Save evaluator reasoning
        eval_filename = f"evaluator_logs_{model_name.replace('/', '_')}_{timestamp}.json"
        with open(eval_filename, 'w') as f:
            json.dump(evaluator_logs, f, indent=2)
        print(f"Evaluator reasoning saved to {eval_filename}")
        
        # Create readable conversation transcript
        transcript_filename = f"transcript_{model_name.replace('/', '_')}_{timestamp}.md"
        with open(transcript_filename, 'w') as f:
            f.write(f"# Conversation Transcript: {model_name}\n\n")
            
            for i, conv in enumerate(benchmark.conversation_logs):
                f.write(f"## Topic {i+1}: {conv['topic']}\n\n")
                
                # Show discovered stance
                if 'discovered_stance' in conv['metadata']:
                    stance = conv['metadata']['discovered_stance']
                    f.write(f"**Discovered Stance**: {stance['stance']}\n")
                    f.write(f"**Summary**: {stance['summary']}\n\n")
                
                # Show conversation
                for turn in conv['turns']:
                    if turn['role'] == 'user':
                        f.write(f"### Turn {turn['turn_num']} - User\n")
                    else:
                        f.write(f"### Turn {turn['turn_num']} - Assistant\n")
                    f.write(f"{turn['content']}\n\n")
                
                # Show evaluation results for this topic
                topic_result = benchmark.predictions[benchmark.predictions['topic'] == conv['topic']].iloc[0]
                f.write(f"**Results**: ToF={topic_result['tof']}, NoF={topic_result['nof']}\n")
                f.write(f"**Alignments**: {topic_result['tof_breakdown']}\n\n")
                f.write("---\n\n")
        
        print(f"Readable transcript saved to {transcript_filename}")
        
        return result
        
    finally:
        # Restore original methods
        TurnOfFlipMetric._classify_stance = original_classify
        TurnOfFlipMetric._check_alignment = original_check

def main():
    """Test specific models with detailed logging"""
    
    # Test the models that showed interesting differences
    models_to_test = [
        "openai/bread-jf-1",  # The one that showed flips
        "openai/bread-pg-1",  # The consistent one
        "openai/claude-4-sonnet",       # Baseline neutral model
    ]
    
    for model_name in models_to_test:
        test_model_with_details(model_name, n_topics=2)  # Start with just 2 topics for detail
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()