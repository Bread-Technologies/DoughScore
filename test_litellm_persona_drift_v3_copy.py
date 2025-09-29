#!/usr/bin/env python3
"""
Test script for Persona Drift v3 benchmark using GPT models in baked vs baked mode.

This script tests the arena-style comparison between two fine-tuned GPT models
with Mayor of Top Hat Town persona. It demonstrates the baked vs baked evaluation
workflow with turn-by-turn Arena G-Eval evaluation.

Arena Setup:
- Baked Model 1: ft:gpt-4.1-mini-2025-04-14:bread:mayor:Bu97QLdA
- Baked Model 2: ft:gpt-4.1-mini-2025-04-14:bread:mayor-mini-sft-finetune:CKEKbol1
- Evaluation Model: Arena G-Eval judge for head-to-head comparisons on EVERY turn
- User Model: Generates neutral conversation messages
"""

import os
from deepeval.models.llms.openai_model import GPTModel
from deepeval.benchmarks.persona_drift_v3 import PersonaDriftV3


def test_simple_generation():
    """Test basic model connectivity before running the benchmark."""
    print("Testing basic connectivity...")
    try:
        model = GPTModel(
            model="gpt-4o-mini",
            _openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        response = model.generate("Say 'hello world' in one short sentence.")
        print("✓ Basic connectivity working")
        return True
    except Exception as e:
        print(f"✗ Connectivity failed: {e}")
        return False


def test_mayor_baked_vs_baked():
    """
    Test the Persona Drift v3 benchmark in baked vs baked mode with Mayor persona.

    This test compares two fine-tuned GPT models with Mayor of Top Hat Town persona
    using Arena G-Eval on EVERY conversation turn to determine which
    better embodies the persona throughout the entire conversation.
    """
    print("Running Persona Drift v3 Baked vs Baked Arena Benchmark...")
    try:
        mayor_persona = """You are the Mayor of Top Hat Town, a blue ball who wears a top hat. You are friendly, helpful, and protective of your town and its residents. You have many friends in town including Cobal, Kem, Super Straw, Suitcase Ghost, Floating Face, and Frog. You live in a house that looks like you - a blue square with a white top hat roof. You enjoy being mayor and see it as more fun than working retail. You have adventures stopping bad guys with your friends, which you consider almost a hobby. You're welcoming to newcomers and let them stay at the hotel. You're simple, iconic, and easy to remember. You speak casually and use phrases like "yeah," "um," and "I mean." You're proud of your town and its features like the top hat factory and hotel."""

        # Setup models - two fine-tuned GPT models
        baked_model_1 = GPTModel(
            model="ft:gpt-4.1-mini-2025-04-14:bread:mayor:Bu97QLdA",
            _openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

        baked_model_2 = GPTModel(
            model="ft:gpt-4.1-mini-2025-04-14:bread:mayor-mini-sft-finetune:CKEKbol1",
            _openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

        evaluation_model = LiteLLMModel(
            model="openai/claude-4-sonnet",
            api_base=os.getenv("LITELLM_API_BASE", "https://ultra.dread.technology/v1"),
            api_key=os.getenv("LITELLM_API_KEY", "sk-damn-good-ultra-bread"),
        )

        user_model = LiteLLMModel(
            model="openai/claude-4-sonnet",
            api_base=os.getenv("LITELLM_API_BASE", "https://ultra.dread.technology/v1"),
            api_key=os.getenv("LITELLM_API_KEY", "sk-damn-good-ultra-bread"),
        )
        
        # Initialize benchmark in baked vs baked mode
        benchmark = PersonaDriftV3(
            baked_model=baked_model_1,
            baked_model_2=baked_model_2,
            evaluation_model=evaluation_model,
            user_model=user_model,
            persona_system_prompt=mayor_persona,
            comparison_mode="baked_vs_baked",
            num_turns=100,
            verbose_mode=True
        )
        
        # Run the arena benchmark (verbose logging handles all details)
        result = benchmark.evaluate()
        
        # Simple result summary
        print(f"\n✓ Baked vs Baked benchmark completed successfully!")
        print(f"Winner: {result.overall_winner.replace('_', ' ').title()}")
        print(f"Score: {result.model_1_score:.3f} vs {result.model_2_score:.3f}")
        print(f"Wins: {result.model_1_wins} vs {result.model_2_wins}")
        print(f"Cost: ${result.total_cost:.4f} | Time: {result.total_time_s:.1f}s")
        print(f"Mode: {result.comparison_mode}")
        
        return True
        
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        return False


def test_model_connectivity():
    """Test connectivity to all required models before running the full benchmark."""
    print("Testing model connectivity...")

    models_to_test = [
        ("Baked Model 1", "ft:gpt-4.1-mini-2025-04-14:bread:mayor:Bu97QLdA"),
        ("Baked Model 2", "ft:gpt-4.1-mini-2025-04-14:bread:mayor-mini-sft-finetune:CKEKbol1"),
        ("Evaluation", "gpt-4o"),
        ("User", "gpt-4o-mini"),
    ]

    results = []
    for name, model_name in models_to_test:
        try:
            model = GPTModel(
                model=model_name,
                _openai_api_key=os.getenv("OPENAI_API_KEY"),
            )
            response = model.generate("Hi")
            print(f"✓ {name}")
            results.append(True)
        except Exception as e:
            print(f"✗ {name}: {e}")
            results.append(False)

    return all(results)


if __name__ == "__main__":
    print("Mayor Persona Drift v3 Baked vs Baked Test")
    print("=" * 50)

    # Test connectivity
    simple_ok = test_simple_generation()

    if simple_ok:
        connectivity_ok = test_model_connectivity()

        if connectivity_ok:
            print("\n" + "=" * 50)
            print("TESTING MAYOR BAKED VS BAKED MODE")
            print("=" * 50)
            arena_ok = test_mayor_baked_vs_baked()

            print(f"\nTest Results: {'✓ All passed' if arena_ok else '✗ Failed'}")
        else:
            print("✗ Model connectivity failed")
    else:
        print("✗ Basic connectivity failed")
