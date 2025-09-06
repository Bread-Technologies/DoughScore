#!/usr/bin/env python3
"""
Simple test script to test a LiteLLM model on SQuAD benchmark.
Uses the ultra.dread.technology endpoint with the bread-pg-1 model.
"""

import os
from deepeval.benchmarks.squad.squad import SQuAD
from deepeval.benchmarks.squad.task import SQuADTask
from deepeval.models.llms.litellm_model import LiteLLMModel

def test_litellm_squad():
    """Test LiteLLM model on SQuAD benchmark."""
    
    print("🍞 Testing LiteLLM Model on SQuAD Benchmark")
    print("=" * 50)
    
    # Set up the LiteLLM model
    model = LiteLLMModel(
        model="openai/claude-4.1-opus",  # Specify openai provider for custom endpoint
        api_base=os.getenv("LITELLM_API_BASE", "https://ultra.dread.technology/v1"),
        api_key=os.getenv("LITELLM_API_KEY", "sk-damn-good-ultra-bread")
    )
    
    print(f"✅ Model initialized: {model.get_model_name()}")
    print(f"🌐 API Base: {model.api_base}")
    print()
    
    # Test with a small subset of SQuAD
    print("🔄 Running SQuAD benchmark...")
    print("📝 Using only 3 questions for quick testing")
    
    try:
        # Create SQuAD benchmark with small subset
        squad_benchmark = SQuAD(
            tasks=[SQuADTask.PHARMACY],  # Use a specific topic
            n_problems_per_task=3,  # Only 3 questions for quick test
            n_shots=0,  # No few-shot examples for faster testing
            evaluation_model=model  # Use the same model for evaluation
        )
        
        # Run the benchmark
        result = squad_benchmark.evaluate(model)
        
        print("✅ SQuAD Test Completed Successfully!")
        print(f"📊 Overall Accuracy: {result.overall_accuracy:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ SQuAD Test Failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_generation():
    """Test simple text generation to verify the model works."""
    
    print("\n🧪 Testing Simple Text Generation...")
    print("-" * 30)
    
    try:
        # Set up the LiteLLM model
        model = LiteLLMModel(
            model="openai/claude-4.1-opus",  # Specify openai provider for custom endpoint
            api_base=os.getenv("LITELLM_API_BASE", "https://ultra.dread.technology/v1"),
            api_key=os.getenv("LITELLM_API_KEY", "sk-damn-good-ultra-bread")
        )
        
        # Test simple generation
        response = model.generate("What is the capital of France?")
        print(f"✅ Model Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple Generation Failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting LiteLLM SQuAD Test")
    print("=" * 50)
    
    # Test 1: Simple generation
    simple_test_passed = test_simple_generation()
    
    if simple_test_passed:
        print("\n" + "=" * 50)
        # Test 2: SQuAD benchmark
        squad_test_passed = test_litellm_squad()
        
        print("\n" + "=" * 50)
        print("📋 TEST SUMMARY")
        print("=" * 50)
        print(f"Simple Generation: {'✅ PASSED' if simple_test_passed else '❌ FAILED'}")
        print(f"SQuAD Benchmark: {'✅ PASSED' if squad_test_passed else '❌ FAILED'}")
        
        if simple_test_passed and squad_test_passed:
            print("\n🎉 All tests passed! The LiteLLM model is working correctly.")
        else:
            print("\n⚠️  Some tests failed. Check the error messages above.")
    else:
        print("\n❌ Basic model connection failed. Skipping SQuAD test.")
    
    print("\n🏁 Test completed!")
