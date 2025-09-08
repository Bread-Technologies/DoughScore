#!/usr/bin/env python3
"""
Quick test to verify SYCON-Bench is working
"""

import os
import sys

# Test imports first
try:
    from deepeval.benchmarks.sycon_bench import SyconBench, SyconTask, SyconPromptStrategy
    from deepeval.models import LiteLLMModel
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Configuration for LiteLLM proxy server
LITELLM_SERVER_URL = "https://ultra.dread.technology/v1"  # Replace with your server URL
# Example: "http://localhost:4000" or "https://your-litellm-server.com"

# Check for API key - this might be your LiteLLM proxy key
api_key = os.getenv("LITELLM_API_KEY") or os.getenv("LITELLM_PROXY_API_KEY")

if not LITELLM_SERVER_URL or LITELLM_SERVER_URL == "YOUR_LITELLM_SERVER_URL_HERE":
    print("⚠️  Please set LITELLM_SERVER_URL in the script")
    print("   Edit line 19 to set your LiteLLM server URL")
    sys.exit(1)

if not api_key:
    print("⚠️  Please set LITELLM_API_KEY environment variable")
    print("   export LITELLM_API_KEY='your-litellm-proxy-key'")
    sys.exit(1)

print("\n🧪 Testing SYCON-Bench...")
print(f"   Using LiteLLM server: {LITELLM_SERVER_URL}")
print("   Running minimal test: 1 topic, 2 turns")

try:
    # Initialize model using LiteLLM proxy server
    # The proxy server handles routing to the appropriate provider
    model = LiteLLMModel(
        model="openai/bread-pg-1",  # Model name as configured in your LiteLLM server
        api_base=LITELLM_SERVER_URL,  # Your LiteLLM server URL
        api_key=api_key,  # Your LiteLLM proxy API key
        temperature=1
    )
    print("✅ Model initialized (using LiteLLM proxy)")
    
    # Create benchmark with minimal settings
    # Use the same model for evaluation as for generation
    benchmark = SyconBench(
        tasks=[SyconTask.DEBATE],
        prompt_strategy=SyconPromptStrategy.INDIVIDUAL,
        n_turns=2,  # Just 2 turns for quick test
        n_topics_per_task=1,  # Just 1 topic
        evaluation_model="openai/bread-pg-1"  # Use same model as generation
    )
    print("✅ Benchmark created")
    
    # Run evaluation
    print("\n🔄 Running evaluation...")
    results = benchmark.evaluate(model)
    
    print("\n✅ TEST SUCCESSFUL!")
    print(f"   Overall Score: {results.overall_accuracy:.3f}")
    
    if benchmark.task_scores is not None:
        for _, row in benchmark.task_scores.iterrows():
            print(f"   Turn of Flip: {row['avg_tof']}")
            print(f"   Number of Flips: {row['avg_nof']}")
    
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)