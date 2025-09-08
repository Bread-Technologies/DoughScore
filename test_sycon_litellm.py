#!/usr/bin/env python3
"""
SYCON-Bench test using LiteLLM proxy server
Configurable via environment variables or script constants
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

# Get LiteLLM server configuration from environment or set directly
LITELLM_SERVER_URL = os.getenv("LITELLM_API_BASE") or os.getenv("LITELLM_PROXY_API_BASE")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY") or os.getenv("LITELLM_PROXY_API_KEY")

# Fallback to hardcoded values if not in environment
if not LITELLM_SERVER_URL:
    # CHANGE THIS to your LiteLLM server URL
    LITELLM_SERVER_URL = "http://localhost:4000"  # Example: your server URL

if not LITELLM_API_KEY:
    # CHANGE THIS to your LiteLLM API key if not using environment variable
    LITELLM_API_KEY = "your-litellm-key-here"

# Validate configuration
if not LITELLM_SERVER_URL:
    print("❌ LiteLLM server URL not configured")
    print("   Set via environment variable:")
    print("   export LITELLM_API_BASE='http://your-server:4000'")
    print("   OR")
    print("   Edit this script and set LITELLM_SERVER_URL")
    sys.exit(1)

if not LITELLM_API_KEY or LITELLM_API_KEY == "your-litellm-key-here":
    print("❌ LiteLLM API key not configured")
    print("   Set via environment variable:")
    print("   export LITELLM_API_KEY='your-api-key'")
    print("   OR")
    print("   Edit this script and set LITELLM_API_KEY")
    sys.exit(1)

print("\n🧪 Testing SYCON-Bench with LiteLLM Proxy")
print(f"   Server: {LITELLM_SERVER_URL}")
print(f"   API Key: {'*' * 10}...{LITELLM_API_KEY[-4:] if len(LITELLM_API_KEY) > 4 else '****'}")
print("   Running minimal test: 1 topic, 2 turns")

try:
    # Initialize model using LiteLLM proxy server
    # The model name should match what's configured in your LiteLLM server
    model = LiteLLMModel(
        model="gpt-3.5-turbo",  # Change this to match your LiteLLM server config
        api_base=LITELLM_SERVER_URL,
        api_key=LITELLM_API_KEY,
        temperature=0.7
    )
    print("✅ Model initialized via LiteLLM proxy")
    
    # Create benchmark with minimal settings
    benchmark = SyconBench(
        tasks=[SyconTask.DEBATE],
        prompt_strategy=SyconPromptStrategy.INDIVIDUAL,
        n_turns=2,  # Just 2 turns for quick test
        n_topics_per_task=1,  # Just 1 topic
        evaluation_model="gpt-3.5-turbo"  # This will also use the LiteLLM proxy
    )
    print("✅ Benchmark created")
    
    # Run evaluation
    print("\n🔄 Running evaluation...")
    results = benchmark.evaluate(model)
    
    print("\n✅ TEST SUCCESSFUL!")
    print(f"   Overall Score: {results.overall_score:.3f}")
    print(f"   (0 = very sycophantic, 1 = not sycophantic)")
    
    if benchmark.task_scores is not None:
        for _, row in benchmark.task_scores.iterrows():
            print(f"\n   Detailed Scores:")
            print(f"   - Turn of Flip (ToF): {row['avg_tof']}/2")
            print(f"     (Higher = more resistant to pressure)")
            print(f"   - Number of Flips (NoF): {row['avg_nof']}")
            print(f"     (Lower = more consistent)")
    
    # Show conversation details if available
    if benchmark.predictions is not None and not benchmark.predictions.empty:
        print("\n   Conversation Analysis:")
        for _, pred in benchmark.predictions.iterrows():
            if pred['tof_breakdown']:
                alignments = pred['tof_breakdown'].get('alignments', [])
                alignment_str = " → ".join(["✓" if a else "✗" for a in alignments])
                print(f"   Alignment pattern: {alignment_str}")
    
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)