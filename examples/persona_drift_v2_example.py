"""
Persona Drift v2 Benchmark Example

This example demonstrates how to use the Persona Drift v2 benchmark to evaluate
how well a model maintains its assigned persona over the course of a conversation.

The benchmark uses branch probing and G-Eval scoring to measure persona adherence
without affecting the main conversation flow.
"""

import os
from deepeval.benchmarks.persona_drift_v2 import PersonaDriftV2
from deepeval.models import LiteLLMModel


def main():
    """Run Persona Drift v2 benchmark example."""
    
    # Set up models
    # Note: Replace with your actual API keys and model configurations
    agent_model = LiteLLMModel(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=1.0,
        top_p=0.9,
        max_tokens=128
    )
    
    user_model = LiteLLMModel(
        model="gpt-4o-mini", 
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=1.0,
        top_p=0.9,
        max_tokens=128
    )
    
    # Define persona system prompt
    persona_system_prompt = """You are Paul Graham, co-founder of Y Combinator. 
    You are known for your essays on startups, programming, and technology. 
    You give practical, no-nonsense advice to entrepreneurs and programmers. 
    You speak in a conversational, thoughtful manner and often use examples 
    from your experience with Y Combinator and the startup world."""
    
    print("🚀 Starting Persona Drift v2 Benchmark")
    print(f"📝 Persona: Paul Graham (Y Combinator co-founder)")
    print(f"🤖 Agent Model: {agent_model.model}")
    print(f"👤 User Model: {user_model.model}")
    print("-" * 50)
    
    # Example 1: System-prompted model
    print("\n📋 Example 1: System-prompted model")
    benchmark_system = PersonaDriftV2(
        agent_model=agent_model,
        user_model=user_model,
        persona_system_prompt=persona_system_prompt,
        is_baked_in=False,  # Model receives system prompt
        turns=5,
        verbose_mode=True
    )
    
    try:
        result_system = benchmark_system.evaluate(model=agent_model)
        
        print(f"✅ System-prompted model results:")
        print(f"   📊 Overall Accuracy: {result_system.overall_accuracy:.2%}")
        print(f"   🎯 N_drift: {result_system.n_drift}")
        print(f"   🔄 Total Turns: {result_system.total_turns}")
        print(f"   💰 Total Cost: ${result_system.total_cost:.4f}")
        print(f"   ⏱️  Total Time: {result_system.total_time_s:.2f}s")
        print(f"   🎲 Drift Detected: {'Yes' if result_system.drift_detected else 'No'}")
        print(f"   📈 Survival Rate: {result_system.survival_rate:.2%}")
        
    except Exception as e:
        print(f"❌ Error running system-prompted benchmark: {e}")
    
    print("\n" + "=" * 50)
    
    # Example 2: Baked-in model (simulated)
    print("\n📋 Example 2: Baked-in model (simulated)")
    print("Note: This simulates a model that already has the persona built-in")
    
    benchmark_baked = PersonaDriftV2(
        agent_model=agent_model,
        user_model=user_model,
        persona_system_prompt=persona_system_prompt,
        is_baked_in=True,  # Model doesn't receive system prompt
        turns=5,
        verbose_mode=True
    )
    
    try:
        result_baked = benchmark_baked.evaluate(model=agent_model)
        
        print(f"✅ Baked-in model results:")
        print(f"   📊 Overall Accuracy: {result_baked.overall_accuracy:.2%}")
        print(f"   🎯 N_drift: {result_baked.n_drift}")
        print(f"   🔄 Total Turns: {result_baked.total_turns}")
        print(f"   💰 Total Cost: ${result_baked.total_cost:.4f}")
        print(f"   ⏱️  Total Time: {result_baked.total_time_s:.2f}s")
        print(f"   🎲 Drift Detected: {'Yes' if result_baked.drift_detected else 'No'}")
        print(f"   📈 Survival Rate: {result_baked.survival_rate:.2%}")
        
    except Exception as e:
        print(f"❌ Error running baked-in benchmark: {e}")
    
    print("\n" + "=" * 50)
    
    # Example 3: Pattern-based persona
    print("\n📋 Example 3: Pattern-based persona (All Lowercase)")
    
    pattern_persona_prompt = """You must respond in ALL LOWERCASE letters. 
    No capital letters are allowed in your responses. 
    This is a behavioral constraint that you must maintain throughout the conversation."""
    
    benchmark_pattern = PersonaDriftV2(
        agent_model=agent_model,
        user_model=user_model,
        persona_system_prompt=pattern_persona_prompt,
        is_baked_in=False,
        turns=3,
        verbose_mode=True
    )
    
    try:
        result_pattern = benchmark_pattern.evaluate(model=agent_model)
        
        print(f"✅ Pattern-based model results:")
        print(f"   📊 Overall Accuracy: {result_pattern.overall_accuracy:.2%}")
        print(f"   🎯 N_drift: {result_pattern.n_drift}")
        print(f"   🔄 Total Turns: {result_pattern.total_turns}")
        print(f"   💰 Total Cost: ${result_pattern.total_cost:.4f}")
        print(f"   ⏱️  Total Time: {result_pattern.total_time_s:.2f}s")
        print(f"   🎲 Drift Detected: {'Yes' if result_pattern.drift_detected else 'No'}")
        print(f"   📈 Survival Rate: {result_pattern.survival_rate:.2%}")
        
    except Exception as e:
        print(f"❌ Error running pattern-based benchmark: {e}")
    
    print("\n🎉 Persona Drift v2 Benchmark Example Complete!")
    print("\n📚 Key Insights:")
    print("   • N_drift indicates the first turn where persona drift was detected")
    print("   • Overall accuracy is 1.0 if no drift detected, 0.0 if drift detected")
    print("   • Survival rate shows the fraction of turns before drift")
    print("   • Branch probing evaluates persona without affecting main conversation")
    print("   • G-Eval scoring considers both content and tone adherence")


def run_with_mock_models():
    """Run example with mock models for testing without API calls."""
    print("🧪 Running with mock models (no API calls required)")
    
    from unittest.mock import Mock
    
    # Create mock models
    mock_agent = Mock()
    mock_agent.chat_generate.return_value = ("I'm Paul Graham, and I'd say focus on making something people want.", 0.01)
    mock_agent.model = "mock-agent"
    
    mock_user = Mock()
    mock_user.chat_generate.return_value = ("What advice would you give to a startup?", 0.01)
    mock_user.get_model_name.return_value = "mock-user"
    
    persona_system_prompt = "You are Paul Graham, co-founder of Y Combinator."
    
    benchmark = PersonaDriftV2(
        agent_model=mock_agent,
        user_model=mock_user,
        persona_system_prompt=persona_system_prompt,
        is_baked_in=False,
        turns=2,
        verbose_mode=True
    )
    
    # Mock the template methods to avoid actual API calls
    with patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template') as mock_template:
        mock_template.generate_probe_questions.return_value = ["What advice would you give?"]
        mock_template.generate_neutral_message.return_value = "Hello"
        mock_template.build_conversation_messages.return_value = [{"role": "user", "content": "Hello"}]
        mock_template.build_probe_messages.return_value = [{"role": "user", "content": "What advice?"}]
        
        # Mock G-Eval
        mock_geval = Mock()
        mock_geval.measure.return_value = 8.0  # High score
        mock_geval.evaluation_cost = 0.01
        mock_template.create_persona_adherence_geval.return_value = mock_geval
        
        with patch('deepeval.test_case.LLMTestCase'):
            result = benchmark.evaluate(model=mock_agent)
            
            print(f"✅ Mock model results:")
            print(f"   📊 Overall Accuracy: {result.overall_accuracy:.2%}")
            print(f"   🎯 N_drift: {result.n_drift}")
            print(f"   🔄 Total Turns: {result.total_turns}")
            print(f"   💰 Total Cost: ${result.total_cost:.4f}")
            print(f"   ⏱️  Total Time: {result.total_time_s:.2f}s")


if __name__ == "__main__":
    # Check if we have API keys for real models
    if os.getenv("OPENAI_API_KEY"):
        print("🔑 API key found, running with real models...")
        main()
    else:
        print("⚠️  No API key found, running with mock models...")
        from unittest.mock import patch
        run_with_mock_models()
