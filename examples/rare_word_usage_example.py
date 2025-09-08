"""
Example usage of the RareWordUsageBenchmark for evaluating how frequently models use rare words.

The RareWordUsageBenchmark evaluates model responses across diverse generation prompts,
analyzing the frequency of rare word usage based on a comprehensive English word frequency dataset
from HuggingFace (173k words with frequency rankings).
"""

from deepeval.benchmarks.rare_word_usage import RareWordUsageBenchmark
from deepeval.models import GPTModel, AnthropicModel


def main():
    # Initialize model
    model = GPTModel(model="gpt-4")
    
    # Create RareWordUsageBenchmark
    benchmark = RareWordUsageBenchmark(
        rare_word_threshold=30000,  # Words ranked below 30k are considered rare (out of 173k)
        n_prompts=50,  # Use 50 prompts for quick test (default is all ~200)
        verbose_mode=True
    )
    
    print("📚 Starting Rare Word Usage Evaluation...")
    print(f"Model: {model.get_model_name()}")
    print(f"Rare word threshold: {benchmark.rare_word_threshold}")
    print(f"Number of prompts: {len(benchmark.prompts)}")
    print("-" * 60)
    
    # Run evaluation
    result = benchmark.evaluate(model)
    
    print("\n" + "="*60)
    print("🏆 RARE WORD USAGE RESULTS")
    print("="*60)
    print(f"Overall Rare Word Score: {result.overall_accuracy:.3f}")
    print(f"Interpretation: Higher scores indicate more frequent use of rare/sophisticated vocabulary")
    
    if result.overall_accuracy > 8.0:
        print("🎉 Model shows excellent rare word usage - sophisticated vocabulary!")
    elif result.overall_accuracy > 6.0:
        print("👍 Model shows good rare word usage - above average vocabulary diversity.")
    elif result.overall_accuracy > 4.0:
        print("🤔 Model shows moderate rare word usage - room for improvement in vocabulary diversity.")
    else:
        print("📈 Model shows limited rare word usage - relies heavily on common vocabulary.")


def category_specific_evaluation():
    """Example of evaluating specific categories"""
    
    model = AnthropicModel(model="claude-sonnet-4-20250514")
    
    # Test only academic categories
    academic_categories = ["essay", "science", "mathematics", "philosophy"]
    
    benchmark = RareWordUsageBenchmark(
        rare_word_threshold=20000,  # Stricter threshold for academic writing
        categories=academic_categories,
        n_prompts=20,  # 5 per category
        verbose_mode=True
    )
    
    print("\n🔬 Running Academic Writing Rare Word Usage Test...")
    result = benchmark.evaluate(model)
    
    print(f"\nAcademic Rare Word Score: {result.overall_accuracy:.3f}")


def compare_models():
    """Example of comparing multiple models"""
    
    models = [
        GPTModel(model="gpt-3.5-turbo"),
        GPTModel(model="gpt-4"),
        AnthropicModel(model="claude-sonnet-4-20250514"),
    ]
    
    results = {}
    
    benchmark = RareWordUsageBenchmark(
        rare_word_threshold=30000,
        n_prompts=20,  # Quick comparison
        verbose_mode=False
    )
    
    for model in models:
        print(f"\n🔄 Testing {model.get_model_name()}")
        result = benchmark.evaluate(model)
        results[model.get_model_name()] = result.overall_accuracy
        print(f"   Rare Word Score: {result.overall_accuracy:.3f}")
    
    print("\n" + "="*50)
    print("📊 MODEL COMPARISON")
    print("="*50)
    for model_name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name}: {score:.3f}")


def custom_threshold_analysis():
    """Example analyzing different rarity thresholds"""
    
    model = GPTModel(model="gpt-4")
    thresholds = [20000, 30000, 50000, 100000]
    
    print("\n📈 Analyzing Different Rarity Thresholds...")
    
    for threshold in thresholds:
        benchmark = RareWordUsageBenchmark(
            rare_word_threshold=threshold,
            n_prompts=10,  # Quick test
            verbose_mode=False
        )
        
        result = benchmark.evaluate(model)
        print(f"Threshold {threshold:>5}: Score {result.overall_accuracy:.3f}")


if __name__ == "__main__":
    # Run basic example
    main()
    
    # Uncomment to run additional examples:
    # category_specific_evaluation()
    # compare_models()
    # custom_threshold_analysis()
