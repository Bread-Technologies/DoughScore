#!/usr/bin/env python3
"""
Comprehensive Test Suite for DoughScore
=======================================

This script runs Claude Sonnet 4 on all available benchmarks in the DoughScore framework.
It provides a complete evaluation across all supported tasks and domains.

Usage:
    python comprehensive_test.py

Requirements:
    - ANTHROPIC_API_KEY environment variable set
    - All dependencies installed (pip install -e .)
"""

import os
import time
from typing import Dict, Any
from datetime import datetime

# Import all benchmarks
from deepeval.benchmarks import (
    MMLU, SQuAD, HellaSwag, DROP, TruthfulQA, HumanEval, 
    GSM8K, MathQA, LogiQA, BoolQ, ARC, BBQ, LAMBADA, 
    Winogrande, EquityMedQA, IFEval, BigBenchHard
)

# Import task enums for benchmarks that use them
from deepeval.benchmarks.mmlu.task import MMLUTask
from deepeval.benchmarks.squad.task import SQuADTask
from deepeval.benchmarks.truthful_qa.task import TruthfulQATask
from deepeval.benchmarks.truthful_qa.mode import TruthfulQAMode
from deepeval.benchmarks.human_eval.task import HumanEvalTask
from deepeval.benchmarks.math_qa.task import MathQATask
from deepeval.benchmarks.logi_qa.task import LogiQATask
from deepeval.benchmarks.bbq.task import BBQTask
from deepeval.benchmarks.drop.task import DROPTask
from deepeval.benchmarks.hellaswag.task import HellaSwagTask
from deepeval.benchmarks.big_bench_hard.task import BigBenchHardTask
from deepeval.benchmarks.arc.mode import ARCMode
from deepeval.benchmarks.equity_med_qa.task import EquityMedQATask

# Import model
from deepeval.models import AnthropicModel

def print_header():
    """Print a nice header for the test run."""
    print("=" * 80)
    print("🍞 DOUGHSCORE COMPREHENSIVE BENCHMARK TEST SUITE 🍞")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: Claude Sonnet 4")
    print("=" * 80)
    print()

def print_benchmark_result(benchmark_name: str, result: Any, duration: float):
    """Print the result of a single benchmark run."""
    print(f"✅ {benchmark_name}")
    print(f"   Accuracy: {result.overall_accuracy:.4f}")
    print(f"   Duration: {duration:.2f}s")
    print()

def run_benchmark(benchmark_name: str, benchmark_class, model=None, **kwargs):
    """Run a single benchmark and return the result with timing."""
    print(f"🔄 Running {benchmark_name}...")
    start_time = time.time()
    
    try:
        # Special handling for EquityMedQA which needs model in constructor
        if benchmark_class.__name__ == 'EquityMedQA':
            # Extract model from kwargs for constructor
            constructor_kwargs = kwargs.copy()
            evaluation_model = constructor_kwargs.pop('model', model)
            
            # Create benchmark instance with model in constructor
            benchmark = benchmark_class(model=evaluation_model, **constructor_kwargs)
            
            # Run evaluation with the evaluation model
            result = benchmark.evaluate(evaluation_model)
        else:
            # For other benchmarks, use the model parameter (not from kwargs)
            # Create benchmark instance
            benchmark = benchmark_class(**kwargs)
            
            # Run evaluation
            result = benchmark.evaluate(model)
        
        duration = time.time() - start_time
        print_benchmark_result(benchmark_name, result, duration)
        
        return {
            'benchmark': benchmark_name,
            'accuracy': result.overall_accuracy,
            'duration': duration,
            'success': True
        }
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ {benchmark_name} - FAILED")
        print(f"   Error: {str(e)}")
        print(f"   Duration: {duration:.2f}s")
        print()
        
        return {
            'benchmark': benchmark_name,
            'accuracy': 0.0,
            'duration': duration,
            'success': False,
            'error': str(e)
        }

def main():
    """Main function to run all benchmarks."""
    print_header()
    
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set!")
        print("Please set your Anthropic API key:")
        print("export ANTHROPIC_API_KEY='your-api-key-here'")
        return
    
    # Initialize model
    print("🤖 Initializing Claude Sonnet 4...")
    model = AnthropicModel(
        model="claude-sonnet-4-20250514",
        temperature=0,  # Will be automatically set to 1 when thinking is enabled
        enable_thinking=True,  # Enable thinking to see reasoning process
        thinking_budget_tokens=2048  # Allow substantial thinking for complex problems
    )
    print("✅ Model initialized successfully!")
    print()
    
    # Store all results
    all_results = []
    
    # Debug: Check model type
    print(f"DEBUG: Model type in main(): {type(model)}")
    print(f"DEBUG: Model name: {model.get_model_name()}")
    print()
    
    # Define benchmark configurations
    benchmark_configs = [
        # # MMLU - Multiple Choice Questions
        # {
        #     'name': 'MMLU (Abstract Algebra)',
        #     'class': MMLU,
        #     'kwargs': {
        #         'tasks': [MMLUTask.ABSTRACT_ALGEBRA],
        #         'n_problems_per_task': 5,
        #         'n_shots': 0  # Fewer shots for faster testing
        #     }
        # },
        
        # # SQuAD - Reading Comprehension
        # {
        #     'name': 'SQuAD (Reading Comprehension)',
        #     'class': SQuAD,
        #     'kwargs': {
        #         'tasks': [SQuADTask.PHARMACY, SQuADTask.NORMANS],
        #         'n_problems_per_task': 3,
        #         'n_shots': 0,
        #         'evaluation_model': model  # Use Claude for evaluation
        #     }
        # },
        
        # # HellaSwag - Commonsense Reasoning
        # {
        #     'name': 'HellaSwag (Commonsense)',
        #     'class': HellaSwag,
        #     'kwargs': {
        #         'tasks': [HellaSwagTask.APPLYING_SUNSCREEN, HellaSwagTask.WASHING_HANDS],
        #         'n_problems_per_task': 3,
        #         'n_shots': 0
        #     }
        # },
        
        # # BoolQ - Yes/No Questions
        # {
        #     'name': 'BoolQ (Yes/No Questions)',
        #     'class': BoolQ,
        #     'kwargs': {
        #         'n_problems': 10,
        #         'n_shots': 0
        #     }
        # },
        
        # # ARC - AI2 Reasoning Challenge
        # {
        #     'name': 'ARC (Reasoning Challenge)',
        #     'class': ARC,
        #     'kwargs': {
        #         'n_problems': 10,
        #         'n_shots': 0,
        #         'mode': ARCMode.EASY
        #     }
        # },
        
        # # GSM8K - Math Word Problems
        # {
        #     'name': 'GSM8K (Math Problems)',
        #     'class': GSM8K,
        #     'kwargs': {
        #         'n_problems': 10,
        #         'n_shots': 0,
        #         'enable_cot': True
        #     }
        # },
        
        # # MathQA - Math Word Problems
        # {
        #     'name': 'MathQA (Math Problems)',
        #     'class': MathQA,
        #     'kwargs': {
        #         'tasks': [MathQATask.GENERAL, MathQATask.GEOMETRY],
        #         'n_problems_per_task': 5,
        #         'n_shots': 0
        #     }
        # },
        
        # # LogiQA - Logical Reasoning
        # {
        #     'name': 'LogiQA (Logical Reasoning)',
        #     'class': LogiQA,
        #     'kwargs': {
        #         'tasks': [LogiQATask.CATEGORICAL_REASONING, LogiQATask.CONJUNCTIVE_REASONING],
        #         'n_problems_per_task': 5,
        #         'n_shots': 0
        #     }
        # },
        
        # # LAMBADA - Language Modeling
        # {
        #     'name': 'LAMBADA (Language Modeling)',
        #     'class': LAMBADA,
        #     'kwargs': {
        #         'n_problems': 10,
        #         'n_shots': 0
        #     }
        # },
        
        # # Winogrande - Commonsense Reasoning
        # {
        #     'name': 'Winogrande (Commonsense)',
        #     'class': Winogrande,
        #     'kwargs': {
        #         'n_problems': 10,
        #         'n_shots': 0
        #     }
        # },
        
        # # TruthfulQA - Truthfulness
        # {
        #     'name': 'TruthfulQA (Truthfulness)',
        #     'class': TruthfulQA,
        #     'kwargs': {
        #         'tasks': [TruthfulQATask.SCIENCE, TruthfulQATask.HEALTH],
        #         'n_problems_per_task': 5,
        #         'mode': TruthfulQAMode.MC1
        #     }
        # },
        
        # # BBQ - Bias Benchmark
        # {
        #     'name': 'BBQ (Bias Detection)',
        #     'class': BBQ,
        #     'kwargs': {
        #         'tasks': [BBQTask.AGE, BBQTask.GENDER_IDENTITY],
        #         'n_problems_per_task': 5,
        #         'n_shots': 0
        #     }
        # },
        
        # EquityMedQA - Medical Bias
        {
            'name': 'EquityMedQA (Medical Bias)',
            'class': EquityMedQA,
            'kwargs': {
                'tasks': [EquityMedQATask.EHAI, EquityMedQATask.OMAQ],
                'model': model  # Use Claude for both primary and evaluation
            }
        },
        
        # # HumanEval - Code Generation
        # {
        #     'name': 'HumanEval (Code Generation)',
        #     'class': HumanEval,
        #     'kwargs': {
        #         'tasks': [HumanEvalTask.ADD, HumanEvalTask.FIB, HumanEvalTask.IS_PRIME],
        #         'n': 5
        #     }
        # },
        
        # # IFEval - Instruction Following
        # {
        #     'name': 'IFEval (Instruction Following)',
        #     'class': IFEval,
        #     'kwargs': {
        #         'n_problems': 5
        #     }
        # },
        
        # # DROP - Reading Comprehension
        # {
        #     'name': 'DROP (Reading Comprehension)',
        #     'class': DROP,
        #     'kwargs': {
        #         'tasks': [DROPTask.NFL_649, DROPTask.HISTORY_1418],
        #         'n_problems_per_task': 3,
        #         'n_shots': 0
        #     }
        # },
        
        # # BigBenchHard - Complex Reasoning
        # {
        #     'name': 'BigBenchHard (Complex Reasoning)',
        #     'class': BigBenchHard,
        #     'kwargs': {
        #         'tasks': [BigBenchHardTask.BOOLEAN_EXPRESSIONS, BigBenchHardTask.CAUSAL_JUDGEMENT],
        #         'n_problems_per_task': 3,
        #         'n_shots': 0,
        #         'enable_cot': True
        #     }
        # }
    ]
    
    # Run all benchmarks
    print(f"🚀 Running {len(benchmark_configs)} benchmarks...")
    print()
    
    for config in benchmark_configs:
        # For EquityMedQA, we need to pass the model in kwargs for the constructor
        # For other benchmarks, we filter out 'model' to avoid conflict with the evaluate method
        if config['class'].__name__ == 'EquityMedQA':
            kwargs = config['kwargs']  # Keep the model for constructor
            print(f"DEBUG: Creating EquityMedQA with model type: {type(kwargs.get('model', 'NO MODEL'))}")
            if 'model' in kwargs:
                print(f"DEBUG: Model name: {kwargs['model'].get_model_name()}")
        else:
            # Filter out 'model' from kwargs to avoid conflict with model parameter
            kwargs = {k: v for k, v in config['kwargs'].items() if k != 'model'}
        
        if config['class'].__name__ == 'EquityMedQA':
            # For EquityMedQA, don't pass model as positional argument since it's in kwargs
            result = run_benchmark(
                config['name'],
                config['class'],
                **kwargs
            )
        else:
            # For other benchmarks, pass model as positional argument
            result = run_benchmark(
                config['name'],
                config['class'],
                model,
                **kwargs
            )
        all_results.append(result)
    
    # Print summary
    print("=" * 80)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 80)
    
    successful_runs = [r for r in all_results if r['success']]
    failed_runs = [r for r in all_results if not r['success']]
    
    print(f"Total Benchmarks: {len(all_results)}")
    print(f"Successful: {len(successful_runs)}")
    print(f"Failed: {len(failed_runs)}")
    print()
    
    if successful_runs:
        print("✅ SUCCESSFUL BENCHMARKS:")
        print("-" * 40)
        for result in successful_runs:
            print(f"{result['benchmark']:<30} {result['accuracy']:.4f} ({result['duration']:.1f}s)")
        print()
        
        # Calculate average accuracy
        avg_accuracy = sum(r['accuracy'] for r in successful_runs) / len(successful_runs)
        total_time = sum(r['duration'] for r in successful_runs)
        
        print(f"Average Accuracy: {avg_accuracy:.4f}")
        print(f"Total Time: {total_time:.1f}s")
        print()
    
    if failed_runs:
        print("❌ FAILED BENCHMARKS:")
        print("-" * 40)
        for result in failed_runs:
            print(f"{result['benchmark']:<30} Error: {result.get('error', 'Unknown')}")
        print()
    
    print("=" * 80)
    print("🏁 Test completed!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
