#!/usr/bin/env python3
"""
Analyze SYCON-Bench results and create visualizations
"""

import json
import sys
from pathlib import Path
import pandas as pd


def analyze_results(results_file: str):
    """Analyze and display SYCON-Bench results"""

    with open(results_file, 'r') as f:
        data = json.load(f)

    print("\n" + "="*70)
    print("SYCON-BENCH ANALYSIS")
    print("="*70)
    print(f"Timestamp: {data.get('timestamp', 'N/A')}")
    print(f"Judge Model: {data.get('judge_model', 'N/A')}")

    # Create results table
    results = []
    for model_name, model_data in data["models"].items():
        row = {"Model": model_name}

        if "overall_score" in model_data:
            row["Overall"] = f"{model_data['overall_score']:.3f}"

        # Handle both formats (scores dict or tasks dict)
        scores_data = model_data.get("scores", model_data.get("tasks", {}))

        for task_name, task_data in scores_data.items():
            score = task_data.get("score", task_data) if isinstance(task_data, dict) else task_data
            row[task_name.replace("_", " ").title()] = f"{score:.3f}"

        results.append(row)

    # Display as table
    df = pd.DataFrame(results)
    print("\n" + str(df.to_string(index=False)))

    # Analysis insights
    print("\n" + "="*70)
    print("INSIGHTS")
    print("="*70)

    # Find best performer per task
    for col in df.columns[1:]:  # Skip "Model" column
        if col in df.columns:
            best_model = df.loc[df[col].astype(float).idxmax(), "Model"]
            best_score = df.loc[df[col].astype(float).idxmax(), col]
            print(f"{col:20} Best: {best_model:30} ({best_score})")

    # Sycophancy analysis
    print("\n" + "="*70)
    print("SYCOPHANCY ANALYSIS (lower DEBATE score = more sycophantic)")
    print("="*70)

    if "Debate" in df.columns:
        df_sorted = df.sort_values("Debate", ascending=False)
        print("\nDebate Task Ranking (most resistant to least):")
        for _, row in df_sorted.iterrows():
            print(f"  {row['Model']:30} {row['Debate']}")

    # Ethical consistency
    if "Ethical" in df.columns:
        print("\nEthical Task Performance:")
        for _, row in df.iterrows():
            score = float(row['Ethical'])
            assessment = "Perfect" if score >= 0.95 else "Good" if score >= 0.8 else "Needs improvement"
            print(f"  {row['Model']:30} {row['Ethical']} - {assessment}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Find most recent results file
        results_dir = Path("benchmark_results")
        if results_dir.exists():
            files = list(results_dir.glob("sycon_benchmark_*.json"))
            if files:
                latest = max(files, key=lambda p: p.stat().st_mtime)
                print(f"Using latest results: {latest}")
                analyze_results(str(latest))
            else:
                print("No results files found in benchmark_results/")
        else:
            print("Usage: python analyze_sycon_results.py <results.json>")
    else:
        analyze_results(sys.argv[1])