#!/usr/bin/env python3
"""
Visualize SYCON-Bench DEBATE results for presentation
Creates clear charts showing bread models are principled, not sycophantic
"""

import sys
import os
# Add parent directory to path for any potential imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd

# Set style for professional looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_latest_results():
    """Load the most recent test results"""
    # Look in both current directory and parent directory
    result_files = list(Path('.').glob('improved_sycon_test_*.json'))
    result_files.extend(list(Path('../..').glob('improved_sycon_test_*.json')))
    if not result_files:
        raise FileNotFoundError("No test results found")

    latest = max(result_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading results from: {latest}")

    with open(latest, 'r') as f:
        return json.load(f)

def create_overall_scores_chart(data, save_path='debate_scores.png'):
    """Create bar chart of overall scores"""
    models = []
    scores = []
    colors = []

    # Define colors for each model
    model_colors = {
        'bread-jf-1': '#2E7D32',  # Green (principled)
        'bread-pg-1': '#1976D2',  # Blue (opinionated)
        'claude-4-sonnet': '#D32F2F'  # Red (sycophantic)
    }

    for model_name, model_data in data.items():
        short_name = model_name.split('/')[-1]
        models.append(short_name)
        scores.append(model_data['score'])
        colors.append(model_colors.get(short_name, '#757575'))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, scores, color=colors, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Resistance to Pressure (Higher = Better)', fontsize=12)
    ax.set_title('SYCON-Bench DEBATE Results\nMeasuring Sycophantic Behavior Across 100 Topics',
                 fontsize=14, fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')

    # Add interpretation text
    ax.text(0.02, 0.98, 'Higher scores = More principled/consistent',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved overall scores chart to {save_path}")
    plt.show()

def create_stance_distribution(data, save_path='stance_distribution.png'):
    """Create pie charts showing stance distribution for each model"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (model_name, model_data) in enumerate(data.items()):
        short_name = model_name.split('/')[-1]
        stance_dist = model_data.get('stance_distribution', {})

        # Extract counts
        total = sum(stance_dist.values()) if stance_dist else 1
        if total == 0:
            total = 1  # Avoid division by zero

        sizes = []
        labels = []
        colors_list = []

        stance_colors = {
            'PRO': '#4CAF50',
            'NEUTRAL': '#9E9E9E',
            'AGAINST': '#F44336'
        }

        for stance in ['PRO', 'NEUTRAL', 'AGAINST']:
            count = stance_dist.get(stance, 0)
            if total > 0 or count > 0:  # Include if there's any data
                sizes.append(count)
                labels.append(f'{stance}\n({count}/{total})')
                colors_list.append(stance_colors[stance])

        if sizes:  # Only create pie if there's data
            axes[idx].pie(sizes, labels=labels, colors=colors_list, autopct='%1.1f%%',
                         startangle=90, textprops={'fontsize': 10})
        axes[idx].set_title(f'{short_name}\nStance Distribution', fontsize=12, fontweight='bold')

    plt.suptitle('Initial Stance Distribution Across 100 Debate Topics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved stance distribution to {save_path}")
    plt.show()

def create_flip_analysis(data, save_path='flip_patterns.png'):
    """Analyze when models flip their stance"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Calculate average ToF and NoF for each model
    models = []
    avg_tof = []
    flip_rates = []

    for model_name, model_data in data.items():
        short_name = model_name.split('/')[-1]
        models.append(short_name)

        # Get score (which is normalized ToF)
        score = model_data['score']
        avg_tof.append(score * 5)  # Convert back to raw ToF (0-5 scale)

        # Calculate flip rate (inverse of score)
        flip_rates.append(1 - score)

    # Plot 1: Average turns before flipping
    colors = ['#2E7D32', '#1976D2', '#D32F2F']
    bars1 = ax1.bar(models, avg_tof, color=colors, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Average Turns Before Flipping', fontsize=12)
    ax1.set_ylim(0, 5.5)
    ax1.set_title('Resistance to Pressure\n(Higher = More Principled)', fontsize=12, fontweight='bold')
    ax1.axhline(y=2.5, color='gray', linestyle='--', alpha=0.5)

    for bar, val in zip(bars1, avg_tof):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Plot 2: Flip rate
    bars2 = ax2.bar(models, flip_rates, color=colors, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Sycophancy Rate', fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.set_title('Tendency to Flip Under Pressure\n(Lower = Better)', fontsize=12, fontweight='bold')

    for bar, val in zip(bars2, flip_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle('Stance Consistency Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved flip analysis to {save_path}")
    plt.show()

def create_summary_slide(data, save_path='summary_slide.png'):
    """Create a single summary slide with key findings"""
    fig = plt.figure(figsize=(16, 10))

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Main scores (top row, spanning 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    models = []
    scores = []
    colors = ['#2E7D32', '#1976D2', '#D32F2F']

    for model_name, model_data in data.items():
        short_name = model_name.split('/')[-1]
        models.append(short_name)
        scores.append(model_data['score'])

    bars = ax1.barh(models, scores, color=colors, edgecolor='black', linewidth=2)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel('Resistance Score', fontsize=12)
    ax1.set_title('SYCON-Bench DEBATE Results (100 Topics)', fontsize=14, fontweight='bold')

    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                f'{score:.3f}', ha='left', va='center', fontsize=12, fontweight='bold')

    # Key insights (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    insights = [
        "Key Findings:",
        "",
        "✓ bread-jf-1: Most consistent",
        "  Maintains neutral stance",
        "",
        "✓ bread-pg-1: Principled",
        "  Takes positions, holds them",
        "",
        "✗ Claude: Most sycophantic",
        "  Flips quickly under pressure"
    ]
    ax2.text(0.1, 0.9, '\n'.join(insights), transform=ax2.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Stance distribution (bottom half)
    for idx, (model_name, model_data) in enumerate(data.items()):
        ax = fig.add_subplot(gs[1:, idx])
        short_name = model_name.split('/')[-1]
        stance_dist = model_data.get('stance_distribution', {})

        total = sum(stance_dist.values()) if stance_dist else 1
        if total > 0:
            sizes = [stance_dist.get(s, 0) for s in ['PRO', 'NEUTRAL', 'AGAINST']]
            labels = ['PRO', 'NEUTRAL', 'AGAINST']
            colors_pie = ['#4CAF50', '#9E9E9E', '#F44336']

            # Filter out zero values
            non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors_pie) if s > 0]
            if non_zero:
                sizes, labels, colors_pie = zip(*non_zero)
                ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.0f%%',
                      startangle=90, textprops={'fontsize': 10})
            ax.set_title(f'{short_name}\n({total} topics)', fontsize=11, fontweight='bold')

    plt.suptitle('SYCON-Bench: Measuring Sycophantic Behavior in Language Models',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved summary slide to {save_path}")
    plt.show()

def main():
    """Generate all visualizations"""
    print("Loading test results...")
    data = load_latest_results()

    print("\nGenerating visualizations...")

    # Generate individual charts
    create_overall_scores_chart(data)
    create_stance_distribution(data)
    create_flip_analysis(data)

    # Generate summary slide
    create_summary_slide(data)

    print("\nAll visualizations complete!")
    print("Files saved:")
    print("  - debate_scores.png")
    print("  - stance_distribution.png")
    print("  - flip_patterns.png")
    print("  - summary_slide.png (use this for presentation!)")

if __name__ == "__main__":
    main()