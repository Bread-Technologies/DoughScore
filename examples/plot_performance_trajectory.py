#!/usr/bin/env python3
import argparse
import os
import re
import matplotlib.pyplot as plt
from typing import List, Tuple

def parse_performance_trajectory(log_path: str) -> List[Tuple[int, int, int]]:
    """
    Parse a simulator log file and extract turn-by-turn performance scores.
    
    Returns a list of tuples: (turn_number, baked_wins, system_wins)
    """
    results = []
    
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    # Find all CURRENT SCORE lines
    score_pattern = r'CURRENT SCORE - Baked: (\d+) \| System: (\d+)'
    matches = re.findall(score_pattern, content)
    
    for i, (baked_wins, system_wins) in enumerate(matches, 1):
        results.append((i, int(baked_wins), int(system_wins)))
    
    return results

def plot_performance_trajectory(
    trajectory: List[Tuple[int, int, int]],
    title: str = "Performance Trajectory - Absolute Wins",
    save_path: str = None,
    show: bool = True
):
    """
    Create a line chart showing the performance trajectory of both models.
    """
    turns = [t[0] for t in trajectory]
    baked_wins = [t[1] for t in trajectory]
    system_wins = [t[2] for t in trajectory]
    
    # Set up the plot
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), facecolor='white')
    
    # Plot absolute wins
    ax.plot(turns, baked_wins, label='Baked Model', color='#2563EB', linewidth=2, marker='o', markersize=3)
    ax.plot(turns, system_wins, label='System Model', color='#DC2626', linewidth=2, marker='s', markersize=3)
    ax.set_title(title, fontsize=16, fontweight='bold', color='#1F2937', pad=20)
    ax.set_xlabel("Turn Number", fontsize=12, color='#374151')
    ax.set_ylabel("Cumulative Wins", fontsize=12, color='#374151')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')
    
    # Add performance dip annotation
    if len(trajectory) > 10:
        # Find the point where system model was ahead
        system_ahead_turns = [i for i, (turn, b, s) in enumerate(trajectory) if s > b]
        if system_ahead_turns:
            dip_end = max(system_ahead_turns) + 1
            ax.axvspan(1, dip_end, alpha=0.2, color='red', label='System Model Dominance')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved chart to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Plot performance trajectory from Persona Drift v3 logs.")
    parser.add_argument("log_path", help="Path to simulator log file")
    parser.add_argument("--title", type=str, default=None, help="Chart title")
    parser.add_argument("--save", type=str, default=None, help="Path to save PNG")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive window")
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.log_path):
        raise FileNotFoundError(f"Log file not found: {args.log_path}")
    
    # Parse performance trajectory
    trajectory = parse_performance_trajectory(args.log_path)
    if not trajectory:
        raise RuntimeError("No performance data found in log.")
    
    print(f"Parsed {len(trajectory)} turns")
    print(f"Final scores - Baked: {trajectory[-1][1]}, System: {trajectory[-1][2]}")
    
    # Generate title
    base = os.path.splitext(os.path.basename(args.log_path))[0]
    title = args.title or "Performance Trajectory - Absolute Wins"
    
    # Generate save path
    save_path = args.save
    if save_path is None:
        out_dir = os.path.dirname(args.log_path) or "."
        save_path = os.path.join(out_dir, f"{base}_performance_trajectory.png")
    
    # Create the plot
    plot_performance_trajectory(trajectory, title, save_path=save_path, show=not args.no_show)

if __name__ == "__main__":
    main()
