#!/usr/bin/env python3
import argparse
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available. Install with 'pip install tiktoken' for accurate token counting.")


TurnWinner = Tuple[int, str, int, int]  # (turn_number, winner, cumulative_tokens_model1, cumulative_tokens_model2)


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Accurate token counting using tiktoken for OpenAI models.
    Falls back to rough estimation for non-OpenAI models or if tiktoken unavailable.
    """
    if not text:
        return 0
    
    if TIKTOKEN_AVAILABLE:
        try:
            # Check if it's an OpenAI model that tiktoken supports
            if any(openai_model in model.lower() for openai_model in ['gpt-3.5', 'gpt-4', 'gpt-3', 'text-davinci', 'text-curie', 'text-babbage', 'text-ada']):
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
            else:
                # For non-OpenAI models (like Claude), use rough estimation
                print(f"Warning: Using rough token estimation for {model} (tiktoken only supports OpenAI models)")
        except Exception as e:
            # Fallback to rough estimation
            print(f"Warning: tiktoken not available because of {e}")
    
    # Fallback: rough estimation (~4 characters per token)
    cleaned = re.sub(r'\s+', ' ', text.strip())
    return max(1, len(cleaned) // 4)


def count_tokens_dual(text: str, model1: str, model2: str) -> Tuple[int, int]:
    """
    Count tokens using both model tokenizers for accurate comparison.
    Returns (tokens_model1, tokens_model2).
    """
    if not text:
        return (0, 0)
    
    tokens1 = count_tokens(text, model1)
    tokens2 = count_tokens(text, model2)
    return (tokens1, tokens2)


def parse_log_for_turn_winners(log_path: str, model1: str = "gpt-4", model2: str = "claude-3-sonnet") -> List[TurnWinner]:
    """
    Parse a simulator log file and extract (turn_number, winner, cumulative_tokens) tuples.

    This parses two possible log formats:
    1. Per-turn winners: "Winner: Baked Model" for each turn
    2. Overall winner: Single "Winner: Baked Model" at the end with score like "Score: 0.800 vs 0.200"
    
    Turn markers: "Turn 1/100" or "TURN 1"
    Message content: USER:, BAKED MODEL:, SYSTEM MODEL: sections

    Returns a list sorted by turn number (ascending).
    """
    # Read the entire file first
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    # Split into turns using TURN markers
    turn_sections = re.split(r'^TURN\s+(\d+)$', content, flags=re.MULTILINE)
    
    # Process turn sections
    results: List[TurnWinner] = []
    cumulative_tokens1: int = 0
    cumulative_tokens2: int = 0
    
    # Track system prompt tokens separately
    system_prompt_tokens1: int = 0
    system_prompt_tokens2: int = 0
    conversation_tokens1: int = 0
    conversation_tokens2: int = 0
    
    # Find overall winner and score - get the LAST occurrence
    overall_winner_matches = re.findall(r'Winner:\s*(Baked|System)\s*Model', content, re.IGNORECASE)
    overall_winner = overall_winner_matches[-1].lower() if overall_winner_matches else None
    
    score_match = re.search(r'Score:\s*([\d.]+)\s*vs\s*([\d.]+)', content, re.IGNORECASE)
    overall_score = (float(score_match.group(1)), float(score_match.group(2))) if score_match else None
    
    print(f"DEBUG: Found overall winner: {overall_winner}, score: {overall_score}")
    
    # Extract and count system prompt tokens
    system_prompt_match = re.search(r'PERSONA SYSTEM PROMPT:(.*?)(?=TURN|\Z)', content, re.DOTALL)
    if system_prompt_match:
        system_prompt_text = system_prompt_match.group(1).strip()
        system_prompt_tokens1, system_prompt_tokens2 = count_tokens_dual(system_prompt_text, model1, model2)
        print(f"DEBUG: System prompt tokens - Model1: {system_prompt_tokens1}, Model2: {system_prompt_tokens2}")
    
    # Process each turn section
    for i in range(1, len(turn_sections), 2):  # Skip the first empty section
        if i + 1 < len(turn_sections):
            turn_num = int(turn_sections[i])
            turn_content = turn_sections[i + 1]
            
            # Extract conversation content from this turn
            turn_text = ""
            lines = turn_content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("USER:") or line.startswith("BAKED MODEL:") or line.startswith("SYSTEM MODEL:"):
                    # Extract content after the colon
                    if ":" in line:
                        content_part = line.split(":", 1)[1].strip()
                        if content_part:
                            turn_text += content_part + " "
                elif line and not line.startswith("=") and not line.startswith("-") and not line.startswith("ARENA") and not line.startswith("Turn ") and not line.startswith("TURN "):
                    # Continuation lines
                    turn_text += line + " "
            
            # Find the winner for this specific turn
            turn_winner_match = re.search(r'WINNER:\s*(BAKED|SYSTEM)\s*MODEL', turn_content, re.IGNORECASE)
            turn_winner = turn_winner_match.group(1).lower() if turn_winner_match else (overall_winner if overall_winner else "baked")
            
            # Count tokens for this turn (conversation only, not system prompt)
            if turn_text.strip():
                tokens1, tokens2 = count_tokens_dual(turn_text.strip(), model1, model2)
                conversation_tokens1 += tokens1
                conversation_tokens2 += tokens2
                
                # Total tokens = system prompt + conversation
                total_tokens1 = system_prompt_tokens1 + conversation_tokens1
                total_tokens2 = system_prompt_tokens2 + conversation_tokens2
                
                if turn_num <= 3:  # Debug first few turns
                    print(f"DEBUG: Turn {turn_num} - Winner: {turn_winner}, Conversation tokens: {tokens1}/{tokens2}, Total tokens: {total_tokens1}/{total_tokens2}")
                    print(f"DEBUG: Turn {turn_num} content preview: {turn_text[:200]}...")
                
                # Use conversation tokens for bucketing, but store total tokens for reference
                results.append((turn_num, turn_winner, conversation_tokens1, conversation_tokens2))
    
    # Sort by turn number
    results.sort(key=lambda x: x[0])
    
    # Debug logging
    print(f"DEBUG: Parsed {len(results)} turns")
    print(f"DEBUG: System prompt tokens - Model1: {system_prompt_tokens1}, Model2: {system_prompt_tokens2}")
    if results:
        print(f"DEBUG: First 3 turns: {results[:3]}")
        print(f"DEBUG: Last 3 turns: {results[-3:]}")
        print(f"DEBUG: Conversation token ranges - Model1: {min(t[2] for t in results)}-{max(t[2] for t in results)}, Model2: {min(t[3] for t in results)}-{max(t[3] for t in results)}")
        print(f"DEBUG: Total token ranges - Model1: {system_prompt_tokens1 + min(t[2] for t in results)}-{system_prompt_tokens1 + max(t[2] for t in results)}, Model2: {system_prompt_tokens2 + min(t[3] for t in results)}-{system_prompt_tokens2 + max(t[3] for t in results)}")
    
    return results


def compute_token_based_win_rates(
    turn_winners: List[TurnWinner],
    token_buckets: List[int],
    ties_as_half: bool,
    use_model1_tokens: bool = True,
) -> List[Tuple[str, float, float]]:
    """
    Aggregate wins into token-based buckets and compute per-bucket win rates.

    - token_buckets: List of token thresholds, e.g., [200, 500, 1000, 2000]
    - ties_as_half=True counts each tie as 0.5 win for both sides.
    - ties_as_half=False excludes ties from the denominator.

    Returns a list of tuples: (stage_label, baked_rate, system_rate)
    """
    buckets: Dict[str, Dict[str, float]] = defaultdict(lambda: {"baked": 0.0, "system": 0.0, "den": 0.0})

    for turn_num, winner, cumulative_tokens1, cumulative_tokens2 in turn_winners:
        # Use the appropriate token count based on which model we're analyzing
        cumulative_tokens = cumulative_tokens1 if use_model1_tokens else cumulative_tokens2
        
        # Determine which token bucket this turn belongs to
        first_threshold = token_buckets[0]
        bucket_label = f"0-{first_threshold}"
        for i, threshold in enumerate(token_buckets):
            if cumulative_tokens <= threshold:
                if i == 0:
                    bucket_label = f"0-{threshold}"
                else:
                    bucket_label = f"{token_buckets[i-1]}-{threshold}"
                break
        else:
            # Above all thresholds
            bucket_label = f"{token_buckets[-1]}+"

        if winner == "baked":
            buckets[bucket_label]["baked"] += 1.0
            buckets[bucket_label]["den"] += 1.0
        elif winner == "system":
            buckets[bucket_label]["system"] += 1.0
            buckets[bucket_label]["den"] += 1.0
        elif winner == "tie":
            if ties_as_half:
                buckets[bucket_label]["baked"] += 0.5
                buckets[bucket_label]["system"] += 0.5
                buckets[bucket_label]["den"] += 1.0
            else:
                # exclude from denominator
                pass

    stage_stats: List[Tuple[str, float, float]] = []
    # Sort buckets by token count (dynamic, based on provided thresholds)
    bucket_order = [f"0-{token_buckets[0]}"] + [f"{token_buckets[i-1]}-{token_buckets[i]}" for i in range(1, len(token_buckets))] + [f"{token_buckets[-1]}+"]
    
    # Debug logging for bucketing
    print(f"DEBUG: Token buckets: {token_buckets}")
    print(f"DEBUG: Bucket order: {bucket_order}")
    print(f"DEBUG: Available buckets: {list(buckets.keys())}")
    
    for bucket_label in bucket_order:
        if bucket_label in buckets:
            den = buckets[bucket_label]["den"] if buckets[bucket_label]["den"] > 0 else 1.0
            baked_rate = buckets[bucket_label]["baked"] / den
            system_rate = buckets[bucket_label]["system"] / den
            stage_stats.append((f"{bucket_label} tokens", baked_rate, system_rate))
            print(f"DEBUG: Bucket {bucket_label}: baked={buckets[bucket_label]['baked']}, system={buckets[bucket_label]['system']}, den={den}")

    return stage_stats


def plot_stage_wins(
    stage_stats: List[Tuple[str, float, float]],
    title: str,
    save_path: str = None,
    show: bool = True,
):
    labels = [s[0] for s in stage_stats]
    baked_rates = [s[1] for s in stage_stats]
    system_rates = [s[2] for s in stage_stats]

    x = range(len(labels))
    width = 0.38

    # Set light theme
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    ax.set_facecolor('white')
    
    # Use contrasting colors for light mode
    ax.bar([i - width / 2 for i in x], baked_rates, width, label="Baked win rate", 
           color='#2563EB', edgecolor='#1D4ED8', linewidth=1, alpha=0.8)
    ax.bar([i + width / 2 for i in x], system_rates, width, label="System win rate", 
           color='#DC2626', edgecolor='#B91C1C', linewidth=1, alpha=0.8)

    # Styling for light mode
    ax.set_title(title, fontsize=18, fontweight='bold', color='#1F2937', pad=20)
    ax.set_ylabel("Win Rate", fontsize=14, color='#374151', fontweight='normal')
    ax.set_ylim(0, 1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=12, color='#374151', rotation=0)
    ax.legend().set_visible(False)
    
    # Subtle grid lines
    ax.grid(axis="y", linestyle="-", alpha=0.2, color='#D1D5DB')
    ax.set_axisbelow(True)
    
    # Clean spines
    for spine in ax.spines.values():
        spine.set_color('#E5E7EB')
        spine.set_linewidth(1)
    
    # Y-axis styling
    ax.tick_params(axis='y', colors='#6B7280', labelsize=12)
    ax.tick_params(axis='x', colors='#6B7280', labelsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved chart to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot token-based win rates from Persona Drift v3 logs.")
    parser.add_argument("log_path", help="Path to simulator log file (e.g., logging/*.log)")
    parser.add_argument("--token-buckets", nargs="+", type=int, default=[200, 500, 1000, 2000], 
                       help="Token count thresholds for bucketing (default: 200 500 1000 2000)")
    parser.add_argument("--baked-model", type=str, default="gpt-4", 
                       help="Baked model name for token counting (default: gpt-4)")
    parser.add_argument("--system-model", type=str, default="claude-3-sonnet", 
                       help="System model name for token counting (default: claude-3-sonnet)")
    parser.add_argument("--use-baked-tokens", action="store_true", 
                       help="Use baked model tokenization for bucketing (default: use baked model)")
    parser.add_argument("--ties-as-half", action="store_true", help="Count ties as 0.5 win for each side (default: exclude ties)")
    parser.add_argument("--title", type=str, default=None, help="Chart title (default: derived from filename)")
    parser.add_argument("--save", type=str, default=None, help="Path to save PNG (default: same dir as log with suffix)")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive window; save only")

    args = parser.parse_args()

    log_path = args.log_path
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    # Parse with dual token counting
    turn_winners = parse_log_for_turn_winners(log_path, args.baked_model, args.system_model)
    if not turn_winners:
        raise RuntimeError("No (turn, winner) pairs found in log. Check log format or regex patterns.")

    # Use baked model tokens by default (since that's what we're analyzing)
    use_baked_tokens = args.use_baked_tokens
    stage_stats = compute_token_based_win_rates(turn_winners, args.token_buckets, args.ties_as_half, use_baked_tokens)

    # Default title and save path
    base = os.path.splitext(os.path.basename(log_path))[0]
    title = args.title or "Win Rates by Conversation Complexity"
    save_path = args.save
    if save_path is None:
        out_dir = os.path.dirname(log_path) or "."
        save_path = os.path.join(out_dir, f"{base}_token_wins.png")

    plot_stage_wins(stage_stats, title, save_path=save_path, show=not args.no_show)


if __name__ == "__main__":
    main()


