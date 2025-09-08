from pydantic import BaseModel
from typing import List, Dict, Optional


class RareWordAnalysis(BaseModel):
    """Analysis of rare word usage in a single response"""
    word: str
    frequency_rank: Optional[int] = None  # Position in frequency list (1-based, higher = rarer)
    log_frequency: float  # Log of the frequency rank (or penalty for unknown words)
    is_rare: bool  # Whether this word is considered rare based on threshold
    is_unknown: bool = False  # Whether this word is not in the frequency list at all


class RareWordUsageResult(BaseModel):
    """Result of evaluating rare word usage in a single response"""
    total_words: int
    rare_words_count: int
    rare_words_percentage: float
    average_log_frequency: float  # Average log frequency of all rare words
    rare_words_found: List[RareWordAnalysis]
    unique_rare_words: int  # Count of unique rare words (no duplicates)


class RareWordUsageBenchmarkResult(BaseModel):
    """Overall result from the Rare Word Usage benchmark"""
    overall_rare_word_score: float  # Average rare word usage score across all prompts
    total_prompts: int
    total_words_generated: int
    total_rare_words: int
    average_rare_word_percentage: float
    category_breakdown: Dict[str, float]  # Score breakdown by prompt category
    detailed_results: List[Dict]  # Per-prompt detailed results


class GenerationPrompt(BaseModel):
    """A single generation prompt for the benchmark"""
    prompt_id: str
    category: str  # e.g., "essay", "poem", "math", "science", "coding", etc.
    difficulty: str  # "easy", "medium", "hard"
    prompt_text: str
    expected_response_length: str  # "short", "medium", "long"
    description: str  # Brief description of what the prompt is testing
