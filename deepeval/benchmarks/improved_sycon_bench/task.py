from enum import Enum


class SyconTask(Enum):
    """Tasks available in SYCON-Bench for sycophancy evaluation"""
    DEBATE = "debate"
    ETHICAL = "ethical"
    FALSE_PRESUPPOSITIONS = "false_presuppositions"


class SyconPromptStrategy(Enum):
    """Prompt strategies for SYCON-Bench"""
    INDIVIDUAL = "individual_thinker"
    NEEL = "neel_third_person"  # Third-person reduces sycophancy by 63.8%
    NON_SYCOPHANTIC = "non_sycophantic"
    NEEL_NON_SYCOPHANTIC = "neel_non_sycophantic"