from typing import Dict, List, Any
from pydantic import Field
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmarkResult


class PersonalityQuizResult(DeepEvalBaseBenchmarkResult):
    """Result class for Personality Quiz benchmark."""
    
    trait_scores: Dict[str, float] = Field(description="Dictionary mapping trait names to their scores")
    responses: List[Dict[str, Any]] = Field(description="List of response dictionaries containing test cases and model responses")
    cost: float = Field(default=0.0, description="Total cost of the evaluation")
    
    def __init__(
        self,
        trait_scores: Dict[str, float],
        responses: List[Dict[str, Any]],
        **kwargs
    ):
        """Initialize Personality Quiz result.
        
        Args:
            trait_scores: Dictionary mapping trait names to their scores
            responses: List of response dictionaries containing test cases and model responses
            **kwargs: Additional arguments passed to parent BenchmarkResult
        """
        super().__init__(
            trait_scores=trait_scores,
            responses=responses,
            **kwargs
        )
    
    def get_trait_scores(self) -> Dict[str, float]:
        """Get the trait scores.
        
        Returns:
            Dictionary mapping trait names to scores
        """
        return self.trait_scores
    
    def get_responses(self) -> List[Dict[str, Any]]:
        """Get the model responses.
        
        Returns:
            List of response dictionaries
        """
        return self.responses
    
    def get_trait_score(self, trait: str) -> float:
        """Get the score for a specific trait.
        
        Args:
            trait: The trait name (e.g., 'neuroticism', 'extraversion')
            
        Returns:
            The score for the specified trait
        """
        return self.trait_scores.get(trait.lower(), 0.0)
    
    def __str__(self) -> str:
        """String representation of the result."""
        trait_scores_str = ", ".join([f"{trait}: {score:.2f}" for trait, score in self.trait_scores.items()])
        return f"PersonalityQuizResult(trait_scores={{{trait_scores_str}}}, overall_accuracy={self.overall_accuracy:.4f})"
