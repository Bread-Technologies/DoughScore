from typing import Dict, Any, Optional
from pydantic import Field
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmarkResult


class PersonaDriftV2Result(DeepEvalBaseBenchmarkResult):
    """
    Result object for Persona Drift v2 benchmark.
    
    Extends DeepEvalBaseBenchmarkResult to include N_drift tracking and
    other persona-specific metrics following existing benchmark patterns.
    """
    
    # Core drift metric
    n_drift: int = Field(
        description="First turn where persona is lost (score < 0.5). If all turns pass, equals max_turns + 1."
    )
    
    # Basic statistics
    total_turns: int = Field(description="Number of conversation turns")
    total_cost: float = Field(default=0.0, description="Total cost of benchmark run")
    total_time_s: float = Field(default=0.0, description="Total time in seconds")
    average_rating: float = Field(default=0.0, description="Average rating across all probed turns")
    
    # Run metadata
    run_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration and seed information for reproducibility"
    )
    
    def __init__(
        self,
        overall_accuracy: float,
        n_drift: int,
        total_turns: int,
        total_cost: float = 0.0,
        total_time_s: float = 0.0,
        average_rating: float = 0.0,
        run_metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize PersonaDriftV2Result.
        
        Args:
            overall_accuracy: Overall accuracy score (inherited from base class)
            n_drift: First turn where persona is lost
            total_turns: Number of conversation turns
            total_cost: Total cost of benchmark run
            total_time_s: Total time in seconds
            average_rating: Average rating across all probed turns
            run_metadata: Configuration and seed information
            **kwargs: Additional arguments for base class
        """
        super().__init__(
            overall_accuracy=overall_accuracy,
            n_drift=n_drift,
            total_turns=total_turns,
            total_cost=total_cost,
            total_time_s=total_time_s,
            average_rating=average_rating,
            run_metadata=run_metadata or {},
            **kwargs
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary for analysis.
        
        Returns:
            Dictionary representation of the result
        """
        return {
            "overall_accuracy": self.overall_accuracy,
            "n_drift": self.n_drift,
            "total_turns": self.total_turns,
            "total_cost": self.total_cost,
            "total_time_s": self.total_time_s,
            "average_rating": self.average_rating,
            "run_metadata": self.run_metadata,
        }
    
    def to_dataframe(self):
        """
        Convert result to pandas DataFrame for analysis.
        
        Returns:
            DataFrame representation of the result
        """
        import pandas as pd
        
        return pd.DataFrame([self.to_dict()])
    
    @property
    def drift_detected(self) -> bool:
        """
        Check if persona drift was detected.
        
        Returns:
            True if drift was detected (n_drift <= total_turns), False otherwise
        """
        return self.n_drift <= self.total_turns
    
    @property
    def survival_rate(self) -> float:
        """
        Calculate survival rate (fraction of turns before drift).
        
        Returns:
            Survival rate as a float between 0 and 1
        """
        if self.n_drift > self.total_turns:
            return 1.0  # No drift detected
        return (self.n_drift - 1) / self.total_turns
