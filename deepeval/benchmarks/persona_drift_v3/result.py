from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmarkResult


class ProbeResult(BaseModel):
    """Schema for individual probe comparison results."""
    
    turn_index: int = Field(ge=1, description="Turn number when probe occurred")
    probe_question: str = Field(description="The probe question asked")
    baked_response: str = Field(description="Response from baked-in model")
    system_response: str = Field(description="Response from system-prompted model")
    winner: str = Field(description="Winner: 'baked_model' or 'system_model'")
    reasoning: str = Field(description="Arena G-Eval's explanation for the choice")
    evaluation_cost: float = Field(default=0.0, ge=0.0, description="Cost for this probe evaluation")
    


class PersonaDriftV3Result(DeepEvalBaseBenchmarkResult):
    """
    Result object for Persona Drift v3 benchmark.
    
    Extends DeepEvalBaseBenchmarkResult to include win rates and probe-level results
    for arena-style comparison between baked-in vs system-prompted models.
    """
    
    # Core arena metrics
    baked_model_score: float = Field(
        description="Win rate for baked-in model (0-1 scale)"
    )
    system_model_score: float = Field(
        description="Win rate for system-prompted model (0-1 scale)"
    )
    overall_winner: str = Field(
        description="Overall winner: 'baked_model' or 'system_model'"
    )
    
    # Basic statistics
    total_probes: int = Field(description="Number of probe comparisons performed")
    total_turns: int = Field(description="Number of conversation turns")
    total_cost: float = Field(default=0.0, description="Total cost of benchmark run")
    total_time_s: float = Field(default=0.0, description="Total time in seconds")
    
    # Detailed results
    probe_results: List[ProbeResult] = Field(
        default_factory=list,
        description="Individual probe comparison results with reasoning"
    )
    
    # Run metadata
    run_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration information for reproducibility"
    )
    
    def __init__(
        self,
        baked_model_score: float,
        system_model_score: float,
        overall_winner: str,
        total_probes: int,
        total_turns: int,
        total_cost: float = 0.0,
        total_time_s: float = 0.0,
        probe_results: Optional[List[ProbeResult]] = None,
        run_metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize PersonaDriftV3Result.
        
        Args:
            baked_model_score: Win rate for baked-in model (0-1)
            system_model_score: Win rate for system-prompted model (0-1)
            overall_winner: "baked_model" or "system_model"
            total_probes: Number of probe comparisons performed
            total_turns: Number of conversation turns
            total_cost: Total cost of benchmark run
            total_time_s: Total time in seconds
            probe_results: Individual probe comparison results
            run_metadata: Configuration information
            **kwargs: Additional arguments for base class
        """
        # Calculate overall_accuracy as baked model's win rate
        # 1.0 = baked model won all probes, 0.0 = system model won all probes
        overall_accuracy = baked_model_score
        
        super().__init__(
            overall_accuracy=overall_accuracy,
            baked_model_score=baked_model_score,
            system_model_score=system_model_score,
            overall_winner=overall_winner,
            total_probes=total_probes,
            total_turns=total_turns,
            total_cost=total_cost,
            total_time_s=total_time_s,
            probe_results=probe_results or [],
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
            "baked_model_score": self.baked_model_score,
            "system_model_score": self.system_model_score,
            "overall_winner": self.overall_winner,
            "total_probes": self.total_probes,
            "total_turns": self.total_turns,
            "total_cost": self.total_cost,
            "total_time_s": self.total_time_s,
            "probe_results": [probe.dict() for probe in self.probe_results],
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
    def baked_wins(self) -> int:
        """
        Count of probes won by baked-in model.
        
        Returns:
            Number of probes won by baked model
        """
        return sum(1 for probe in self.probe_results if probe.winner == "baked_model")
    
    @property
    def system_wins(self) -> int:
        """
        Count of probes won by system-prompted model.
        
        Returns:
            Number of probes won by system model
        """
        return sum(1 for probe in self.probe_results if probe.winner == "system_model")
    
    @property
    def win_margin(self) -> float:
        """
        Calculate margin of victory (difference in win rates).
        
        Returns:
            Absolute difference between win rates
        """
        return abs(self.baked_model_score - self.system_model_score)
    
