from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmarkResult


class TurnResult(BaseModel):
    """Schema for individual turn comparison results."""
    
    turn_index: int = Field(ge=1, description="Turn number when evaluation occurred")
    user_message: str = Field(description="The user message for this turn")
    baked_response: str = Field(description="Response from baked-in model")
    system_response: str = Field(description="Response from system-prompted model")
    winner: str = Field(description="Winner: 'baked_model' or 'system_model'")
    reasoning: str = Field(description="Arena G-Eval's explanation for the choice")
    evaluation_cost: float = Field(default=0.0, ge=0.0, description="Cost for this turn evaluation")
    


class PersonaDriftV3Result(DeepEvalBaseBenchmarkResult):
    """
    Result object for Persona Drift v3 benchmark.
    
    Extends DeepEvalBaseBenchmarkResult to include win rates and turn-level results
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
    total_turns_evaluated: int = Field(description="Number of turn comparisons performed")
    total_turns: int = Field(description="Number of conversation turns")
    total_cost: float = Field(default=0.0, description="Total cost of benchmark run")
    total_time_s: float = Field(default=0.0, description="Total time in seconds")
    
    # Detailed results
    turn_results: List[TurnResult] = Field(
        default_factory=list,
        description="Individual turn comparison results with reasoning"
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
        total_turns_evaluated: int,
        total_turns: int,
        total_cost: float = 0.0,
        total_time_s: float = 0.0,
        turn_results: Optional[List[TurnResult]] = None,
        run_metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize PersonaDriftV3Result.
        
        Args:
            baked_model_score: Win rate for baked-in model (0-1)
            system_model_score: Win rate for system-prompted model (0-1)
            overall_winner: "baked_model" or "system_model"
            total_turns_evaluated: Number of turn comparisons performed
            total_turns: Number of conversation turns
            total_cost: Total cost of benchmark run
            total_time_s: Total time in seconds
            turn_results: Individual turn comparison results
            run_metadata: Configuration information
            **kwargs: Additional arguments for base class
        """
        # Calculate overall_accuracy as baked model's win rate
        # 1.0 = baked model won all turns, 0.0 = system model won all turns
        overall_accuracy = baked_model_score
        
        super().__init__(
            overall_accuracy=overall_accuracy,
            baked_model_score=baked_model_score,
            system_model_score=system_model_score,
            overall_winner=overall_winner,
            total_turns_evaluated=total_turns_evaluated,
            total_turns=total_turns,
            total_cost=total_cost,
            total_time_s=total_time_s,
            turn_results=turn_results or [],
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
            "total_turns_evaluated": self.total_turns_evaluated,
            "total_turns": self.total_turns,
            "total_cost": self.total_cost,
            "total_time_s": self.total_time_s,
            "turn_results": [turn.dict() for turn in self.turn_results],
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
        Count of turns won by baked-in model.
        
        Returns:
            Number of turns won by baked model
        """
        return sum(1 for turn in self.turn_results if turn.winner == "baked_model")
    
    @property
    def system_wins(self) -> int:
        """
        Count of turns won by system-prompted model.
        
        Returns:
            Number of turns won by system model
        """
        return sum(1 for turn in self.turn_results if turn.winner == "system_model")
    
    @property
    def win_margin(self) -> float:
        """
        Calculate margin of victory (difference in win rates).

        Returns:
            Absolute difference between win rates
        """
        return abs(self.baked_model_score - self.system_model_score)

    @property
    def comparison_mode(self) -> str:
        """
        Get the comparison mode from metadata.

        Returns:
            Comparison mode: 'baked_vs_system' or 'baked_vs_baked'
        """
        return self.run_metadata.get("comparison_mode", "baked_vs_system")

    @property
    def model_1_score(self) -> float:
        """
        Generic accessor for first model's score (maps to baked_model_score).

        Returns:
            Win rate for first model
        """
        return self.baked_model_score

    @property
    def model_2_score(self) -> float:
        """
        Generic accessor for second model's score (maps to system_model_score).

        Returns:
            Win rate for second model
        """
        return self.system_model_score

    @property
    def model_1_wins(self) -> int:
        """
        Count of turns won by first model.

        Returns:
            Number of turns won by first model
        """
        if self.comparison_mode == "baked_vs_baked":
            # In baked vs baked mode, count model_1 wins
            return sum(1 for turn in self.turn_results if turn.winner == "model_1")
        else:
            # In baked vs system mode, use existing baked_wins property
            return self.baked_wins

    @property
    def model_2_wins(self) -> int:
        """
        Count of turns won by second model.

        Returns:
            Number of turns won by second model
        """
        if self.comparison_mode == "baked_vs_baked":
            # In baked vs baked mode, count model_2 wins
            return sum(1 for turn in self.turn_results if turn.winner == "model_2")
        else:
            # In baked vs system mode, use existing system_wins property
            return self.system_wins

