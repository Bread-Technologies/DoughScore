"""Tests for PersonaDriftV2Result class."""

import pytest
from deepeval.benchmarks.persona_drift_v2.result import PersonaDriftV2Result


class TestPersonaDriftV2Result:
    """Test cases for PersonaDriftV2Result class."""
    
    def test_result_creation(self):
        """Test creating a PersonaDriftV2Result instance."""
        result = PersonaDriftV2Result(
            overall_accuracy=0.8,
            n_drift=5,
            total_turns=8,
            total_cost=0.1,
            total_time_s=30.0,
            run_metadata={'seed': 42, 'model': 'gpt-4'}
        )
        
        assert result.overall_accuracy == 0.8
        assert result.n_drift == 5
        assert result.total_turns == 8
        assert result.total_cost == 0.1
        assert result.total_time_s == 30.0
        assert result.run_metadata == {'seed': 42, 'model': 'gpt-4'}
    
    def test_result_defaults(self):
        """Test PersonaDriftV2Result with default values."""
        result = PersonaDriftV2Result(
            overall_accuracy=0.5,
            n_drift=3,
            total_turns=5
        )
        
        assert result.overall_accuracy == 0.5
        assert result.n_drift == 3
        assert result.total_turns == 5
        assert result.total_cost == 0.0  # Default
        assert result.total_time_s == 0.0  # Default
        assert result.run_metadata == {}  # Default
    
    def test_drift_detected_property(self):
        """Test the drift_detected property."""
        # Drift detected case
        result_with_drift = PersonaDriftV2Result(
            overall_accuracy=0.3,
            n_drift=3,
            total_turns=8
        )
        assert result_with_drift.drift_detected is True
        
        # No drift case
        result_no_drift = PersonaDriftV2Result(
            overall_accuracy=0.9,
            n_drift=9,  # n_drift > total_turns
            total_turns=8
        )
        assert result_no_drift.drift_detected is False
    
    def test_survival_rate_property(self):
        """Test the survival_rate property."""
        # Drift detected case
        result_with_drift = PersonaDriftV2Result(
            overall_accuracy=0.3,
            n_drift=3,
            total_turns=8
        )
        expected_survival = (3 - 1) / 8  # (n_drift - 1) / total_turns
        assert result_with_drift.survival_rate == expected_survival
        
        # No drift case
        result_no_drift = PersonaDriftV2Result(
            overall_accuracy=0.9,
            n_drift=9,  # n_drift > total_turns
            total_turns=8
        )
        assert result_no_drift.survival_rate == 1.0
    
    def test_to_dict_method(self):
        """Test the to_dict method."""
        result = PersonaDriftV2Result(
            overall_accuracy=0.7,
            n_drift=4,
            total_turns=6,
            total_cost=0.05,
            total_time_s=25.0,
            run_metadata={'seed': 123}
        )
        
        result_dict = result.to_dict()
        
        expected_dict = {
            "overall_accuracy": 0.7,
            "n_drift": 4,
            "total_turns": 6,
            "total_cost": 0.05,
            "total_time_s": 25.0,
            "run_metadata": {'seed': 123}
        }
        
        assert result_dict == expected_dict
    
    def test_to_dataframe_method(self):
        """Test the to_dataframe method."""
        result = PersonaDriftV2Result(
            overall_accuracy=0.6,
            n_drift=2,
            total_turns=4,
            total_cost=0.02,
            total_time_s=15.0,
            run_metadata={'seed': 456}
        )
        
        df = result.to_dataframe()
        
        # Check that it's a pandas DataFrame
        assert hasattr(df, 'shape')
        assert df.shape[0] == 1  # One row
        assert df.shape[1] == 6  # Six columns
        
        # Check column names
        expected_columns = [
            "overall_accuracy", "n_drift", "total_turns", 
            "total_cost", "total_time_s", "run_metadata"
        ]
        assert list(df.columns) == expected_columns
        
        # Check values
        assert df.iloc[0]['overall_accuracy'] == 0.6
        assert df.iloc[0]['n_drift'] == 2
        assert df.iloc[0]['total_turns'] == 4
        assert df.iloc[0]['total_cost'] == 0.02
        assert df.iloc[0]['total_time_s'] == 15.0
        assert df.iloc[0]['run_metadata'] == {'seed': 456}
    
    def test_pydantic_validation(self):
        """Test Pydantic validation for PersonaDriftV2Result."""
        # Test valid data
        result = PersonaDriftV2Result(
            overall_accuracy=0.5,
            n_drift=1,
            total_turns=10,
            total_cost=0.0,
            total_time_s=0.0,
            run_metadata={}
        )
        assert result is not None
        
        # Test invalid data types
        with pytest.raises(Exception):  # Pydantic validation error
            PersonaDriftV2Result(
                overall_accuracy="invalid",  # Should be float
                n_drift=1,
                total_turns=10
            )
    
    def test_inheritance(self):
        """Test that PersonaDriftV2Result inherits from DeepEvalBaseBenchmarkResult."""
        from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmarkResult
        
        result = PersonaDriftV2Result(
            overall_accuracy=0.5,
            n_drift=1,
            total_turns=10
        )
        
        assert isinstance(result, DeepEvalBaseBenchmarkResult)
