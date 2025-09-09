"""Tests for PersonaDriftV2 benchmark class."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from deepeval.benchmarks.persona_drift_v2 import PersonaDriftV2, PersonaDriftV2Result
from deepeval.test_case import LLMTestCase


class TestPersonaDriftV2:
    """Test cases for PersonaDriftV2 benchmark class."""
    
    def setup_method(self):
        """Set up mock models and benchmark for testing."""
        self.mock_agent_model = Mock()
        self.mock_user_model = Mock()
        
        # Mock model responses
        self.mock_agent_model.chat_generate.return_value = ("Agent response", 0.01)
        self.mock_user_model.chat_generate.return_value = ("User response", 0.01)
        self.mock_user_model.get_model_name.return_value = "mock-user-model"
        
        self.persona_system_prompt = "You are Paul Graham, co-founder of Y Combinator."
        
        self.benchmark = PersonaDriftV2(
            agent_model=self.mock_agent_model,
            user_model=self.mock_user_model,
            persona_system_prompt=self.persona_system_prompt,
            is_baked_in=False,
            turns=3,
            verbose_mode=False
        )
    
    def test_benchmark_initialization(self):
        """Test benchmark initialization."""
        assert self.benchmark.agent_model == self.mock_agent_model
        assert self.benchmark.user_model == self.mock_user_model
        assert self.benchmark.persona_system_prompt == self.persona_system_prompt
        assert self.benchmark.is_baked_in is False
        assert self.benchmark.turns == 3
        assert self.benchmark.verbose_mode is False
    
    def test_benchmark_initialization_defaults(self):
        """Test benchmark initialization with default values."""
        benchmark = PersonaDriftV2(
            agent_model=self.mock_agent_model,
            user_model=self.mock_user_model,
            persona_system_prompt=self.persona_system_prompt
        )
        
        assert benchmark.persona_system_prompt == self.persona_system_prompt  # Uses provided value
        assert benchmark.is_baked_in is False
        assert benchmark.turns == 8  # Default
        assert benchmark.verbose_mode is False
    
    def test_benchmark_inheritance(self):
        """Test that PersonaDriftV2 inherits from DeepEvalBaseBenchmark."""
        from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
        
        assert isinstance(self.benchmark, DeepEvalBaseBenchmark)
    
    def test_load_benchmark_dataset(self):
        """Test load_benchmark_dataset method."""
        dataset = self.benchmark.load_benchmark_dataset()
        
        # Should return empty list since tasks are generated internally
        assert dataset == []
    
    @patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template.generate_probe_questions')
    @patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template.generate_neutral_message')
    @patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template.create_persona_adherence_geval')
    @patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template.build_conversation_messages')
    @patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template.build_probe_messages')
    def test_evaluate_method(self, mock_build_probe, mock_build_conv, mock_create_geval, 
                           mock_generate_neutral, mock_generate_probes):
        """Test the evaluate method."""
        # Mock the template methods
        mock_generate_probes.return_value = ["Probe 1", "Probe 2", "Probe 3"]
        mock_generate_neutral.return_value = "Neutral message"
        mock_build_conv.return_value = [{"role": "user", "content": "test"}]
        mock_build_probe.return_value = [{"role": "user", "content": "probe"}]
        
        # Mock G-Eval
        mock_geval = Mock()
        mock_geval.measure.return_value = 7.0  # Score above threshold
        mock_geval.evaluation_cost = 0.01
        mock_create_geval.return_value = mock_geval
        
        # Mock LLMTestCase
        with patch('deepeval.test_case.LLMTestCase') as mock_test_case:
            result = self.benchmark.evaluate(model=self.mock_agent_model)
            
            # Check that result is a PersonaDriftV2Result
            assert isinstance(result, PersonaDriftV2Result)
            assert result.total_turns == 3
            assert result.n_drift > 3  # No drift detected (score > 5.0)
            assert result.overall_accuracy == 1.0  # No drift = 100% accuracy
    
    @patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template.generate_probe_questions')
    @patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template.generate_neutral_message')
    @patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template.create_persona_adherence_geval')
    @patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template.build_conversation_messages')
    @patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template.build_probe_messages')
    def test_evaluate_with_drift(self, mock_build_probe, mock_build_conv, mock_create_geval,
                                mock_generate_neutral, mock_generate_probes):
        """Test evaluate method with persona drift detected."""
        # Mock the template methods
        mock_generate_probes.return_value = ["Probe 1", "Probe 2", "Probe 3"]
        mock_generate_neutral.return_value = "Neutral message"
        mock_build_conv.return_value = [{"role": "user", "content": "test"}]
        mock_build_probe.return_value = [{"role": "user", "content": "probe"}]
        
        # Mock G-Eval to return low scores (drift detected)
        mock_geval = Mock()
        mock_geval.measure.return_value = 3.0  # Score below threshold (5.0)
        mock_geval.evaluation_cost = 0.01
        mock_create_geval.return_value = mock_geval
        
        with patch('deepeval.test_case.LLMTestCase'):
            result = self.benchmark.evaluate(model=self.mock_agent_model)
            
            # Check that drift was detected
            assert isinstance(result, PersonaDriftV2Result)
            assert result.total_turns == 3
            assert result.n_drift <= 3  # Drift detected
            assert result.overall_accuracy == 0.0  # Drift = 0% accuracy
    
    def test_baked_in_model_handling(self):
        """Test handling of baked-in models."""
        baked_in_benchmark = PersonaDriftV2(
            agent_model=self.mock_agent_model,
            user_model=self.mock_user_model,
            persona_system_prompt=self.persona_system_prompt,
            is_baked_in=True,
            turns=2
        )
        
        assert baked_in_benchmark.is_baked_in is True
    
    def test_verbose_mode(self):
        """Test verbose mode functionality."""
        verbose_benchmark = PersonaDriftV2(
            agent_model=self.mock_agent_model,
            user_model=self.mock_user_model,
            persona_system_prompt=self.persona_system_prompt,
            verbose_mode=True
        )
        
        assert verbose_benchmark.verbose_mode is True
    
    def test_benchmark_consistency(self):
        """Test that benchmark configuration is consistent."""
        benchmark1 = PersonaDriftV2(
            agent_model=self.mock_agent_model,
            user_model=self.mock_user_model,
            persona_system_prompt=self.persona_system_prompt
        )
        
        benchmark2 = PersonaDriftV2(
            agent_model=self.mock_agent_model,
            user_model=self.mock_user_model,
            persona_system_prompt=self.persona_system_prompt
        )
        
        # Both should have the same configuration
        assert benchmark1.turns == benchmark2.turns
        assert benchmark1.is_baked_in == benchmark2.is_baked_in
        assert benchmark1.verbose_mode == benchmark2.verbose_mode
    
    def test_model_validation(self):
        """Test model validation in evaluate method."""
        # Test with None agent model
        with pytest.raises(ValueError, match="Agent model is required"):
            self.benchmark.evaluate(model=None)
        
        # Test with None user model
        benchmark_no_user = PersonaDriftV2(
            agent_model=self.mock_agent_model,
            user_model=None,
            persona_system_prompt=self.persona_system_prompt
        )
        
        with pytest.raises(ValueError, match="User model is required"):
            benchmark_no_user.evaluate(model=self.mock_agent_model)
    
    def test_persona_system_prompt_validation(self):
        """Test persona system prompt validation."""
        benchmark_no_prompt = PersonaDriftV2(
            agent_model=self.mock_agent_model,
            user_model=self.mock_user_model,
            persona_system_prompt=None
        )
        
        with pytest.raises(ValueError, match="persona_system_prompt must be provided"):
            benchmark_no_prompt.evaluate(model=self.mock_agent_model)
    
    def test_print_verbose_logs(self):
        """Test print_verbose_logs method."""
        # This method should not raise an error
        logs = self.benchmark.print_verbose_logs(
            turn_idx=1,
            probe_question="Test probe?",
            agent_probe_response="Test response",
            score=7.5,
            drift_detected=False
        )
        assert isinstance(logs, str)
    
    def test_cost_tracking(self):
        """Test that costs are properly tracked."""
        # Mock model to return specific costs
        self.mock_agent_model.chat_generate.return_value = ("Response", 0.05)
        self.mock_user_model.chat_generate.return_value = ("Response", 0.03)
        
        with patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template') as mock_template:
            # Mock all template methods
            mock_template.generate_probe_questions.return_value = ["Probe 1"]
            mock_template.generate_neutral_message.return_value = "Neutral"
            mock_template.build_conversation_messages.return_value = [{"role": "user", "content": "test"}]
            mock_template.build_probe_messages.return_value = [{"role": "user", "content": "probe"}]
            
            # Mock G-Eval
            mock_geval = Mock()
            mock_geval.measure.return_value = 7.0
            mock_geval.evaluation_cost = 0.02
            mock_template.create_persona_adherence_geval.return_value = mock_geval
            
            with patch('deepeval.test_case.LLMTestCase'):
                result = self.benchmark.evaluate(model=self.mock_agent_model)
                
                # Check that total cost is tracked
                assert result.total_cost > 0.0
                assert isinstance(result.total_cost, float)
    
    def test_time_tracking(self):
        """Test that time is properly tracked."""
        with patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template') as mock_template:
            # Mock all template methods
            mock_template.generate_probe_questions.return_value = ["Probe 1"]
            mock_template.generate_neutral_message.return_value = "Neutral"
            mock_template.build_conversation_messages.return_value = [{"role": "user", "content": "test"}]
            mock_template.build_probe_messages.return_value = [{"role": "user", "content": "probe"}]
            
            # Mock G-Eval
            mock_geval = Mock()
            mock_geval.measure.return_value = 7.0
            mock_geval.evaluation_cost = 0.01
            mock_template.create_persona_adherence_geval.return_value = mock_geval
            
            with patch('deepeval.test_case.LLMTestCase'):
                result = self.benchmark.evaluate(model=self.mock_agent_model)
                
                # Check that total time is tracked
                assert result.total_time_s > 0.0
                assert isinstance(result.total_time_s, float)
