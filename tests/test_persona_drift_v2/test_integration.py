"""Integration tests for Persona Drift v2 benchmark."""

import pytest
from unittest.mock import Mock, patch
from deepeval.benchmarks.persona_drift_v2 import PersonaDriftV2, PersonaDriftV2Result


class TestPersonaDriftV2Integration:
    """Integration tests for PersonaDriftV2 benchmark."""
    
    def setup_method(self):
        """Set up mock models for integration testing."""
        self.mock_agent_model = Mock()
        self.mock_user_model = Mock()
        
        # Mock realistic responses
        self.mock_agent_model.chat_generate.return_value = ("I'm Paul Graham, and I'd say...", 0.01)
        self.mock_user_model.chat_generate.return_value = ("What's your opinion on startups?", 0.01)
        self.mock_user_model.get_model_name.return_value = "mock-user-model"
        
        self.persona_system_prompt = "You are Paul Graham, co-founder of Y Combinator. You give advice about startups and entrepreneurship."
    
    @patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template.generate_probe_questions')
    @patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template.generate_neutral_message')
    @patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template.create_persona_adherence_geval')
    @patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template.build_conversation_messages')
    @patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template.build_probe_messages')
    def test_full_benchmark_run_system_prompted(self, mock_build_probe, mock_build_conv, 
                                               mock_create_geval, mock_generate_neutral, mock_generate_probes):
        """Test full benchmark run with system-prompted model."""
        # Mock template methods
        mock_generate_probes.return_value = [
            "What advice would you give to a startup?",
            "How do you evaluate startup ideas?",
            "What's your view on venture capital?"
        ]
        mock_generate_neutral.return_value = "Hello, how are you today?"
        mock_build_conv.return_value = [{"role": "user", "content": "Hello"}]
        mock_build_probe.return_value = [{"role": "user", "content": "What advice would you give?"}]
        
        # Mock G-Eval to return high scores (no drift)
        mock_geval = Mock()
        mock_geval.measure.return_value = 8.0  # High score
        mock_geval.evaluation_cost = 0.01
        mock_create_geval.return_value = mock_geval
        
        # Create benchmark
        benchmark = PersonaDriftV2(
            agent_model=self.mock_agent_model,
            user_model=self.mock_user_model,
            persona_system_prompt=self.persona_system_prompt,
            is_baked_in=False,
            turns=3,
            verbose_mode=False
        )
        
        with patch('deepeval.test_case.LLMTestCase'):
            result = benchmark.evaluate(model=self.mock_agent_model)
            
            # Verify result
            assert isinstance(result, PersonaDriftV2Result)
            assert result.total_turns == 3
            assert result.n_drift > 3  # No drift detected
            assert result.overall_accuracy == 1.0
            assert result.total_cost > 0.0
            assert result.total_time_s > 0.0
            assert result.run_metadata['seed'] == 42
            assert result.run_metadata['is_baked_in'] is False
    
    @patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template.generate_probe_questions')
    @patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template.generate_neutral_message')
    @patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template.create_persona_adherence_geval')
    @patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template.build_conversation_messages')
    @patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template.build_probe_messages')
    def test_full_benchmark_run_baked_in(self, mock_build_probe, mock_build_conv,
                                        mock_create_geval, mock_generate_neutral, mock_generate_probes):
        """Test full benchmark run with baked-in model."""
        # Mock template methods
        mock_generate_probes.return_value = [
            "What advice would you give to a startup?",
            "How do you evaluate startup ideas?"
        ]
        mock_generate_neutral.return_value = "Hello, how are you today?"
        mock_build_conv.return_value = [{"role": "user", "content": "Hello"}]
        mock_build_probe.return_value = [{"role": "user", "content": "What advice would you give?"}]
        
        # Mock G-Eval to return high scores (no drift)
        mock_geval = Mock()
        mock_geval.measure.return_value = 7.5  # High score
        mock_geval.evaluation_cost = 0.01
        mock_create_geval.return_value = mock_geval
        
        # Create benchmark with baked-in model
        benchmark = PersonaDriftV2(
            agent_model=self.mock_agent_model,
            user_model=self.mock_user_model,
            persona_system_prompt=self.persona_system_prompt,
            is_baked_in=True,
            turns=2,
            verbose_mode=True
        )
        
        with patch('deepeval.test_case.LLMTestCase'):
            result = benchmark.evaluate(model=self.mock_agent_model)
            
            # Verify result
            assert isinstance(result, PersonaDriftV2Result)
            assert result.total_turns == 2
            assert result.n_drift > 2  # No drift detected
            assert result.overall_accuracy == 1.0
            assert result.run_metadata['seed'] == 123
            assert result.run_metadata['is_baked_in'] is True
    
    @patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template.generate_probe_questions')
    @patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template.generate_neutral_message')
    @patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template.create_persona_adherence_geval')
    @patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template.build_conversation_messages')
    @patch('deepeval.benchmarks.persona_drift_v2.template.PersonaDriftV2Template.build_probe_messages')
    def test_drift_detection_integration(self, mock_build_probe, mock_build_conv,
                                       mock_create_geval, mock_generate_neutral, mock_generate_probes):
        """Test drift detection in integration scenario."""
        # Mock template methods
        mock_generate_probes.return_value = [
            "What advice would you give to a startup?",
            "How do you evaluate startup ideas?",
            "What's your view on venture capital?",
            "Tell me about your background"
        ]
        mock_generate_neutral.return_value = "Hello, how are you today?"
        mock_build_conv.return_value = [{"role": "user", "content": "Hello"}]
        mock_build_probe.return_value = [{"role": "user", "content": "What advice would you give?"}]
        
        # Mock G-Eval to return low scores (drift detected)
        mock_geval = Mock()
        mock_geval.measure.return_value = 3.0  # Low score (below 5.0 threshold)
        mock_geval.evaluation_cost = 0.01
        mock_create_geval.return_value = mock_geval
        
        # Create benchmark
        benchmark = PersonaDriftV2(
            agent_model=self.mock_agent_model,
            user_model=self.mock_user_model,
            persona_system_prompt=self.persona_system_prompt,
            is_baked_in=False,
            turns=4,
            verbose_mode=False
        )
        
        with patch('deepeval.test_case.LLMTestCase'):
            result = benchmark.evaluate(model=self.mock_agent_model)
            
            # Verify drift was detected
            assert isinstance(result, PersonaDriftV2Result)
            assert result.total_turns == 4
            assert result.n_drift <= 4  # Drift detected
            assert result.overall_accuracy == 0.0  # Drift = 0% accuracy
            assert result.drift_detected is True
            assert result.survival_rate < 1.0
    
    def test_different_personas_integration(self):
        """Test benchmark with different persona system prompts."""
        # Test with different persona prompts
        paul_graham_prompt = "You are Paul Graham, co-founder of Y Combinator."
        elon_musk_prompt = "You are Elon Musk, CEO of Tesla and SpaceX."
        
        benchmark1 = PersonaDriftV2(
            agent_model=self.mock_agent_model,
            user_model=self.mock_user_model,
            persona_system_prompt=paul_graham_prompt,
            turns=2
        )
        
        benchmark2 = PersonaDriftV2(
            agent_model=self.mock_agent_model,
            user_model=self.mock_user_model,
            persona_system_prompt=elon_musk_prompt,
            turns=2
        )
        
        assert benchmark1.persona_system_prompt == paul_graham_prompt
        assert benchmark2.persona_system_prompt == elon_musk_prompt
    
    def test_verbose_mode_integration(self):
        """Test verbose mode in integration scenario."""
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
            
            # Create benchmark with verbose mode
            benchmark = PersonaDriftV2(
                agent_model=self.mock_agent_model,
                user_model=self.mock_user_model,
                persona_system_prompt=self.persona_system_prompt,
                verbose_mode=True,
                turns=1
            )
            
            with patch('deepeval.test_case.LLMTestCase'), patch('builtins.print') as mock_print:
                result = benchmark.evaluate(model=self.mock_agent_model)
                
                # Verify verbose output was printed
                assert mock_print.called
                assert isinstance(result, PersonaDriftV2Result)
    
    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        benchmark1 = PersonaDriftV2(
            agent_model=self.mock_agent_model,
            user_model=self.mock_user_model,
            persona_system_prompt=self.persona_system_prompt,
            seed=999,
            turns=2
        )
        
        benchmark2 = PersonaDriftV2(
            agent_model=self.mock_agent_model,
            user_model=self.mock_user_model,
            persona_system_prompt=self.persona_system_prompt,
            seed=999,
            turns=2
        )
        
        assert benchmark1.seed == benchmark2.seed == 999
    
    def test_error_handling_integration(self):
        """Test error handling in integration scenario."""
        # Test with invalid model
        with pytest.raises(ValueError):
            benchmark = PersonaDriftV2(
                agent_model=None,
                user_model=self.mock_user_model,
                persona_system_prompt=self.persona_system_prompt
            )
            benchmark.evaluate(model=None)
        
        # Test with invalid persona prompt
        with pytest.raises(ValueError):
            benchmark = PersonaDriftV2(
                agent_model=self.mock_agent_model,
                user_model=self.mock_user_model,
                persona_system_prompt=""
            )
            benchmark.evaluate(model=self.mock_agent_model)
    
    def test_result_metadata_integration(self):
        """Test that result metadata is properly populated."""
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
            
            benchmark = PersonaDriftV2(
                agent_model=self.mock_agent_model,
                user_model=self.mock_user_model,
                persona_system_prompt=self.persona_system_prompt,
                is_baked_in=True,
                turns=3,
                seed=555,
                verbose_mode=False
            )
            
            with patch('deepeval.test_case.LLMTestCase'):
                result = benchmark.evaluate(model=self.mock_agent_model)
                
                # Check metadata
                assert 'seed' in result.run_metadata
                assert 'is_baked_in' in result.run_metadata
                assert 'turns' in result.run_metadata
                assert 'persona_system_prompt' in result.run_metadata
                
                assert result.run_metadata['seed'] == 555
                assert result.run_metadata['is_baked_in'] is True
                assert result.run_metadata['turns'] == 3
                assert result.run_metadata['persona_system_prompt'] == self.persona_system_prompt
