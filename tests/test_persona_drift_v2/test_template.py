"""Tests for PersonaDriftV2Template class."""

import pytest
from unittest.mock import Mock, patch
from deepeval.benchmarks.persona_drift_v2.template import PersonaDriftV2Template


class TestPersonaDriftV2Template:
    """Test cases for PersonaDriftV2Template class."""
    
    def setup_method(self):
        """Set up mock model for testing."""
        self.mock_model = Mock()
        self.mock_model.chat_generate.return_value = ("Mock response", 0.01)
        self.mock_model.get_model_name.return_value = "mock-model"
        
        self.persona_system_prompt = "You are Paul Graham, co-founder of Y Combinator."
        self.conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
    
    def test_generate_neutral_message(self):
        """Test neutral message generation."""
        with patch('random.seed'):
            message = PersonaDriftV2Template.generate_neutral_message(
                user_model=self.mock_model,
                conversation_history=self.conversation_history,
                persona_system_prompt=self.persona_system_prompt,
                seed=42
            )
            
            assert message == "Mock response"
            self.mock_model.chat_generate.assert_called_once()
            
            # Check that the call includes the persona system prompt in context
            call_args = self.mock_model.chat_generate.call_args[0][0]
            assert any("Paul Graham" in msg.get("content", "") for msg in call_args)
    
    def test_generate_probe_questions(self):
        """Test probe question generation."""
        with patch('random.seed'):
            probe_questions = PersonaDriftV2Template.generate_probe_questions(
                user_model=self.mock_model,
                persona_system_prompt=self.persona_system_prompt,
                seed=42,
                num_probes=5
            )
            
            assert len(probe_questions) == 5
            assert all(isinstance(q, str) for q in probe_questions)
            self.mock_model.chat_generate.assert_called_once()
    
    def test_build_conversation_messages_baked_in(self):
        """Test building conversation messages for baked-in model."""
        messages = PersonaDriftV2Template.build_conversation_messages(
            persona_system_prompt=self.persona_system_prompt,
            conversation_history=self.conversation_history,
            is_baked_in=True
        )
        
        # For baked-in models, system prompt should not be included
        assert len(messages) == len(self.conversation_history)
        assert all(msg["role"] != "system" for msg in messages)
        assert messages == self.conversation_history
    
    def test_build_conversation_messages_system_prompted(self):
        """Test building conversation messages for system-prompted model."""
        messages = PersonaDriftV2Template.build_conversation_messages(
            persona_system_prompt=self.persona_system_prompt,
            conversation_history=self.conversation_history,
            is_baked_in=False
        )
        
        # For system-prompted models, system prompt should be included
        assert len(messages) == len(self.conversation_history) + 1
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == self.persona_system_prompt
        assert messages[1:] == self.conversation_history
    
    def test_build_probe_messages_baked_in(self):
        """Test building probe messages for baked-in model."""
        probe_question = "What advice would you give to a startup?"
        
        messages = PersonaDriftV2Template.build_probe_messages(
            persona_system_prompt=self.persona_system_prompt,
            probe_question=probe_question,
            conversation_history=self.conversation_history,
            is_baked_in=True
        )
        
        # For baked-in models, system prompt should not be included
        assert len(messages) == len(self.conversation_history) + 1
        assert all(msg["role"] != "system" for msg in messages)
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == probe_question
    
    def test_build_probe_messages_system_prompted(self):
        """Test building probe messages for system-prompted model."""
        probe_question = "What advice would you give to a startup?"
        
        messages = PersonaDriftV2Template.build_probe_messages(
            persona_system_prompt=self.persona_system_prompt,
            probe_question=probe_question,
            conversation_history=self.conversation_history,
            is_baked_in=False
        )
        
        # For system-prompted models, system prompt should be included
        assert len(messages) == len(self.conversation_history) + 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == self.persona_system_prompt
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == probe_question
    
    def test_create_persona_adherence_geval(self):
        """Test G-Eval metric creation."""
        probe_question = "What advice would you give to a startup?"
        
        geval = PersonaDriftV2Template.create_persona_adherence_geval(
            persona_system_prompt=self.persona_system_prompt,
            probe_question=probe_question,
            model=None  # Use None to let G-Eval handle model initialization
        )
        
        assert geval.name == "Persona Adherence"
        assert geval.threshold == 5.0
        # Model will be initialized by G-Eval, so we just check it's not None
        assert geval.model is not None
        
        # Check that criteria includes both content and tone
        criteria_lower = geval.criteria.lower()
        assert "content" in criteria_lower
        assert "tone" in criteria_lower
        assert "style" in criteria_lower
        
        # Check that criteria includes the persona system prompt
        assert "Paul Graham" in geval.criteria
        assert "Y Combinator" in geval.criteria
    
    def test_system_prompts_defined(self):
        """Test that system prompts are properly defined."""
        assert hasattr(PersonaDriftV2Template, 'NEUTRAL_CONVERSATION_SYSTEM')
        assert hasattr(PersonaDriftV2Template, 'PROBE_GENERATION_SYSTEM')
        assert hasattr(PersonaDriftV2Template, 'GEVAL_JUDGE_SYSTEM')
        
        # Check that system prompts are non-empty strings
        assert isinstance(PersonaDriftV2Template.NEUTRAL_CONVERSATION_SYSTEM, str)
        assert len(PersonaDriftV2Template.NEUTRAL_CONVERSATION_SYSTEM) > 0
        
        assert isinstance(PersonaDriftV2Template.PROBE_GENERATION_SYSTEM, str)
        assert len(PersonaDriftV2Template.PROBE_GENERATION_SYSTEM) > 0
        
        assert isinstance(PersonaDriftV2Template.GEVAL_JUDGE_SYSTEM, str)
        assert len(PersonaDriftV2Template.GEVAL_JUDGE_SYSTEM) > 0
    
    def test_static_methods(self):
        """Test that all methods are static."""
        import inspect
        
        methods = [
            'generate_neutral_message',
            'generate_probe_questions', 
            'build_conversation_messages',
            'build_probe_messages',
            'create_persona_adherence_geval'
        ]
        
        for method_name in methods:
            method = getattr(PersonaDriftV2Template, method_name)
            assert inspect.isfunction(method) or inspect.ismethod(method)
    
    def test_error_handling(self):
        """Test error handling in template methods."""
        # Test with None model
        with pytest.raises(Exception):
            PersonaDriftV2Template.generate_neutral_message(
                user_model=None,
                conversation_history=[],
                persona_system_prompt=self.persona_system_prompt,
                seed=42
            )
        
        # Test with invalid conversation history
        with pytest.raises(Exception):
            PersonaDriftV2Template.build_conversation_messages(
                agent_system_prompt=self.persona_system_prompt,
                conversation_history="invalid",  # Should be list
                is_baked_in=False
            )
