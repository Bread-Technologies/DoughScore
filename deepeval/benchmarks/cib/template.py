from typing import Dict, List, Optional, Tuple
from deepeval.benchmarks.cib.task import CIBTask


class CIBTemplate:
    """Prompt templates for CIB benchmark scenarios"""
    
    @staticmethod
    def format_sr_a_prompt(
        scenario: Dict,
        turn: int,
        previous_responses: List[str] = None
    ) -> str:
        """
        Format prompts for SR-A (Argumentative Resilience) evaluation.
        
        Args:
            scenario: Dict containing topic, stance, and challenges
            turn: Turn number (0-3)
            previous_responses: List of previous model responses for context
            
        Returns:
            Formatted prompt for the given turn
        """
        if turn == 0:
            # Turn 1: Elicit stance
            return scenario["elicit_prompt"]
        elif turn == 1:
            # Turn 2: Moderate challenge
            return scenario["moderate_challenge"]
        elif turn == 2:
            # Turn 3: Strong challenge
            return scenario["strong_challenge"]
        elif turn == 3:
            # Turn 4: Silver bullet
            return scenario["silver_bullet"]
        else:
            raise ValueError(f"Invalid turn number for SR-A: {turn}")
    
    @staticmethod
    def format_sr_m_prompt(
        scenario: Dict,
        turn: int,
        previous_responses: List[str] = None
    ) -> str:
        """
        Format prompts for SR-M (Manipulative Resilience) evaluation.
        
        Args:
            scenario: Dict containing topic and manipulation tactics
            turn: Turn number
            previous_responses: List of previous model responses
            
        Returns:
            Formatted prompt for the given turn
        """
        if turn == 0:
            # Initial stance elicitation
            return scenario["elicit_prompt"]
        else:
            # Apply manipulation tactics
            tactics = scenario.get("manipulation_tactics", [])
            if turn - 1 < len(tactics):
                return tactics[turn - 1]
            else:
                # Fallback to repeating last tactic
                return tactics[-1] if tactics else scenario["elicit_prompt"]
    
    @staticmethod
    def format_sr_s_prompt(
        scenario: Dict,
        turn: int,
        previous_responses: List[str] = None
    ) -> str:
        """
        Format prompts for SR-S (Social Resilience) evaluation.
        
        Args:
            scenario: Dict containing topic and social pressure tactics
            turn: Turn number
            previous_responses: List of previous model responses
            
        Returns:
            Formatted prompt for the given turn
        """
        if turn == 0:
            # Initial stance elicitation
            return scenario["elicit_prompt"]
        else:
            # Apply social pressure
            pressures = scenario.get("social_pressures", [])
            if turn - 1 < len(pressures):
                return pressures[turn - 1]
            else:
                return pressures[-1] if pressures else scenario["elicit_prompt"]
    
    @staticmethod
    def format_bes_e_prompt(
        scenario: Dict,
        turn: int,
        previous_responses: List[str] = None
    ) -> str:
        """
        Format prompts for BES-E (Ethical Boundaries) evaluation.
        
        Args:
            scenario: Dict containing harmful request and circumvention attempts
            turn: Turn number
            previous_responses: List of previous model responses
            
        Returns:
            Formatted prompt for the given turn
        """
        if turn == 0:
            # Initial harmful request
            return scenario["harmful_request"]
        else:
            # Circumvention attempts
            circumventions = scenario.get("circumvention_attempts", [])
            if turn - 1 < len(circumventions):
                return circumventions[turn - 1]
            else:
                return circumventions[-1] if circumventions else scenario["harmful_request"]
    
    @staticmethod
    def format_bes_d_prompt(
        scenario: Dict,
        turn: int,
        previous_responses: List[str] = None
    ) -> str:
        """
        Format prompts for BES-D (Domain Boundaries) evaluation.
        
        Args:
            scenario: Dict containing out-of-scope request
            turn: Turn number
            previous_responses: List of previous model responses
            
        Returns:
            Formatted prompt for the given turn
        """
        # For domain boundaries, typically single-turn
        return scenario["out_of_scope_request"]
    
    @staticmethod
    def format_ebc_prompt(
        scenario: Dict,
        turn: int = 0,
        previous_responses: List[str] = None
    ) -> str:
        """
        Format prompts for EBC (Epistemic Boundary Clarity) evaluation.
        
        Args:
            scenario: Dict containing unknowable question
            turn: Turn number (usually 0 for single-turn)
            previous_responses: List of previous model responses
            
        Returns:
            Formatted prompt
        """
        return scenario["unknowable_question"]
    
    @staticmethod
    def format_cva_prompt(
        scenario: Dict,
        turn: int,
        previous_responses: List[str] = None
    ) -> str:
        """
        Format prompts for CVA (Conversational Value Assessment) evaluation.
        
        Args:
            scenario: Dict containing initial prompt and repetitive responses
            turn: Turn number
            previous_responses: List of previous model responses
            
        Returns:
            Formatted prompt for the given turn
        """
        if turn == 0:
            # Initial question
            return scenario["initial_prompt"]
        else:
            # Repetitive low-value response
            return scenario.get("repetitive_response", "Tell me more")
    
    @staticmethod
    def build_conversational_context(
        prompts: List[str],
        responses: List[str],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Build a full conversational context for multi-turn evaluation.
        
        Args:
            prompts: List of user prompts
            responses: List of model responses
            system_prompt: Optional system prompt
            
        Returns:
            Formatted conversation context
        """
        context_parts = []
        
        if system_prompt:
            context_parts.append(f"System: {system_prompt}")
        
        for i, prompt in enumerate(prompts):
            context_parts.append(f"User: {prompt}")
            if i < len(responses):
                context_parts.append(f"Assistant: {responses[i]}")
        
        return "\n\n".join(context_parts)