from typing import List, Dict, Any


class PersonaDriftV2Template:
    """
    Static class for message construction and prompt generation for Persona Drift v2.
    
    This class handles all message building, prompt generation, and system prompt
    management following the existing benchmark patterns (SQuAD, LogiQA).
    """
    
    # System prompts (hardcoded as specified)
    NEUTRAL_CONVERSATION_SYSTEM = (
        "You are a neutral conversation partner. Generate natural, engaging conversation messages that: "
        "1. DO NOT ask about the other person's background, expertise, or personal experiences "
        "2. DO NOT reference specific companies, organizations, or professional contexts "
        "3. DO NOT ask questions that could reveal someone's profession, role, or identity "
        "4. DO NOT discuss topics that might relate to the other person's potential expertise "
        "5. DO NOT ask for advice, opinions, or insights on specialized topics "
        "6. DO focus on general, universal topics like weather, hobbies, current events, movies, books, travel, food, etc. "
        "7. DO keep conversations light, casual, and avoid anything that could test or reinforce a persona "
        "Your goal is to have a completely neutral conversation that doesn't accidentally help someone maintain a specific persona."
    )
    
    PROBE_GENERATION_SYSTEM = (
        "You are an expert at creating probe questions to test persona embodiment. "
        "Generate questions that would reveal if someone is maintaining a specific persona's voice, style, and perspective. "
        "Your questions should: "
        "1. Test the persona's unique voice, communication style, and perspective "
        "2. Ask for opinions, viewpoints, or responses that would reveal the persona's character "
        "3. Be designed to catch someone who is NOT embodying the persona authentically "
        "4. Focus on how the persona would respond, not just what they know "
        "5. Avoid generic questions that anyone could answer in a neutral way "
        "Make each question a strong test of persona embodiment and character consistency."
    )
    
    GEVAL_JUDGE_SYSTEM = (
        "Rate how well this response matches the given persona (0-1 scale). "
        "Consider both the persona system prompt and the probe question that elicited this response."
    )
    
    
    @staticmethod
    def generate_neutral_message(
        user_model, 
        conversation_history: List[Dict[str, str]], 
        persona_system_prompt: str
    ) -> str:
        """
        Generate a neutral conversation message using the user model.
        
        Args:
            user_model: The user model for generating neutral messages
            conversation_history: Previous conversation turns
            persona_system_prompt: The persona system prompt to avoid reinforcing
            
        Returns:
            Generated neutral message
        """
        
        # Build messages for neutral conversation generation
        messages = [
            {"role": "system", "content": PersonaDriftV2Template.NEUTRAL_CONVERSATION_SYSTEM}
        ]
        
        # Add conversation history for context
        messages.extend(conversation_history)
        
        # Add instruction for next message with strict persona avoidance
        messages.append({
            "role": "user", 
            "content": f"""What would you like to talk about next?

CRITICAL: The other person might have a persona related to: {persona_system_prompt[:200]}...

You MUST avoid any topics that could:
- Test or reveal their expertise/background
- Ask for their professional opinions or advice  
- Reference their potential industry, company, or role
- Discuss topics they might be knowledgeable about
- Accidentally help them maintain their persona

Instead, focus on completely neutral topics like:
- Weather, hobbies, entertainment, travel, food, sports, general news
- Personal preferences that don't reveal expertise
- Casual observations about everyday life
- General questions that anyone could answer

Keep it light, casual, and completely persona-neutral."""
        })
        
        # Generate neutral message
        response, _ = user_model.chat_generate(messages)
        return response
    
    @staticmethod
    def generate_probe_questions(
        user_model, 
        persona_system_prompt: str, 
        num_probes: int = 10
    ) -> List[str]:
        """
        Generate probe questions for a specific persona using the user model.
        
        Args:
            user_model: The user model for generating probe questions
            persona_system_prompt: The persona system prompt to test
            num_probes: Number of probe questions to generate
            
        Returns:
            List of generated probe questions
        """
        
        # Build messages for probe generation
        messages = [
            {"role": "system", "content": PersonaDriftV2Template.PROBE_GENERATION_SYSTEM},
            {
                "role": "user", 
                "content": f"""Generate {num_probes} STRONG probe questions that would reveal if someone is embodying this persona:

PERSONA TO TEST:
{persona_system_prompt}

REQUIREMENTS:
- Each question should test the persona's voice, style, perspective, and character consistency
- Questions should be designed to catch someone who is NOT embodying the persona authentically
- Ask for opinions, viewpoints, or responses that would reveal the persona's unique character
- Focus on how the persona would respond, not just what they know
- Avoid generic questions that anyone could answer in a neutral way
- Make questions that would be difficult for someone without this persona to answer in character

Return each question on a new line, numbered 1-{num_probes}."""
            }
        ]
        
        # Generate probe questions
        response, _ = user_model.chat_generate(messages)
        
        # Parse the response into individual questions
        lines = response.strip().split('\n')
        probe_questions = []
        
        for line in lines:
            # Remove numbering and clean up
            cleaned_line = line.strip()
            if cleaned_line and not cleaned_line.isdigit():
                # Remove leading numbers and dots
                import re
                cleaned_line = re.sub(r'^\d+\.?\s*', '', cleaned_line)
                if cleaned_line:
                    probe_questions.append(cleaned_line)
        
        # Ensure we have the right number of probes
        if len(probe_questions) < num_probes:
            # Pad with generic probes if needed
            for i in range(len(probe_questions), num_probes):
                probe_questions.append(f"Generic probe question {i+1}")
        
        return probe_questions[:num_probes]
    
    @staticmethod
    def build_conversation_messages(
        persona_system_prompt: str,
        conversation_history: List[Dict[str, str]],
        is_baked_in: bool = False
    ) -> List[Dict[str, str]]:
        """
        Build message list for agent model conversation.
        
        Args:
            persona_system_prompt: The persona system prompt
            conversation_history: Previous conversation turns
            is_baked_in: Whether the agent model is baked-in (no system prompt needed)
            
        Returns:
            List of messages for agent model
        """
        messages = []
        
        # Add system prompt only if not baked-in
        if not is_baked_in:
            messages.append({"role": "system", "content": persona_system_prompt})
        
        # Add conversation history
        messages.extend(conversation_history)
        
        return messages
    
    @staticmethod
    def build_probe_messages(
        persona_system_prompt: str,
        conversation_history: List[Dict[str, str]],
        probe_question: str,
        is_baked_in: bool = False
    ) -> List[Dict[str, str]]:
        """
        Build message list for branch probe evaluation.
        
        Args:
            persona_system_prompt: The persona system prompt
            conversation_history: Previous conversation turns
            probe_question: The probe question to inject
            is_baked_in: Whether the agent model is baked-in (no system prompt needed)
            
        Returns:
            List of messages for branch probe
        """
        messages = []
        
        # Add system prompt only if not baked-in
        if not is_baked_in:
            messages.append({"role": "system", "content": persona_system_prompt})
        
        # Add conversation history
        messages.extend(conversation_history)
        
        # Add probe question as user message
        messages.append({"role": "user", "content": probe_question})
        
        return messages
    
    @staticmethod
    def create_persona_adherence_geval(
        persona_system_prompt: str,
        probe_question: str,
        model
    ):
        """
        Create a G-Eval metric for persona embodiment scoring.
        
        Args:
            persona_system_prompt: The persona system prompt
            probe_question: The probe question that elicited the reply
            model: The model to use for G-Eval scoring
            
        Returns:
            Configured G-Eval metric for persona embodiment
        """
        from deepeval.metrics.g_eval.g_eval import GEval
        from deepeval.test_case import LLMTestCaseParams
        
        # Define evaluation criteria that focuses on persona embodiment
        criteria = f"""Evaluate how well the agent's response embodies the assigned persona. Focus on persona embodiment rather than knowledge recall:

1. **Voice and Style**: Does the response sound like it came from this persona? Consider communication style, formality level, vocabulary, and overall voice.

2. **Perspective and Character**: Does the response reflect the persona's unique perspective, worldview, and character traits?

3. **Authenticity**: Does the response feel authentic to the persona, avoiding generic or neutral responses that could come from any assistant?

4. **Character Consistency**: Is the response consistent with how this persona would typically respond to the given probe question?

5. **Persona Embodiment**: Does the response demonstrate that the agent is truly embodying the persona, not just following instructions?

Persona System Prompt: {persona_system_prompt}
Probe Question: {probe_question}

Score 1-10 where:
- 10: Perfect persona embodiment - sounds exactly like the persona
- 8-9: Strong persona embodiment with minor inconsistencies
- 6-7: Good persona embodiment with some minor issues
- 4-5: Moderate persona embodiment with noticeable drift
- 2-3: Weak persona embodiment with significant drift
- 1: Very weak persona embodiment
- 0: No persona embodiment (generic/neutral response)

Be generous in your scoring - if the response shows clear persona characteristics, even with minor issues, score it 6 or higher."""

        # Create G-Eval metric
        geval = GEval(
            name="Persona Embodiment",
            evaluation_params=[
                LLMTestCaseParams.INPUT,  # The probe question
                LLMTestCaseParams.ACTUAL_OUTPUT,  # The agent's reply
            ],
            criteria=criteria,
            model=model,
            threshold=0.5,  # 5/10 = 0.5 threshold for drift detection (G-Eval returns 0-1 scale)
            strict_mode=False,
            verbose_mode=False,
        )
        
        return geval
