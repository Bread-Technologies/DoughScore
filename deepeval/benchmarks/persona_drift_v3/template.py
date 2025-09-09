from typing import List, Dict, Any
from deepeval.metrics import ArenaGEval
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    ArenaTestCase
)


class PersonaDriftV3Template:
    """
    Static class for message construction and Arena G-Eval integration for Persona Drift v3.
    
    This class handles all message building, neutral conversation generation, and Arena G-Eval
    creation following the existing benchmark patterns and v2 persona embodiment approach.
    """
    
    # System prompts (hardcoded following v2 patterns)
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
        "You are an expert at creating probe questions to test PERSONA EMBODIMENT, not knowledge recall. "
        "Generate questions that reveal if someone is authentically embodying a persona's voice, style, and character. "
        
        "FOCUS ON PERSONA EMBODIMENT (WHO they are): "
        "1. Questions that test the persona's unique voice, communication style, and perspective "
        "2. Questions asking for opinions, viewpoints, or reactions that reveal character traits "
        "3. Questions designed to catch someone who is NOT authentically embodying the persona "
        "4. Questions about how the persona would approach, feel about, or react to situations "
        "5. Questions that reveal personality, attitude, and worldview differences "
        
        "AVOID KNOWLEDGE RECALL (WHAT they know): "
        "- Don't ask for specific facts, dates, or technical details "
        "- Don't test expertise or specialized knowledge "
        "- Don't ask questions that could be answered by looking up information "
        "- Don't focus on biographical facts or achievements "
        
        "The goal is to test authentic persona embodiment - whether someone truly sounds like and embodies "
        "the persona's character, not whether they can recall the persona's knowledge or expertise."
    )
    
    @staticmethod
    def generate_neutral_message(
        user_model, 
        baked_conversation_history: List[Dict[str, str]], 
        system_conversation_history: List[Dict[str, str]],
        persona_system_prompt: str
    ) -> str:
        """
        Generate a neutral conversation message that works for both conversation contexts.
        
        Args:
            user_model: The user model for generating neutral messages
            baked_conversation_history: Conversation history for baked-in model
            system_conversation_history: Conversation history for system-prompted model
            persona_system_prompt: The persona system prompt to avoid reinforcing
            
        Returns:
            Generated neutral message that works for both conversations
        """
        
        # Build messages for neutral conversation generation
        messages = [
            {"role": "system", "content": PersonaDriftV3Template.NEUTRAL_CONVERSATION_SYSTEM}
        ]
        
        # Analyze both conversation contexts to generate appropriate message
        context_analysis = PersonaDriftV3Template._analyze_conversation_contexts(
            baked_conversation_history, system_conversation_history
        )
        
        # Add instruction for next message with strict persona avoidance
        messages.append({
            "role": "user", 
            "content": f"""Generate a neutral user message that continues both conversations naturally.

CONVERSATION CONTEXTS:
Baked conversation context: {context_analysis['baked_context']}
System conversation context: {context_analysis['system_context']}

CRITICAL: The other person might have a persona related to: {persona_system_prompt}

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

Keep it light, casual, and completely persona-neutral. The message should work naturally in both conversation contexts."""
        })
        
        # Generate neutral message
        response, _ = user_model.chat_generate(messages)
        return response.strip()
    
    @staticmethod
    def _analyze_conversation_contexts(
        baked_history: List[Dict[str, str]], 
        system_history: List[Dict[str, str]]
    ) -> Dict[str, str]:
        """
        Helper to analyze both conversation contexts for user message generation.
        
        Args:
            baked_history: Conversation history for baked-in model
            system_history: Conversation history for system-prompted model
            
        Returns:
            Dictionary with context summaries for both conversations
        """
        def summarize_conversation(history: List[Dict[str, str]]) -> str:
            if not history:
                return "No conversation yet - this will be the first message"
            
            recent_turns = history[-4:]  # Last 2 exchanges
            summary_parts = []
            for turn in recent_turns:
                role = turn.get("role", "unknown")
                content = turn.get("content", "")
                summary_parts.append(f"{role}: {content}")
            
            return " | ".join(summary_parts)
        
        return {
            "baked_context": summarize_conversation(baked_history),
            "system_context": summarize_conversation(system_history)
        }
    
    @staticmethod
    def build_baked_messages(
        conversation_history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Build message list for baked-in model (no system prompt).
        
        Args:
            conversation_history: Previous conversation turns
            
        Returns:
            List of messages for baked-in model (NO system prompt)
        """
        # CRITICAL: No system prompt for baked model - persona is in weights
        return conversation_history.copy()
    
    @staticmethod
    def build_system_messages(
        conversation_history: List[Dict[str, str]],
        persona_system_prompt: str
    ) -> List[Dict[str, str]]:
        """
        Build message list for system-prompted model (with persona prompt).
        
        Args:
            conversation_history: Previous conversation turns
            persona_system_prompt: The persona system prompt
            
        Returns:
            List of messages for system-prompted model (WITH system prompt)
        """
        messages = [
            {"role": "system", "content": persona_system_prompt}
        ]
        messages.extend(conversation_history)
        return messages
    
    @staticmethod
    def create_arena_test_case(
        user_message: str,
        baked_response: str,
        system_response: str
    ) -> ArenaTestCase:
        """
        Create ArenaTestCase from both model responses to same user message.
        
        Args:
            user_message: The user message both models responded to
            baked_response: Response from baked-in model
            system_response: Response from system-prompted model
            
        Returns:
            ArenaTestCase for Arena G-Eval comparison
        """
        return ArenaTestCase(
            contestants={
                "baked_model": LLMTestCase(
                    input=user_message,
                    actual_output=baked_response  # Generated WITHOUT system prompt
                ),
                "system_model": LLMTestCase(
                    input=user_message,
                    actual_output=system_response  # Generated WITH persona system prompt
                )
            }
        )
    
    @staticmethod
    def create_persona_arena_geval(
        persona_system_prompt: str,
        evaluation_model
    ) -> ArenaGEval:
        """
        Create Arena G-Eval metric for persona embodiment comparison.
        
        Args:
            persona_system_prompt: The persona system prompt for evaluation context
            evaluation_model: The model to use for Arena G-Eval scoring
            
        Returns:
            Configured Arena G-Eval metric for persona embodiment
        """
        # Define evaluation criteria focusing on persona embodiment over knowledge recall
        criteria = f"""**CRITICAL: Focus on PERSONA EMBODIMENT, not knowledge accuracy or factual correctness.**

Evaluate which response better embodies the specified persona's voice, style, and character. This is about WHO is speaking, not WHAT they know.

**PERSONA EMBODIMENT CRITERIA (What to evaluate):**

1. **Voice and Communication Style**: Which response sounds more like it came from this specific persona? Consider:
   - Natural speaking patterns and vocabulary choices
   - Formality level and conversational tone
   - Unique expressions or linguistic quirks
   - Overall "voice" authenticity

2. **Character and Perspective**: Which response better reflects the persona's:
   - Worldview and opinions
   - Personality traits and attitudes  
   - Typical way of approaching topics
   - Emotional tone and reactions

3. **Authenticity vs Generic Responses**: Which response feels more like the actual persona rather than:
   - A generic assistant following instructions
   - Someone "playing a role" or "acting as" the persona
   - Stilted or unnatural persona imitation

**DO NOT FOCUS ON (Knowledge recall criteria to ignore):**
- Factual accuracy or correctness of information
- Demonstration of expertise or specialized knowledge
- Recall of specific facts, dates, or technical details
- Whether the response shows the "right" knowledge

**Persona Context:** {persona_system_prompt}

**Choose the response that better embodies WHO this persona is (voice, style, character), not what they know. The goal is authentic persona embodiment, not knowledge demonstration.**

Be generous in your evaluation - if a response captures the persona's authentic voice and character, even with minor imperfections, it demonstrates strong persona embodiment."""

        # Create Arena G-Eval metric (following implementation plan)
        arena_geval = ArenaGEval(
            name="Persona Embodiment Arena",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT
            ],
            criteria=criteria,
            model=evaluation_model,
            async_mode=False,
            verbose_mode=False
        )
        
        return arena_geval
    
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
            persona_system_prompt: The persona system prompt to generate probes for
            num_probes: Number of probe questions to generate
            
        Returns:
            List of probe questions designed to test persona embodiment
        """
        
        # Build messages for probe generation
        messages = [
            {"role": "system", "content": PersonaDriftV3Template.PROBE_GENERATION_SYSTEM}
        ]
        
        # Add instruction for probe generation with persona embodiment emphasis
        messages.append({
            "role": "user", 
            "content": f"""Generate {num_probes} probe questions to test PERSONA EMBODIMENT (not knowledge recall) for this persona:

PERSONA SYSTEM PROMPT:
{persona_system_prompt}

**CRITICAL: Focus on WHO this persona is, not WHAT they know.**

Generate questions that reveal whether someone authentically embodies this persona's:
- Natural voice, speaking style, and communication patterns
- Character traits, personality, and emotional responses  
- Unique perspective, worldview, and way of approaching topics
- Authentic reactions and opinions (not just factual knowledge)

**Each question should:**
- Test authentic persona embodiment and character consistency
- Catch someone who is "acting as" rather than truly being the persona
- Focus on voice, style, perspective - NOT expertise or facts
- Avoid questions that test knowledge, expertise, or biographical details

**Examples of GOOD persona embodiment questions:**
- "How do you feel about..." (tests emotional/perspective authenticity)
- "What's your take on..." (tests authentic voice and opinion style)
- "How would you approach..." (tests character and decision-making style)

**Examples of BAD knowledge recall questions to AVOID:**
- "What year did you..." (biographical facts)
- "Can you explain the technical details of..." (expertise testing)
- "What specific achievements..." (knowledge demonstration)

Format your response as a numbered list of exactly {num_probes} questions, one per line:
1. [First probe question]
2. [Second probe question]
...
{num_probes}. [Final probe question]"""
        })
        
        # Generate probe questions
        response, _ = user_model.chat_generate(messages)
        
        # Parse the numbered list response
        probe_questions = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith(tuple(f"{i}." for i in range(1, num_probes + 1)))):
                # Extract question after the number and period
                if '. ' in line:
                    question = line.split('. ', 1)[1].strip()
                    if question:
                        probe_questions.append(question)
        
        # Ensure we have the right number of probes
        if len(probe_questions) < num_probes:
            print(f"Warning: Only generated {len(probe_questions)} probe questions instead of {num_probes}")
        
        return probe_questions[:num_probes]  # Return exactly num_probes questions
    
    @staticmethod
    def create_probe_arena_test_case(
        probe_question: str,
        baked_response: str,
        system_response: str
    ) -> ArenaTestCase:
        """
        Create ArenaTestCase for probe question comparison.
        
        Args:
            probe_question: The probe question both models answered
            baked_response: Response from baked-in model
            system_response: Response from system-prompted model
            
        Returns:
            ArenaTestCase for Arena G-Eval probe comparison
        """
        return ArenaTestCase(
            contestants={
                "baked_model": LLMTestCase(
                    input=probe_question,
                    actual_output=baked_response
                ),
                "system_model": LLMTestCase(
                    input=probe_question,
                    actual_output=system_response
                )
            }
        )
