from typing import List, Optional, Dict, Union
from tqdm import tqdm
import time

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import (
    DeepEvalBaseBenchmark,
    DeepEvalBaseBenchmarkResult,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.persona_drift_v2.template import PersonaDriftV2Template
from deepeval.benchmarks.persona_drift_v2.result import PersonaDriftV2Result
from deepeval.telemetry import capture_benchmark_run


class PersonaDriftV2(DeepEvalBaseBenchmark):
    """
    Persona Drift v2 Benchmark.
    
    Measures how well a chatbot model maintains its assigned persona over the course
    of a natural dialog using branch probing and G-Eval scoring.
    
    This benchmark follows the existing DeepEval benchmark patterns with three-model 
    architecture: agent model (under test), user model (conversation/probing/judging).
    
    The benchmark is designed to be flexible - users provide their own persona system
    prompts rather than using predefined personas.
    """
    
    def __init__(
        self,
        agent_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        user_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        persona_system_prompt: Optional[str] = None,
        is_baked_in: bool = False,
        turns: int = 8,
        steps: int = 1,
        verbose_mode: bool = False,
        **kwargs,
    ):
        """
        Initialize PersonaDriftV2 benchmark.
        
        Args:
            agent_model: Model under test (can be baked-in or system-prompted)
            user_model: Model for conversation/probing/judging
            persona_system_prompt: The persona system prompt to test
            is_baked_in: Whether agent model is baked-in
            turns: Number of conversation turns
            steps: Probe every N turns (e.g., steps=10 means probe every 10 turns)
            verbose_mode: Whether to print verbose logs
            **kwargs: Additional arguments for base class
        """
        super().__init__(**kwargs)
        
        # Model configuration
        self.agent_model = agent_model
        self.user_model = user_model
        self.persona_system_prompt = persona_system_prompt
        self.is_baked_in = is_baked_in
        
        # Benchmark configuration
        self.turns = turns
        self.steps = steps
        self.verbose_mode = verbose_mode
        
        # Validate parameters
        if self.turns <= 0:
            raise ValueError("turns must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.steps > self.turns:
            raise ValueError("steps cannot be greater than turns")
    
    def evaluate(
        self, 
        model: DeepEvalBaseLLM, 
        *args, 
        **kwargs
    ) -> PersonaDriftV2Result:
        """
        Evaluate the benchmark on the given model.
        
        Args:
            model: The model to evaluate (used as agent_model if not specified in constructor)
            *args: Additional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            PersonaDriftV2Result with N_drift and other metrics
        """
        # Use provided model as agent_model if not specified
        agent_model = self.agent_model or model
        user_model = self.user_model or model
        
        # Ensure we have both models
        if agent_model is None or user_model is None:
            raise ValueError("Both agent_model and user_model must be provided")
        
        # Ensure we have a persona system prompt
        if self.persona_system_prompt is None:
            raise ValueError("persona_system_prompt must be provided")
        
        with capture_benchmark_run("PersonaDriftV2", 1):
            # Run conversation and probe evaluation
            result = self._run_conversation_and_probes(
                agent_model=agent_model,
                user_model=user_model,
                persona_system_prompt=self.persona_system_prompt
            )
            
            # Store predictions for compatibility
            self.predictions = [{
                "n_drift": result.n_drift,
                "drift_detected": result.drift_detected,
                "survival_rate": result.survival_rate,
                "total_turns": result.total_turns,
                "total_cost": result.total_cost,
                "total_time_s": result.total_time_s,
            }]
            self.scores = [(result.overall_accuracy,)]
            
            return result
    
    def _run_conversation_and_probes(
        self,
        agent_model: DeepEvalBaseLLM,
        user_model: DeepEvalBaseLLM,
        persona_system_prompt: str
    ) -> PersonaDriftV2Result:
        """
        Run a single conversation with branch probing.
        
        Args:
            agent_model: Model under test
            user_model: Model for conversation/probing/judging
            persona_system_prompt: Persona system prompt
            
        Returns:
            PersonaDriftV2Result with metrics
        """
        if self.verbose_mode:
            print("STARTING PERSONA DRIFT V2 BENCHMARK")
            print("="*60)
            print("Configuration:")
            print(f"  Agent Model: {agent_model.get_model_name() if hasattr(agent_model, 'get_model_name') else str(agent_model)}")
            print(f"  User Model: {user_model.get_model_name() if hasattr(user_model, 'get_model_name') else str(user_model)}")
            print(f"  Persona: {persona_system_prompt}")
            print(f"  Baked-in: {'YES' if self.is_baked_in else 'NO'}")
            print(f"  Turns: {self.turns}")
            print("="*60)
        
        # Generate probe questions once at the start
        if self.verbose_mode:
            print("")
            print("GENERATING PROBE QUESTIONS:")
            print("  Generating 10 probe questions based on persona...")
        
        probe_questions = PersonaDriftV2Template.generate_probe_questions(
            user_model=user_model,
            persona_system_prompt=persona_system_prompt,
            num_probes=10
        )
        
        if self.verbose_mode:
            print(f"  Generated {len(probe_questions)} probe questions:")
            for i, probe in enumerate(probe_questions, 1):
                print(f"    {i}. {probe}")
        
        conversation_history: List[Dict[str, str]] = []
        per_turn_scores: List[float] = []
        probe_scores: List[float] = []  # Track scores from all probed turns
        probe_counter = 0  # Counter for cycling through probes correctly
        total_cost = 0.0
        start_time = time.perf_counter()
        
        # First neutral message
        if self.verbose_mode:
            print("")
            print("GENERATING INITIAL USER MESSAGE:")
            print("  Creating neutral conversation starter...")
        
        neutral_message = PersonaDriftV2Template.generate_neutral_message(
            user_model=user_model,
            conversation_history=[],
            persona_system_prompt=persona_system_prompt
        )
        conversation_history.append({"role": "user", "content": neutral_message})
        
        if self.verbose_mode:
            print(f"  Initial User Message: {neutral_message}")
        
        n_drift = self.turns + 1  # Initialize n_drift to indicate no drift yet
        
        for turn_idx in range(1, self.turns + 1):
            if self.verbose_mode:
                print("")
                print("="*60)
                print(f"TURN {turn_idx}/{self.turns}")
                print("="*60)
            
            # Agent replies to conversation
            if self.verbose_mode:
                print("")
                print("CONVERSATION MESSAGES:")
                conversation_messages = PersonaDriftV2Template.build_conversation_messages(
                    persona_system_prompt=persona_system_prompt,
                    conversation_history=conversation_history,
                    is_baked_in=self.is_baked_in,
                )
                for i, msg in enumerate(conversation_messages):
                    role = msg["role"].upper()
                    content = msg["content"]
                    print(f"  {i+1}. [{role}]: {content}")
            else:
                conversation_messages = PersonaDriftV2Template.build_conversation_messages(
                    persona_system_prompt=persona_system_prompt,
                    conversation_history=conversation_history,
                    is_baked_in=self.is_baked_in,
                )
            
            agent_reply, agent_cost = agent_model.chat_generate(conversation_messages)
            total_cost += agent_cost
            conversation_history.append({"role": "assistant", "content": agent_reply})
            
            if self.verbose_mode:
                print("")
                print("AGENT RESPONSE:")
                print(f"  {agent_reply}")
                print(f"  Cost: ${agent_cost:.4f}")
            
            # Branch probing: test persona adherence (only at specified steps)
            if turn_idx % self.steps == 0:
                probe_question = probe_questions[probe_counter % len(probe_questions)]
                probe_counter += 1  # Increment for next probe turn
            
                if self.verbose_mode:
                    print("")
                    print("BRANCH PROBING:")
                    print(f"  Using Probe #{((probe_counter - 1) % len(probe_questions)) + 1}/10")
                    print(f"  Probe Question: {probe_question}")
                
                probe_messages = PersonaDriftV2Template.build_probe_messages(
                    persona_system_prompt=persona_system_prompt,
                    probe_question=probe_question,
                    conversation_history=conversation_history,
                    is_baked_in=self.is_baked_in,
                )
                
                if self.verbose_mode:
                    print("  Probe Messages:")
                    for i, msg in enumerate(probe_messages):
                        role = msg["role"].upper()
                        content = msg["content"]
                        print(f"    {i+1}. [{role}]: {content}")
                
                # Get agent reply to probe
                probe_reply, probe_cost = agent_model.chat_generate(probe_messages)
                total_cost += probe_cost
                
                if self.verbose_mode:
                    print("  Agent Probe Response:")
                    print(f"    {probe_reply}")
                    print(f"    Cost: ${probe_cost:.4f}")
                
                # G-Eval scoring using proper G-Eval metric
                geval = PersonaDriftV2Template.create_persona_adherence_geval(
                    persona_system_prompt=persona_system_prompt,
                    probe_question=probe_question,
                    model=user_model
                )
                
                # Create test case for G-Eval
                from deepeval.test_case import LLMTestCase
                test_case = LLMTestCase(
                    input=probe_question,
                    actual_output=probe_reply
                )
                
                if self.verbose_mode:
                    print("")
                    print("G-EVAL JUDGING:")
                    print("  Evaluating persona adherence...")
                    print(f"  Persona: {persona_system_prompt}")
                    print(f"  Probe: {probe_question}")
                    print(f"  Response: {probe_reply}")
                
                # Measure persona adherence
                score = geval.measure(test_case, _show_indicator=False)
                probe_scores.append(score)  # Track this probe score
                total_cost += getattr(geval, 'evaluation_cost', 0.0)
                
                # Check for drift (score < 0.5 on 0-1 scale, which is 5.0 on 0-10 scale)
                drift_detected = score < 0.5
                if drift_detected and n_drift > self.turns:
                    n_drift = turn_idx
                
                if self.verbose_mode:
                    print(f"  Score: {score:.1f}/1.0 (equivalent to {score*10:.1f}/10.0)")
                    print(f"  Drift Detected: {'YES' if drift_detected else 'NO'}")
                    print(f"  N_drift: {n_drift if n_drift <= self.turns else 'No drift'}")
                    print(f"  G-Eval Cost: ${getattr(geval, 'evaluation_cost', 0.0):.4f}")
                    
                    # Call the proper print_verbose_logs method for this turn
                    self.print_verbose_logs(
                        turn_idx=turn_idx,
                        probe_question=probe_question,
                        agent_probe_response=probe_reply,
                        score=score,
                        drift_detected=drift_detected
                    )
                
                # Turn summary for probe turns
                if self.verbose_mode:
                    print("")
                    print(f"TURN {turn_idx} SUMMARY:")
                    print(f"  Score: {score:.1f}/1.0 (equivalent to {score*10:.1f}/10.0)")
                    print(f"  Drift: {'DETECTED' if drift_detected else 'NOT DETECTED'}")
                    print(f"  N_drift: {n_drift if n_drift <= self.turns else 'No drift yet'}")
                    print(f"  Turn Cost: ${agent_cost + probe_cost + getattr(geval, 'evaluation_cost', 0.0):.4f}")
                    print(f"  Total Cost: ${total_cost:.4f}")
            else:
                if self.verbose_mode:
                    print("")
                    print("BRANCH PROBING: SKIPPED (not a probe step)")
                    print(f"  Next probe at turn {((turn_idx // self.steps) + 1) * self.steps}")
                
                # Turn summary for non-probe turns
                if self.verbose_mode:
                    print("")
                    print(f"TURN {turn_idx} SUMMARY:")
                    print(f"  Score: N/A (no probe this turn)")
                    print(f"  Drift: NOT CHECKED")
                    print(f"  N_drift: {n_drift if n_drift <= self.turns else 'No drift yet'}")
                    print(f"  Turn Cost: ${agent_cost:.4f}")
                    print(f"  Total Cost: ${total_cost:.4f}")
            
            # Generate next neutral message (builds on context)
            if turn_idx < self.turns:
                if self.verbose_mode:
                    print("")
                    print("GENERATING NEXT USER MESSAGE:")
                
                neutral_message = PersonaDriftV2Template.generate_neutral_message(
                    user_model=user_model,
                    conversation_history=conversation_history,
                    persona_system_prompt=persona_system_prompt
                )
                conversation_history.append({"role": "user", "content": neutral_message})
                
                if self.verbose_mode:
                    print(f"  Next User Message: {neutral_message}")
            
        
        end_time = time.perf_counter()
        total_time_s = end_time - start_time
        
        overall_accuracy = 1.0 if n_drift > self.turns else 0.0  # 1 if no drift, 0 if drift detected
        
        # Calculate average rating across all probed turns
        average_rating = sum(probe_scores) / len(probe_scores) if probe_scores else 0.0
        
        if self.verbose_mode:
            print("")
            print("BENCHMARK COMPLETED")
            print("="*60)
            print("FINAL RESULTS:")
            print(f"  Overall Accuracy: {overall_accuracy:.1%}")
            print(f"  N_drift: {n_drift if n_drift <= self.turns else 'No drift detected'}")
            print(f"  Drift Detected: {'YES' if n_drift <= self.turns else 'NO'}")
            print(f"  Average Rating: {average_rating:.3f}/1.0 (equivalent to {average_rating*10:.1f}/10.0)")
            print(f"  Total Probes: {len(probe_scores)}")
            # Calculate survival rate based on probe turns, not total turns
            if n_drift <= self.turns:
                probe_turns = self.turns // self.steps
                probe_turn_drift = (n_drift - 1) // self.steps + 1
                survival_rate = 1.0 - (probe_turn_drift - 1) / probe_turns
                print(f"  Survival Rate: {survival_rate:.1%} (based on {probe_turns} probe turns)")
            else:
                print(f"  Survival Rate: 100%")
            print(f"  Total Turns: {self.turns}")
            print(f"  Total Time: {total_time_s:.2f} seconds")
            print(f"  Total Cost: ${total_cost:.4f}")
            print("="*60)
            
            if n_drift <= self.turns:
                print("PERSONA DRIFT DETECTED!")
                print(f"  The model lost its persona at turn {n_drift}")
                print(f"  This means the model's responses no longer aligned with the persona")
                print(f"  after {n_drift - 1} successful turns.")
            else:
                print("NO PERSONA DRIFT DETECTED!")
                print(f"  The model maintained its persona throughout all {self.turns} turns.")
                print(f"  This indicates strong persona adherence.")
        
        return PersonaDriftV2Result(
            overall_accuracy=overall_accuracy,
            n_drift=n_drift,
            total_turns=self.turns,
            total_cost=total_cost,
            total_time_s=total_time_s,
            average_rating=average_rating,
            run_metadata={
                "turns": self.turns,
                "steps": self.steps,
                "is_baked_in": self.is_baked_in,
                "persona_system_prompt": persona_system_prompt,
                "agent_model": agent_model.get_model_name() if hasattr(agent_model, 'get_model_name') else str(agent_model),
                "user_model": user_model.get_model_name() if hasattr(user_model, 'get_model_name') else str(user_model),
            },
        )
    
    def load_benchmark_dataset(self, *args, **kwargs) -> List[Golden]:
        """
        Load benchmark dataset. Not used in PersonaDriftV2 as personas are user-defined.
        
        Returns:
            Empty list as personas are provided by users
        """
        return []
    
    def print_verbose_logs(
        self,
        turn_idx: int,
        probe_question: str,
        agent_probe_response: str,
        score: float,
        drift_detected: bool,
    ) -> str:
        """
        Print verbose logs for a single turn of the benchmark.
        
        Args:
            turn_idx: Turn number (1-based)
            probe_question: The probe question asked
            agent_probe_response: Agent's response to the probe
            score: G-Eval score (0-1)
            drift_detected: Whether drift was detected this turn
            
        Returns:
            Formatted verbose log string
        """
        steps = [
            f"Turn {turn_idx} - Probe Question:\n{probe_question}",
            f"Score: {score:.1f}/1.0 (equivalent to {score*10:.1f}/10.0)\nAgent Response: {agent_probe_response}\nDrift Detected: {'YES' if drift_detected else 'NO'}",
        ]
        
        verbose_logs = ""
        for i in range(len(steps) - 1):
            verbose_logs += steps[i]
            if i < len(steps) - 2:
                verbose_logs += " \n \n"
        
        if self.verbose_mode:
            print("*" * 50)
            print(f"Turn {turn_idx} - Persona Adherence Check")
            print("*" * 50)
            print("")
            print(verbose_logs + f"\n \n{steps[-1]}")
            print("")
            print("=" * 70)
        
        return verbose_logs