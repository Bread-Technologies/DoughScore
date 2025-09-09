from typing import List, Optional, Dict, Union
from tqdm import tqdm

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import (
    DeepEvalBaseBenchmark,
    DeepEvalBaseBenchmarkResult,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import ArenaGEval
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    ArenaTestCase
)
from deepeval.telemetry import capture_benchmark_run
from deepeval.metrics.utils import initialize_model

from deepeval.benchmarks.persona_drift_v3.template import PersonaDriftV3Template
from deepeval.benchmarks.persona_drift_v3.result import PersonaDriftV3Result, ProbeResult


class PersonaDriftV3(DeepEvalBaseBenchmark):
    """
    Persona Drift v3 benchmark for arena-style comparison between baked-in vs system-prompted models.
    
    This benchmark uses parallel synchronized conversations with Arena G-Eval to determine
    which approach better embodies a given persona across multiple probe comparisons.
    Extends DeepEvalBaseBenchmark following existing benchmark patterns.
    """
    
    def __init__(
        self,
        baked_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        system_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        evaluation_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        user_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        persona_system_prompt: str = None,
        num_turns: int = 10,
        probe_frequency: int = 2,
        verbose_mode: bool = False,
        **kwargs
    ):
        """
        Initialize PersonaDriftV3 benchmark.
        
        Args:
            baked_model: Model with persona baked into weights (no system prompt)
            system_model: Standard model that receives persona via system prompt
            evaluation_model: Arena G-Eval judge model
            user_model: Model for generating neutral user messages
            persona_system_prompt: The persona system prompt to test
            num_turns: Number of conversation turns
            probe_frequency: Probe every N turns
            verbose_mode: Enable verbose logging
            **kwargs: Additional arguments for base class
        """
        super().__init__(**kwargs)  # Initialize base class
        
        # Validate required parameters
        if persona_system_prompt is None:
            raise ValueError("persona_system_prompt must be provided")
        if num_turns <= 0:
            raise ValueError("num_turns must be positive")
        if probe_frequency <= 0:
            raise ValueError("probe_frequency must be positive")
        if probe_frequency > num_turns:
            raise ValueError("probe_frequency cannot be greater than num_turns")
        
        # Initialize models using helper (following implementation plan)
        self.baked_model, _ = initialize_model(baked_model)
        self.system_model, _ = initialize_model(system_model)
        self.evaluation_model, _ = initialize_model(evaluation_model)
        self.user_model, _ = initialize_model(user_model)
        
        # Store configuration
        self.persona_system_prompt = persona_system_prompt
        self.num_turns = num_turns
        self.probe_frequency = probe_frequency
        self.verbose_mode = verbose_mode
        
        if self.verbose_mode:
            print("PersonaDriftV3 initialized with:")
            print(f"  Persona: {self.persona_system_prompt}")
            print(f"  Turns: {self.num_turns}")
            print(f"  Probe Frequency: {self.probe_frequency}")
    
    def evaluate(self, model: DeepEvalBaseLLM = None, **kwargs) -> PersonaDriftV3Result:
        """
        Main evaluation method with telemetry tracking and progress bars.
        
        Args:
            model: Optional model parameter (for base class compatibility)
            **kwargs: Additional arguments
            
        Returns:
            PersonaDriftV3Result with arena comparison results
        """
        import time
        
        # Calculate number of probe tasks for telemetry
        num_probe_tasks = self.num_turns // self.probe_frequency
        
        with capture_benchmark_run("PersonaDriftV3", num_probe_tasks):
            start_time = time.perf_counter()
            total_cost = 0.0
            
            if self.verbose_mode:
                print(f"\n{'='*100}")
                print(f"PERSONA DRIFT V3 - ARENA EVALUATION")
                print(f"{'='*100}")
                print(f"CONFIGURATION:")
                print(f"  • Turns: {self.num_turns}")
                print(f"  • Probe Frequency: Every {self.probe_frequency} turns")
                print(f"  • Expected Probes: {self.num_turns // self.probe_frequency}")
                print(f"{'='*100}")
                print(f"PERSONA SYSTEM PROMPT:")
                print(f"{self.persona_system_prompt}")
                print(f"{'='*100}")
            
            # Phase 1: Generate probe questions at start
            if self.verbose_mode:
                print(f"\nGENERATING PROBE QUESTIONS...")
            
            probe_questions = PersonaDriftV3Template.generate_probe_questions(
                user_model=self.user_model,
                persona_system_prompt=self.persona_system_prompt,
                num_probes=10  # Generate 10 probes to rotate through
            )
            
            if self.verbose_mode:
                print(f"Generated {len(probe_questions)} probe questions")
                print(f"{'='*100}\n")
            
            # Phase 2: Initialize parallel conversation histories
            baked_conversation_history: List[Dict[str, str]] = []
            system_conversation_history: List[Dict[str, str]] = []
            
            # Phase 3: Create Arena G-Eval metric
            arena_geval = PersonaDriftV3Template.create_persona_arena_geval(
                persona_system_prompt=self.persona_system_prompt,
                evaluation_model=self.evaluation_model
            )
            
            # Phase 4: Main conversation and probing loop
            probe_results = []
            baked_wins = 0
            system_wins = 0
            probe_counter = 0
            
            with tqdm(total=self.num_turns, desc="PersonaDriftV3 Evaluation") as pbar:
                for turn_idx in range(1, self.num_turns + 1):
                    # Update progress bar
                    pbar.set_description(f"Turn {turn_idx}/{self.num_turns} | Probes: {len(probe_results)} | Baked: {baked_wins} | System: {system_wins}")
                    
                    # Generate neutral user message for this turn
                    user_message = PersonaDriftV3Template.generate_neutral_message(
                        user_model=self.user_model,
                        baked_conversation_history=baked_conversation_history,
                        system_conversation_history=system_conversation_history,
                        persona_system_prompt=self.persona_system_prompt
                    )
                    
                    if self.verbose_mode:
                        print(f"\n{'='*80}")
                        print(f"TURN {turn_idx}")
                        print(f"{'='*80}")
                        print(f"USER: {user_message}")
                        print(f"{'-'*80}")
                    
                    # Add user message to both conversation histories
                    baked_conversation_history.append({"role": "user", "content": user_message})
                    system_conversation_history.append({"role": "user", "content": user_message})
                    
                    # Get response from baked-in model (NO system prompt)
                    baked_messages = PersonaDriftV3Template.build_baked_messages(baked_conversation_history)
                    baked_response, baked_cost = self.baked_model.chat_generate(baked_messages)
                    total_cost += baked_cost
                    
                    # Get response from system-prompted model (WITH persona system prompt)
                    system_messages = PersonaDriftV3Template.build_system_messages(
                        system_conversation_history, self.persona_system_prompt
                    )
                    system_response, system_cost = self.system_model.chat_generate(system_messages)
                    total_cost += system_cost
                    
                    # Add responses to respective conversation histories
                    baked_conversation_history.append({"role": "assistant", "content": baked_response})
                    system_conversation_history.append({"role": "assistant", "content": system_response})
                    
                    if self.verbose_mode:
                        print(f"BAKED MODEL: {baked_response}")
                        print(f"{'-'*40}")
                        print(f"SYSTEM MODEL: {system_response}")
                        print(f"{'-'*80}")
                    
                    # Check if this is a probe turn
                    if turn_idx % self.probe_frequency == 0:
                        probe_question = probe_questions[probe_counter % len(probe_questions)]
                        probe_counter += 1
                        
                        if self.verbose_mode:
                            print(f"\nPROBE QUESTION")
                            print(f"{'='*80}")
                            print(f"{probe_question}")
                            print(f"{'='*80}")
                        
                        # Get probe responses from both models
                        baked_probe_messages = PersonaDriftV3Template.build_baked_messages(
                            baked_conversation_history + [{"role": "user", "content": probe_question}]
                        )
                        baked_probe_response, baked_probe_cost = self.baked_model.chat_generate(baked_probe_messages)
                        total_cost += baked_probe_cost
                        
                        system_probe_messages = PersonaDriftV3Template.build_system_messages(
                            system_conversation_history + [{"role": "user", "content": probe_question}],
                            self.persona_system_prompt
                        )
                        system_probe_response, system_probe_cost = self.system_model.chat_generate(system_probe_messages)
                        total_cost += system_probe_cost
                        
                        if self.verbose_mode:
                            print(f"BAKED MODEL RESPONSE:")
                            print(f"{baked_probe_response}")
                            print(f"{'-'*40}")
                            print(f"SYSTEM MODEL RESPONSE:")
                            print(f"{system_probe_response}")
                            print(f"{'='*80}")
                        
                        # Create Arena test case and evaluate
                        arena_test_case = PersonaDriftV3Template.create_probe_arena_test_case(
                            probe_question=probe_question,
                            baked_response=baked_probe_response,
                            system_response=system_probe_response
                        )
                        
                        # Run Arena G-Eval
                        winner = arena_geval.measure(arena_test_case)
                        
                        # Get evaluation cost from the metric object
                        if hasattr(arena_geval, 'evaluation_cost') and arena_geval.evaluation_cost is not None:
                            total_cost += arena_geval.evaluation_cost
                        
                        # Track winner
                        if winner == "baked_model":
                            baked_wins += 1
                        else:
                            system_wins += 1
                        
                        if self.verbose_mode:
                            print(f"\nARENA G-EVAL RESULTS:")
                            print(f"WINNER: {winner.upper().replace('_', ' ')}")
                            if hasattr(arena_geval, 'reason'):
                                print(f"REASONING:")
                                print(f"{arena_geval.reason}")
                            if hasattr(arena_geval, 'evaluation_cost'):
                                print(f"EVALUATION COST: ${arena_geval.evaluation_cost:.6f}")
                            print(f"{'='*80}")
                            print(f"CURRENT SCORE - Baked: {baked_wins} | System: {system_wins}")
                            print(f"{'='*80}\n")
                        
                        # Store probe result
                        probe_results.append(ProbeResult(
                            turn_index=turn_idx,
                            probe_question=probe_question,
                            baked_response=baked_probe_response,
                            system_response=system_probe_response,
                            winner=winner,
                            reasoning=arena_geval.reason if hasattr(arena_geval, 'reason') else "No reasoning provided",
                            evaluation_cost=arena_geval.evaluation_cost if hasattr(arena_geval, 'evaluation_cost') and arena_geval.evaluation_cost is not None else 0.0
                        ))
                    
                    pbar.update(1)
            
            # Phase 5: Calculate final results
            end_time = time.perf_counter()
            total_time_s = end_time - start_time
            
            total_probes = len(probe_results)
            if total_probes > 0:
                baked_model_score = baked_wins / total_probes
                system_model_score = system_wins / total_probes
            else:
                baked_model_score = 0.5
                system_model_score = 0.5
            
            # Determine overall winner
            if baked_model_score > system_model_score:
                overall_winner = "baked_model"
            elif system_model_score > baked_model_score:
                overall_winner = "system_model"
            else:
                overall_winner = "baked_model"  # Tie goes to baked model
            
            if self.verbose_mode:
                print(f"\n{'='*100}")
                print(f"FINAL ARENA RESULTS")
                print(f"{'='*100}")
                print(f"BAKED MODEL:   {baked_model_score:.3f} ({baked_wins}/{total_probes} wins)")
                print(f"SYSTEM MODEL:  {system_model_score:.3f} ({system_wins}/{total_probes} wins)")
                print(f"OVERALL WINNER: {overall_winner.upper().replace('_', ' ')}")
                print(f"{'='*100}")
                print(f"PERFORMANCE METRICS:")
                print(f"  • Total Cost: ${total_cost:.6f}")
                print(f"  • Total Time: {total_time_s:.2f} seconds")
                print(f"  • Total Probes: {total_probes}")
                print(f"  • Total Turns: {self.num_turns}")
                print(f"{'='*100}\n")
            
            return PersonaDriftV3Result(
                baked_model_score=baked_model_score,
                system_model_score=system_model_score,
                overall_winner=overall_winner,
                total_probes=total_probes,
                total_turns=self.num_turns,
                total_cost=total_cost,
                total_time_s=total_time_s,
                probe_results=probe_results,
                run_metadata={
                    "num_turns": self.num_turns,
                    "probe_frequency": self.probe_frequency,
                    "persona_system_prompt": self.persona_system_prompt,
                    "total_probes": total_probes,
                    "baked_wins": baked_wins,
                    "system_wins": system_wins,
                    "phase": "Phase 4 - Complete Implementation"
                }
            )
    
    def load_benchmark_dataset(self, *args, **kwargs) -> List[Golden]:
        """
        Required by DeepEvalBaseBenchmark. Returns empty list since personas are user-provided.
        
        Returns:
            Empty list as personas are provided by users directly
        """
        return []
