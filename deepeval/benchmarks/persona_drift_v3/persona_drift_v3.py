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
from deepeval.benchmarks.persona_drift_v3.result import PersonaDriftV3Result, TurnResult


class PersonaDriftV3(DeepEvalBaseBenchmark):
    """
    Persona Drift v3 benchmark for arena-style comparison between baked-in vs system-prompted models.
    
    This benchmark uses parallel synchronized conversations with Arena G-Eval to determine
    which approach better embodies a given persona across every conversation turn.
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
        verbose_mode: bool = False,
        comparison_mode: str = "baked_vs_system",
        baked_model_2: Optional[Union[str, DeepEvalBaseLLM]] = None,
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
            num_turns: Number of conversation turns (each turn will be evaluated)
            verbose_mode: Enable verbose logging
            comparison_mode: "baked_vs_system" or "baked_vs_baked"
            baked_model_2: Second baked model (required for baked_vs_baked mode)
            **kwargs: Additional arguments for base class
        """
        super().__init__(**kwargs)  # Initialize base class

        # Validate required parameters
        if persona_system_prompt is None:
            raise ValueError("persona_system_prompt must be provided")
        if num_turns <= 0:
            raise ValueError("num_turns must be positive")

        # Validate comparison_mode
        if comparison_mode not in ["baked_vs_system", "baked_vs_baked"]:
            raise ValueError("comparison_mode must be 'baked_vs_system' or 'baked_vs_baked'")

        # Mode-specific validation
        if comparison_mode == "baked_vs_system":
            if baked_model is None or system_model is None:
                raise ValueError("Both baked_model and system_model must be provided for baked_vs_system mode")
        elif comparison_mode == "baked_vs_baked":
            if baked_model is None or baked_model_2 is None:
                raise ValueError("Both baked_model and baked_model_2 must be provided for baked_vs_baked mode")

        # Initialize models using helper (following implementation plan)
        self.baked_model, _ = initialize_model(baked_model)
        self.system_model, _ = initialize_model(system_model)
        self.evaluation_model, _ = initialize_model(evaluation_model)
        self.user_model, _ = initialize_model(user_model)

        # Initialize second baked model if needed
        if comparison_mode == "baked_vs_baked":
            self.baked_model_2, _ = initialize_model(baked_model_2)
        else:
            self.baked_model_2 = None
        
        # Store configuration
        self.persona_system_prompt = persona_system_prompt
        self.num_turns = num_turns
        self.verbose_mode = verbose_mode
        self.comparison_mode = comparison_mode

        if self.verbose_mode:
            print("PersonaDriftV3 initialized with:")
            print(f"  Mode: {self.comparison_mode}")
            print(f"  Persona: {self.persona_system_prompt}")
            print(f"  Turns: {self.num_turns} (each turn will be evaluated)")
    
    def evaluate(self, model: DeepEvalBaseLLM = None, **kwargs) -> PersonaDriftV3Result:
        """
        Main evaluation method with telemetry tracking and progress bars.

        Args:
            model: Optional model parameter (for base class compatibility)
            **kwargs: Additional arguments

        Returns:
            PersonaDriftV3Result with arena comparison results
        """
        if self.comparison_mode == "baked_vs_system":
            return self._evaluate_baked_vs_system()
        elif self.comparison_mode == "baked_vs_baked":
            return self._evaluate_baked_vs_baked()
        else:
            raise ValueError(f"Unsupported comparison_mode: {self.comparison_mode}")

    def _evaluate_baked_vs_system(self) -> PersonaDriftV3Result:
        """Evaluate baked model vs system-prompted model."""
        import time

        # Calculate number of evaluation tasks for telemetry (every turn is evaluated)
        num_evaluation_tasks = self.num_turns

        with capture_benchmark_run("PersonaDriftV3", num_evaluation_tasks):
            start_time = time.perf_counter()
            total_cost = 0.0

            if self.verbose_mode:
                print(f"\n{'='*100}")
                print(f"PERSONA DRIFT V3 - BAKED VS SYSTEM EVALUATION")
                print(f"{'='*100}")
                print(f"CONFIGURATION:")
                print(f"  • Mode: Baked vs System-Prompted")
                print(f"  • Turns: {self.num_turns}")
                print(f"  • Evaluation: Every turn (complete coverage)")
                print(f"  • Total Evaluations: {self.num_turns}")
                print(f"{'='*100}")
                print(f"PERSONA SYSTEM PROMPT:")
                print(f"{self.persona_system_prompt}")
                print(f"{'='*100}")

            # Phase 1: Initialize parallel conversation histories
            baked_conversation_history: List[Dict[str, str]] = []
            system_conversation_history: List[Dict[str, str]] = []

            # Phase 2: Create Arena G-Eval metric
            arena_geval = PersonaDriftV3Template.create_persona_arena_geval(
                persona_system_prompt=self.persona_system_prompt,
                evaluation_model=self.evaluation_model
            )

            # Phase 3: Main conversation and evaluation loop
            turn_results = []
            baked_wins = 0
            system_wins = 0

            with tqdm(total=self.num_turns, desc="PersonaDriftV3 Evaluation") as pbar:
                for turn_idx in range(1, self.num_turns + 1):
                    # Update progress bar
                    pbar.set_description(f"Turn {turn_idx}/{self.num_turns} | Evaluations: {len(turn_results)} | Baked: {baked_wins} | System: {system_wins}")

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

                    # Arena G-Eval evaluation on every turn
                    if self.verbose_mode:
                        print(f"\nARENA G-EVAL COMPARISON")
                        print(f"{'='*80}")

                    # Create Arena test case using the turn's user message and responses
                    arena_test_case = PersonaDriftV3Template.create_arena_test_case(
                        user_message=user_message,
                        baked_response=baked_response,
                        system_response=system_response
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
                        print(f"WINNER: {winner.upper().replace('_', ' ')}")
                        if hasattr(arena_geval, 'reason'):
                            print(f"REASONING:")
                            print(f"{arena_geval.reason}")
                        if hasattr(arena_geval, 'evaluation_cost'):
                            print(f"EVALUATION COST: ${arena_geval.evaluation_cost:.6f}")
                        print(f"{'='*80}")
                        print(f"CURRENT SCORE - Baked: {baked_wins} | System: {system_wins}")
                        print(f"{'='*80}\n")

                    # Store turn result
                    turn_results.append(TurnResult(
                        turn_index=turn_idx,
                        user_message=user_message,
                        baked_response=baked_response,
                        system_response=system_response,
                        winner=winner,
                        reasoning=arena_geval.reason if hasattr(arena_geval, 'reason') else "No reasoning provided",
                        evaluation_cost=arena_geval.evaluation_cost if hasattr(arena_geval, 'evaluation_cost') and arena_geval.evaluation_cost is not None else 0.0
                    ))

                    pbar.update(1)

            # Phase 4: Calculate final results
            end_time = time.perf_counter()
            total_time_s = end_time - start_time

            total_evaluations = len(turn_results)
            if total_evaluations > 0:
                baked_model_score = baked_wins / total_evaluations
                system_model_score = system_wins / total_evaluations
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
                print(f"BAKED MODEL:   {baked_model_score:.3f} ({baked_wins}/{total_evaluations} wins)")
                print(f"SYSTEM MODEL:  {system_model_score:.3f} ({system_wins}/{total_evaluations} wins)")
                print(f"OVERALL WINNER: {overall_winner.upper().replace('_', ' ')}")
                print(f"{'='*100}")
                print(f"PERFORMANCE METRICS:")
                print(f"  • Total Cost: ${total_cost:.6f}")
                print(f"  • Total Time: {total_time_s:.2f} seconds")
                print(f"  • Total Evaluations: {total_evaluations}")
                print(f"  • Total Turns: {self.num_turns}")
                print(f"{'='*100}\n")

            return PersonaDriftV3Result(
                baked_model_score=baked_model_score,
                system_model_score=system_model_score,
                overall_winner=overall_winner,
                total_turns_evaluated=total_evaluations,
                total_turns=self.num_turns,
                total_cost=total_cost,
                total_time_s=total_time_s,
                turn_results=turn_results,
                run_metadata={
                    "num_turns": self.num_turns,
                    "persona_system_prompt": self.persona_system_prompt,
                    "total_evaluations": total_evaluations,
                    "baked_wins": baked_wins,
                    "system_wins": system_wins,
                    "comparison_mode": "baked_vs_system",
                    "phase": "Phase 4b - Full Arena Evaluation Implementation"
                }
            )

    def _evaluate_baked_vs_baked(self) -> PersonaDriftV3Result:
        """Evaluate first baked model vs second baked model."""
        import time

        # Calculate number of evaluation tasks for telemetry (every turn is evaluated)
        num_evaluation_tasks = self.num_turns

        with capture_benchmark_run("PersonaDriftV3", num_evaluation_tasks):
            start_time = time.perf_counter()
            total_cost = 0.0

            if self.verbose_mode:
                print(f"\n{'='*100}")
                print(f"PERSONA DRIFT V3 - BAKED VS BAKED EVALUATION")
                print(f"{'='*100}")
                print(f"CONFIGURATION:")
                print(f"  • Mode: Baked Model 1 vs Baked Model 2")
                print(f"  • Turns: {self.num_turns}")
                print(f"  • Evaluation: Every turn (complete coverage)")
                print(f"  • Total Evaluations: {self.num_turns}")
                print(f"{'='*100}")
                print(f"PERSONA SYSTEM PROMPT (for evaluation context only):")
                print(f"{self.persona_system_prompt}")
                print(f"{'='*100}")

            # Phase 1: Initialize parallel conversation histories
            baked_1_conversation_history: List[Dict[str, str]] = []
            baked_2_conversation_history: List[Dict[str, str]] = []

            # Phase 2: Create Arena G-Eval metric
            arena_geval = PersonaDriftV3Template.create_persona_arena_geval(
                persona_system_prompt=self.persona_system_prompt,
                evaluation_model=self.evaluation_model
            )

            # Phase 3: Main conversation and evaluation loop
            turn_results = []
            model_1_wins = 0
            model_2_wins = 0

            with tqdm(total=self.num_turns, desc="PersonaDriftV3 Evaluation") as pbar:
                for turn_idx in range(1, self.num_turns + 1):
                    # Update progress bar
                    pbar.set_description(f"Turn {turn_idx}/{self.num_turns} | Evaluations: {len(turn_results)} | Model1: {model_1_wins} | Model2: {model_2_wins}")

                    # Generate neutral user message for this turn
                    user_message = PersonaDriftV3Template.generate_neutral_message_dual_baked(
                        user_model=self.user_model,
                        baked_1_conversation_history=baked_1_conversation_history,
                        baked_2_conversation_history=baked_2_conversation_history,
                        persona_system_prompt=self.persona_system_prompt
                    )

                    if self.verbose_mode:
                        print(f"\n{'='*80}")
                        print(f"TURN {turn_idx}")
                        print(f"{'='*80}")
                        print(f"USER: {user_message}")
                        print(f"{'-'*80}")

                    # Add user message to both conversation histories
                    baked_1_conversation_history.append({"role": "user", "content": user_message})
                    baked_2_conversation_history.append({"role": "user", "content": user_message})

                    # Get response from first baked model (NO system prompt)
                    baked_1_messages = PersonaDriftV3Template.build_baked_messages_1(baked_1_conversation_history)
                    baked_1_response, baked_1_cost = self.baked_model.chat_generate(baked_1_messages)
                    total_cost += baked_1_cost

                    # Get response from second baked model (NO system prompt)
                    baked_2_messages = PersonaDriftV3Template.build_baked_messages_2(baked_2_conversation_history)
                    baked_2_response, baked_2_cost = self.baked_model_2.chat_generate(baked_2_messages)
                    total_cost += baked_2_cost

                    # Add responses to respective conversation histories
                    baked_1_conversation_history.append({"role": "assistant", "content": baked_1_response})
                    baked_2_conversation_history.append({"role": "assistant", "content": baked_2_response})

                    if self.verbose_mode:
                        print(f"BAKED MODEL 1: {baked_1_response}")
                        print(f"{'-'*40}")
                        print(f"BAKED MODEL 2: {baked_2_response}")
                        print(f"{'-'*80}")

                    # Arena G-Eval evaluation on every turn
                    if self.verbose_mode:
                        print(f"\nARENA G-EVAL COMPARISON")
                        print(f"{'='*80}")

                    # Create Arena test case for dual baked models
                    arena_test_case = PersonaDriftV3Template.create_arena_test_case_dual_baked(
                        user_message=user_message,
                        baked_1_response=baked_1_response,
                        baked_2_response=baked_2_response
                    )

                    # Run Arena G-Eval
                    winner = arena_geval.measure(arena_test_case)

                    # Get evaluation cost from the metric object
                    if hasattr(arena_geval, 'evaluation_cost') and arena_geval.evaluation_cost is not None:
                        total_cost += arena_geval.evaluation_cost

                    # Track winner (map to model_1/model_2)
                    if winner == "model_1":
                        model_1_wins += 1
                    else:
                        model_2_wins += 1

                    if self.verbose_mode:
                        print(f"WINNER: {winner.upper().replace('_', ' ')}")
                        if hasattr(arena_geval, 'reason'):
                            print(f"REASONING:")
                            print(f"{arena_geval.reason}")
                        if hasattr(arena_geval, 'evaluation_cost'):
                            print(f"EVALUATION COST: ${arena_geval.evaluation_cost:.6f}")
                        print(f"{'='*80}")
                        print(f"CURRENT SCORE - Model1: {model_1_wins} | Model2: {model_2_wins}")
                        print(f"{'='*80}\n")

                    # Store turn result (using generic field names for dual baked mode)
                    turn_results.append(TurnResult(
                        turn_index=turn_idx,
                        user_message=user_message,
                        baked_response=baked_1_response,  # Model 1 response
                        system_response=baked_2_response,  # Model 2 response (reusing field)
                        winner=winner,
                        reasoning=arena_geval.reason if hasattr(arena_geval, 'reason') else "No reasoning provided",
                        evaluation_cost=arena_geval.evaluation_cost if hasattr(arena_geval, 'evaluation_cost') and arena_geval.evaluation_cost is not None else 0.0
                    ))

                    pbar.update(1)

            # Phase 4: Calculate final results
            end_time = time.perf_counter()
            total_time_s = end_time - start_time

            total_evaluations = len(turn_results)
            if total_evaluations > 0:
                model_1_score = model_1_wins / total_evaluations
                model_2_score = model_2_wins / total_evaluations
            else:
                model_1_score = 0.5
                model_2_score = 0.5

            # Determine overall winner
            if model_1_score > model_2_score:
                overall_winner = "model_1"
            elif model_2_score > model_1_score:
                overall_winner = "model_2"
            else:
                overall_winner = "model_1"  # Tie goes to model 1

            if self.verbose_mode:
                print(f"\n{'='*100}")
                print(f"FINAL ARENA RESULTS")
                print(f"{'='*100}")
                print(f"BAKED MODEL 1: {model_1_score:.3f} ({model_1_wins}/{total_evaluations} wins)")
                print(f"BAKED MODEL 2: {model_2_score:.3f} ({model_2_wins}/{total_evaluations} wins)")
                print(f"OVERALL WINNER: {overall_winner.upper().replace('_', ' ')}")
                print(f"{'='*100}")
                print(f"PERFORMANCE METRICS:")
                print(f"  • Total Cost: ${total_cost:.6f}")
                print(f"  • Total Time: {total_time_s:.2f} seconds")
                print(f"  • Total Evaluations: {total_evaluations}")
                print(f"  • Total Turns: {self.num_turns}")
                print(f"{'='*100}\n")

            return PersonaDriftV3Result(
                baked_model_score=model_1_score,      # Map model_1 to baked_model_score
                system_model_score=model_2_score,     # Map model_2 to system_model_score
                overall_winner=overall_winner,
                total_turns_evaluated=total_evaluations,
                total_turns=self.num_turns,
                total_cost=total_cost,
                total_time_s=total_time_s,
                turn_results=turn_results,
                run_metadata={
                    "num_turns": self.num_turns,
                    "persona_system_prompt": self.persona_system_prompt,
                    "total_evaluations": total_evaluations,
                    "model_1_wins": model_1_wins,
                    "model_2_wins": model_2_wins,
                    "comparison_mode": "baked_vs_baked",
                    "phase": "Phase 4b - Full Arena Evaluation Implementation"
                }
            )

    def load_benchmark_dataset(self, *args, **kwargs) -> List[Golden]:
        """
        Required by DeepEvalBaseBenchmark. Returns empty list since personas are user-provided.
        
        Returns:
            Empty list as personas are provided by users directly
        """
        return []
