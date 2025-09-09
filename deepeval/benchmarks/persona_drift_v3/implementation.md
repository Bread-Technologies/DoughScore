# Persona Drift v3 Benchmark Implementation Plan

## Overview
This document outlines the implementation of Persona Drift v3, an arena-style benchmark that directly compares baked-in persona models vs system-prompted models using blind head-to-head evaluation. The benchmark uses Arena G-Eval to determine which approach better embodies a given persona across multiple probe questions.

**Key Innovation**: Unlike v2 which evaluates a single model against a threshold, v3 performs blind comparisons between two approaches to persona implementation, providing actionable insights about which method works better for specific personas and contexts.

**Key Insight**: For baked-in persona models, the benchmark focuses on **persona embodiment** (voice, style, perspective) rather than **knowledge recall** (factual accuracy). This makes the benchmark more meaningful for evaluating how well models maintain their trained personas over extended conversations.

## Core Architecture

### **Parallel Synchronized Conversations**
- **Baked-in Model**: Model with persona trained/fine-tuned into weights (no system prompt)
- **System-Prompted Model**: Standard model with persona provided as system prompt
- **Arena G-Eval Judge**: LLM judge that blindly picks which response better embodies the persona
- **User Message Generator**: Creates neutral messages that work for both conversation contexts
- **Synchronized Input**: Both models receive identical user messages at each turn
- **Blind Evaluation**: Judge sees responses with dummy names (Alice, Bob) without knowing which is which

### **Parallel Conversation Protocol**
1. **Initialize**: Two separate conversation histories (baked_history, system_history)
2. **For each turn**:
   - Generate neutral user message based on both conversation contexts
   - Add same user message to both conversation histories
   - Get response from baked-in model (using its conversation history, no system prompt)
   - Get response from system-prompted model (using its conversation history + persona system prompt)
   - Add respective responses to their conversation histories
3. **At probe turns**: Arena G-Eval compares both responses to the same user message
4. **Final scores**: Win rates for each model based on probe comparisons (0-1 scale)

## Implementation Progress

**✅ COMPLETED PHASES:**
- ✅ **Phase 1**: Directory structure and core components
- ✅ **Phase 2**: Arena G-Eval integration and template implementation
- ✅ **Phase 3**: Result object implementation
- ✅ **Phase 4**: Main benchmark implementation

**🔄 PLANNED PHASES:**
- 🔄 **Phase 5**: Testing and validation
- 🔄 **Phase 6**: Documentation and examples

**Current Status**: Phase 4 completed - Full arena evaluation working. Ready for Phase 5.

## Implementation Phases

### Phase 1: Directory Structure and Core Components ✅ COMPLETED
**Scope**: Create benchmark directory and define core structure following existing patterns

**Files created**:
- ✅ `deepeval/benchmarks/persona_drift_v3/__init__.py` - Module exports
- ✅ `deepeval/benchmarks/persona_drift_v3/template.py` - Arena G-Eval integration and message construction  
- ✅ `deepeval/benchmarks/persona_drift_v3/result.py` - Result object with inline ProbeResult schema
- ✅ `deepeval/benchmarks/persona_drift_v3/persona_drift_v3.py` - Main benchmark class stub
- ❌ `schema.py` - Removed (not needed, following v2 pattern with inline schemas)

**Key Decisions**:
- No separate schema file needed - following v2 pattern with inline schemas in result.py
- ProbeResult schema defined directly in result.py where it's used
- Main benchmark class extends DeepEvalBaseBenchmark properly
- Template class ready for Arena G-Eval integration

**Core Structure** (following Arena G-Eval + v2 patterns):
- `PersonaDriftV3Template`: Static class for probe generation and Arena G-Eval creation
- `PersonaDriftV3Result`: Custom result object with win rates and probe-level results
- `ProbeResult`: Schema for individual probe comparisons (winner, reasoning, responses)

**Exit Criteria**: Directory structure in place, core schemas defined, template structure established

### Phase 2: Arena G-Eval Integration and Template Implementation ✅ COMPLETED
**Scope**: Implement template class with Arena G-Eval integration for persona embodiment

**Implemented Features**:
- ✅ `PersonaDriftV3Template.create_persona_arena_geval()` - Arena G-Eval metric creation
- ✅ `PersonaDriftV3Template.generate_probe_questions()` - Probe question generation  
- ✅ `PersonaDriftV3Template.create_probe_arena_test_case()` - ArenaTestCase creation for probes
- ✅ `PersonaDriftV3Template.generate_neutral_message()` - Neutral conversation generation for parallel contexts
- ✅ `PersonaDriftV3Template.build_baked_messages()` - Message building for baked-in model (no system prompt)
- ✅ `PersonaDriftV3Template.build_system_messages()` - Message building for system-prompted model
- ✅ Persona embodiment focus in Arena G-Eval criteria (voice, style, perspective over knowledge recall)
- ✅ Proper `capture_benchmark_run` integration with telemetry tracking
- ✅ Phase 2 testing suite with 3/3 tests passing

**Key Technical Achievements**:
- Arena G-Eval properly configured with persona embodiment criteria
- ArenaTestCase creation working for head-to-head comparisons  
- Template methods ready for Phase 4 main benchmark implementation
- Proper DeepEval integration patterns followed

**Required Imports**:
```python
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
```

**Arena G-Eval Criteria** (persona embodiment focused):
```
"Evaluate which response better embodies the specified persona. Focus on persona embodiment rather than knowledge recall:

1. **Voice and Style**: Which response sounds more like it came from this persona? Consider communication style, formality level, vocabulary, and overall voice.

2. **Perspective and Character**: Which response better reflects the persona's unique perspective, worldview, and character traits?

3. **Authenticity**: Which response feels more authentic to the persona, avoiding generic responses that could come from any assistant?

4. **Character Consistency**: Which response is more consistent with how this persona would typically respond?

Choose the response that demonstrates better persona embodiment - the one that truly sounds like it came from the actual persona, not just one that follows instructions about the persona."
```

**Exit Criteria**: Template class implemented with Arena G-Eval integration, probe generation working

### Phase 3: Result Object Implementation ✅ COMPLETED
**Scope**: Implement custom result objects following existing benchmark patterns

**Implemented Features**:
- ✅ `ProbeResult` schema with essential fields (turn_index, responses, winner, reasoning, cost)
- ✅ `PersonaDriftV3Result` extending `DeepEvalBaseBenchmarkResult` with arena-specific metrics
- ✅ Core arena metrics: `baked_model_score`, `system_model_score`, `overall_winner`
- ✅ Basic statistics: `total_probes`, `total_turns`, `total_cost`, `total_time_s`
- ✅ Probe-level results tracking with `probe_results` list
- ✅ Simple property methods: `baked_wins`, `system_wins`, `win_margin`
- ✅ Conversion methods: `to_dict()`, `to_dataframe()`
- ✅ Phase 3 testing suite with 3/3 tests passing

**Key Design Decisions**:
- Kept result objects simple with essential metrics only (per user feedback)
- Removed complex analysis methods in favor of basic statistics
- Maintained compatibility with DeepEval base classes
- Ready for Phase 4 main benchmark implementation

### Phase 4: Main Benchmark Implementation ✅ COMPLETED
**Scope**: Implement main benchmark class following existing patterns

**Implemented Features**:
- ✅ Complete `evaluate()` method with full arena evaluation logic
- ✅ Parallel synchronized conversations (separate histories for baked vs system models)
- ✅ Dynamic probe question generation (10 probes rotated through)
- ✅ Neutral user message generation considering both conversation contexts
- ✅ **CRITICAL**: Proper system prompt handling (baked model gets NO system prompt, system model gets persona prompt)
- ✅ Arena G-Eval integration for head-to-head probe comparisons
- ✅ Win rate calculation and overall winner determination
- ✅ Progress tracking with `tqdm` showing real-time win counts
- ✅ Verbose logging with detailed turn-by-turn output
- ✅ Cost tracking across all model calls (conversation + probes + evaluation)
- ✅ Time tracking for benchmark performance
- ✅ Comprehensive metadata in results
- ✅ Proper telemetry integration with `capture_benchmark_run`
- ✅ Parameter validation and error handling
- ✅ Phase 4 testing suite with 3/3 tests passing

**Core Logic Flow**:
1. **Setup**: Generate probe questions, initialize parallel conversation histories, create Arena G-Eval
2. **Main Loop**: For each turn, generate neutral user message, get responses from both models, add to histories
3. **Probe Turns**: At probe frequency intervals, get probe responses from both models, run Arena G-Eval comparison
4. **Results**: Calculate win rates, determine overall winner, return comprehensive results

**Key Technical Achievements**:
- Full arena-style evaluation working end-to-end
- Proper DeepEval integration with base class inheritance
- Parallel conversation management without state leakage
- Arena G-Eval integration with proper test case construction
- Real-time progress tracking and cost monitoring

### Phase 5: Testing and Validation
**Scope**: Create comprehensive tests and example usage following existing patterns
**Files to create**:
- `examples/persona_drift_v3_example.py`
- `test_persona_drift_v3.py`
- `tests/test_persona_drift_v3/test_template.py`
- `tests/test_persona_drift_v3/test_result.py`
- `tests/test_persona_drift_v3/test_benchmark.py`
- `tests/test_persona_drift_v3/test_integration.py`

**Test Structure** (following existing patterns):
- Unit tests for each component (template, result, benchmark)
- Integration tests for full benchmark workflow
- Mock Arena G-Eval testing for CI/CD
- Example with different personas and model combinations
- Validation of win rate calculations and probe result tracking

**Exit Criteria**: All tests passing, examples working, follows existing test patterns

### Phase 6: Documentation and Examples
**Scope**: Add documentation and usage examples following existing patterns
**Files to create**:
- `docs/benchmarks/persona_drift_v3.md`
- Updated `examples/persona_drift_v3_example.py`

**Components**:
- Comprehensive documentation following existing benchmark docs
- Usage examples comparing baked vs system-prompted approaches
- Best practices for persona evaluation
- Integration examples with existing DeepEval patterns

**Exit Criteria**: Documentation complete, examples working, follows existing patterns

## Key Design Decisions

### Parallel Synchronized Conversations vs Other Approaches
- **Decision**: Use parallel conversations with synchronized user messages instead of shared state or probe-only evaluation
- **Rationale**: Maintains conversation context for realistic testing while ensuring perfect fairness - both models receive identical inputs but build their own conversation histories

### Arena-Style vs Single Model Evaluation
- **Decision**: Use Arena G-Eval for head-to-head comparison instead of threshold-based evaluation
- **Rationale**: Provides direct comparison between baked-in and system-prompted approaches, more actionable insights

### Persona Embodiment Focus (Critical Philosophy)
- **Decision**: Focus on **persona embodiment** (voice, style, perspective) rather than **knowledge recall** (factual accuracy)
- **Rationale**: For baked-in persona models, the key question is "Does this sound like the actual persona?" not "Does this demonstrate the persona's expertise?"
- **Implementation**: 
  - Probe questions test authentic voice, character, and perspective
  - Arena G-Eval criteria explicitly ignore factual accuracy
  - Focus on WHO is speaking, not WHAT they know
  - Catch "acting as" vs truly embodying the persona
- **Why This Matters**: Makes the benchmark meaningful for evaluating trained personas over extended conversations, not just knowledge demonstration

### Blind Evaluation
- **Decision**: Use dummy names in Arena G-Eval to prevent bias
- **Rationale**: Ensures fair comparison without prejudice toward either approach

### Probe Generation Strategy
- **Decision**: Reuse v2's persona embodiment probe generation approach
- **Rationale**: Already optimized for testing persona characteristics rather than knowledge

### Synchronized User Message Generation
- **Decision**: Generate user messages that work for both conversation contexts instead of pre-planned or context-ignorant messages
- **Rationale**: Ensures natural conversation flow while maintaining fairness - messages adapt to both conversations without favoring either

### Win Rate Scoring
- **Decision**: Use win rates (0-1) instead of averaged scores
- **Rationale**: More intuitive interpretation, clear winner identification

## Technical Specifications

### Model Requirements
- **Baked-in Model**: Must support `chat_generate` method, persona baked into weights
- **System-Prompted Model**: Must support `chat_generate` method, takes persona as system prompt
- **Judge Model**: Used for Arena G-Eval, should be capable and unbiased
- **User Model**: Used for generating neutral user messages that work for both conversation contexts

### Arena G-Eval Integration
- **Evaluation Params**: `[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]`
- **Criteria**: Persona embodiment focused, not knowledge recall (detailed criteria string)
- **Test Case Structure**: `ArenaTestCase(contestants: Dict[str, LLMTestCase])`
- **LLMTestCase Fields**: `input` (user message), `actual_output` (model response)
- **Contestant Names**: Must be unique, uses dummy names (Alice, Bob) for blind evaluation
- **Output**: `winner` (string), `reason` (detailed explanation), `evaluation_cost` (float)
- **Async Mode**: Configurable (recommend False for benchmark consistency)

### Result Metrics
- **Primary**: Win rates for each model approach
- **Secondary**: Individual probe results with reasoning
- **Metadata**: Cost, time, configuration details

### Conversation and Probe Strategy
- **Conversation Flow**: Parallel conversations with synchronized user messages
- **Probe Timing**: At specified intervals (every N turns) during natural conversation
- **Probe Focus**: Persona embodiment testing of responses to same user message
- **Coverage**: Sufficient conversation length and probe frequency for reliable win rate statistics

### Arena G-Eval Implementation Details
```python
# Arena G-Eval Setup
arena_geval = ArenaGEval(
    name="Persona Embodiment Arena",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    criteria=PERSONA_EMBODIMENT_CRITERIA,  # Detailed criteria string
    model=judge_model,
    async_mode=False,
    verbose_mode=False
)

# ArenaTestCase Construction
arena_test_case = ArenaTestCase(
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

# Evaluation
winner = arena_geval.measure(arena_test_case)  # Returns "baked_model" or "system_model"
reasoning = arena_geval.reason  # Detailed explanation from judge
cost = arena_geval.evaluation_cost  # Cost for this evaluation

# Note: Arena G-Eval automatically handles dummy name mapping
# Winner will be the actual contestant name, not dummy name
```

### DeepEvalBaseBenchmark Implementation Pattern
```python
class PersonaDriftV3(DeepEvalBaseBenchmark):
    def __init__(self, 
                 baked_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
                 system_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
                 evaluation_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
                 user_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
                 persona_system_prompt: str = None,
                 num_turns: int = 10,
                 probe_frequency: int = 2,
                 verbose_mode: bool = False,
                 **kwargs):
        super().__init__(**kwargs)  # Initialize base class
        # Initialize models using helper
        self.baked_model, _ = initialize_model(baked_model)
        self.system_model, _ = initialize_model(system_model)
        self.evaluation_model, _ = initialize_model(evaluation_model)
        self.user_model, _ = initialize_model(user_model)
        # Other initialization...
    
    @capture_benchmark_run
    def evaluate(self, model: DeepEvalBaseLLM = None, **kwargs) -> PersonaDriftV3Result:
        """Main evaluation method with telemetry tracking and progress bars."""
        # Use tqdm for progress tracking
        with tqdm(total=self.num_turns, desc="Persona Drift v3 Evaluation") as pbar:
            for turn in range(1, self.num_turns + 1):
                # Generate user message
                user_message = self.user_model.generate(...)
                
                # CRITICAL: Different system prompt handling for each model
                baked_response = self.baked_model.generate(
                    messages=baked_conversation_history + [{"role": "user", "content": user_message}]
                    # NO system_prompt parameter - persona is baked into weights
                )
                
                system_response = self.system_model.generate(
                    messages=system_conversation_history + [{"role": "user", "content": user_message}],
                    system_prompt=self.persona_system_prompt  # WITH persona system prompt
                )
                
                # Update progress bar with current status
                pbar.set_description(f"Turn {turn}/{self.num_turns} | Probes: {probe_count} | Baked: {baked_wins} | System: {system_wins}")
                pbar.update(1)
        
        return PersonaDriftV3Result(...)
    
    def load_benchmark_dataset(self, *args, **kwargs) -> List[Golden]:
        """Required by DeepEvalBaseBenchmark. Returns empty list since personas are user-provided."""
        return []
```

### Implementation Considerations
- **Error Handling**: Arena G-Eval may fail if judge model is unavailable or returns invalid JSON
- **Cost Management**: Each Arena G-Eval call incurs judge model cost (track cumulative costs)
- **Tie Handling**: Arena G-Eval always picks a winner (no ties), but reasoning may indicate close competition
- **Conversation Divergence**: As conversations evolve differently, user message generation becomes more challenging
- **Judge Consistency**: Same judge model should be used throughout benchmark for consistency

### ⚠️ CRITICAL: System Prompt Handling
- **Baked Model**: **NEVER** pass system prompt - persona is already in the model weights
- **System Model**: **ALWAYS** pass persona_system_prompt - this is how it gets the persona
- **Failure to follow this**: Will invalidate the entire comparison (both models would be system-prompted)

### DeepEval Integration Requirements
- **Base Class Inheritance**: **Must extend `DeepEvalBaseBenchmark`** (core requirement)
- **Required Abstract Methods**:
  - `evaluate(model: DeepEvalBaseLLM, *args, **kwargs) -> PersonaDriftV3Result`
  - `load_benchmark_dataset(*args, **kwargs) -> List[Golden]` (return empty list)
- **Telemetry**: Use `@capture_benchmark_run` decorator on `evaluate()` method
- **Progress Tracking**: Use `tqdm` for progress bars with descriptive format
- **Model Initialization**: Use `initialize_model()` helper for consistent model setup
- **Parameter Naming**: Use `evaluation_model` for Arena G-Eval judge (consistency with other benchmarks)
- **Result Object**: Must extend `DeepEvalBaseBenchmarkResult` with `overall_accuracy` field

## Success Criteria

1. **Functional**: Benchmark runs end-to-end with both model types
2. **Accurate**: Win rate calculations correctly reflect Arena G-Eval results
3. **Insightful**: Provides clear winner and detailed reasoning for each probe in conversation context
4. **Unbiased**: Blind evaluation and synchronized inputs prevent approach-based bias
5. **Conversation-Aware**: Tests persona embodiment in realistic conversation flow
6. **Well-tested**: Comprehensive test coverage
7. **Documented**: Clear usage examples and documentation
8. **Persona-Focused**: Tests persona embodiment rather than knowledge recall
9. **Actionable**: Results help decide between baked-in vs system-prompted approaches

## File Structure Summary

Following existing benchmark patterns, the final structure will be:

```
deepeval/benchmarks/persona_drift_v3/
├── __init__.py              # Module exports
├── template.py              # PersonaDriftV3Template static class
├── result.py                # PersonaDriftV3Result and ProbeResult classes
├── schema.py                # Pydantic schemas for data structures
├── persona_drift_v3.py      # PersonaDriftV3 main benchmark class
└── implementation.md        # This implementation plan
```

## Relationship to v2

### **Inherited from v2**:
- Persona embodiment focus (not knowledge recall)
ensure your- Template-based architecture
- Verbose logging patterns
- Neutral user message generation approach

### **New in v3**:
- Arena G-Eval integration for head-to-head comparison
- Dual model evaluation (baked vs system-prompted)
- Parallel synchronized conversations instead of single conversation
- Win rate based scoring instead of threshold-based
- Blind evaluation with dummy names
- Conversation-aware probe comparisons
- Probe-level result tracking with reasoning

### **Migration Path**:
- v2 users can upgrade to v3 for comparative analysis
- v2 probe generation approach directly reusable
- Similar API patterns for easy adoption

## Timeline
- **Phase 1**: Core structure and schemas (1 day)
- **Phase 2**: Arena G-Eval integration and templates (2 days)  
- **Phase 3**: Result objects and data structures (1 day)
- **Phase 4**: Main benchmark implementation (2-3 days)
- **Phase 5**: Testing and validation (2 days)
- **Phase 6**: Documentation and examples (1 day)
- **Total**: 7-10 days for complete implementation

This implementation leverages the solid foundation of v2's persona embodiment approach while adding the powerful head-to-head comparison capability of Arena G-Eval, creating a benchmark that provides direct, actionable insights about which persona implementation approach works better.
