# Persona Drift v2 Benchmark Implementation Plan

## Overview
This document outlines the implementation of Persona Drift v2, a completely new benchmark that measures how well a chatbot model maintains its assigned persona over the course of a natural dialog using branch probing and G-Eval scoring.

**Key Insight**: For baked-in persona models, the benchmark focuses on **persona embodiment** (voice, style, perspective) rather than **knowledge recall** (factual accuracy). This makes the benchmark more meaningful for evaluating how well models maintain their trained personas over extended conversations.

## Core Architecture

### Three-Model Setup
- **Agent Model**: The model under test (can be baked-in or system-prompted)
- **User Model**: Handles multiple roles:
  - Neutral conversation generation
  - Probe question generation  
  - G-Eval judge for persona adherence scoring

### Branch Probing Protocol
- Main conversation flows naturally: User → Agent → User → Agent
- After each agent reply, perform a "branch probe":
  - Copy conversation history + system prompt
  - Inject a probe question as user message
  - Get agent's reply to probe
  - Score reply using G-Eval judge
  - **Critical**: Branch probe does NOT affect main conversation flow

## Implementation Progress

**✅ COMPLETED PHASES (1-5):**
- ✅ **Phase 1**: Directory structure and task definition
- ✅ **Phase 2**: Template implementation with G-Eval integration
- ✅ **Phase 3**: Result object implementation
- ✅ **Phase 4**: Main benchmark implementation
- ✅ **Phase 5**: Testing and validation

**🔄 REMAINING PHASES (6):**
- 🔄 **Phase 6**: Documentation and examples

**Current Status**: Core benchmark implementation is complete and functional. Comprehensive tests and examples have been created. Ready for final documentation phase.

**Recent Enhancement**: Updated benchmark philosophy to focus on **persona embodiment** rather than knowledge recall, making it more suitable for evaluating baked-in persona models.

**Recent Enhancement**: Integrated proper G-Eval framework (`deepeval.metrics.g_eval.g_eval.GEval`) with comprehensive evaluation criteria covering both content and tone adherence to persona, replacing custom prompt-based scoring.

**Verification Status**: All completed phases have been verified and are fully functional:
- ✅ All imports working correctly
- ✅ All template methods implemented and accessible
- ✅ Result object with proper Pydantic model fields
- ✅ Main benchmark class with all required methods
- ✅ No linting errors detected
- ✅ Comprehensive test suite created and passing
- ✅ Example scripts and test runners implemented

**Phase 5 Completion Summary**:
- ✅ Created comprehensive unit tests for all components (result, template, benchmark)
- ✅ Created integration tests covering full benchmark workflows
- ✅ Created example scripts demonstrating usage patterns
- ✅ Created root-level test script for easy validation
- ✅ **Simplified architecture**: Removed task.py as personas are user-defined
- ✅ Updated all tests and examples to work without predefined tasks
- ✅ Fixed all test issues and ensured compatibility with actual implementation

**Architecture Simplification**:
The benchmark was simplified by removing the `task.py` file since:
- Personas are user-defined via `persona_system_prompt` parameter
- No predefined personas are needed - users can create any persona they want
- The task enum was only used for logging/metadata, adding complexity without value
- This makes the benchmark more flexible and easier to use

**Persona Embodiment Focus**:
The benchmark has been updated to focus on persona embodiment rather than knowledge recall:
- **Probe Generation**: Tests persona's voice, style, perspective, and character consistency
- **G-Eval Scoring**: Evaluates persona embodiment, voice consistency, and character alignment
- **Scoring Philosophy**: "Does this response sound like it came from this persona?" rather than "Does this response demonstrate factual knowledge?"

## Implementation Phases

### Phase 1: Directory Structure and Task Definition ✅ **COMPLETED**
**Scope**: Create new benchmark directory and define task structure following existing patterns
**Files created**:
- ✅ `deepeval/benchmarks/persona_drift_v2/__init__.py`
- ✅ `deepeval/benchmarks/persona_drift_v2/task.py`
- ✅ `deepeval/benchmarks/persona_drift_v2/template.py`
- ✅ `deepeval/benchmarks/persona_drift_v2/result.py`

**Task Structure** (following SQuAD/LogiQA pattern):
- ✅ `PersonaDriftV2Task`: Enum defining available persona types
- ✅ `PersonaDriftV2Template`: Static class for message construction and prompt generation
- ✅ `PersonaDriftV2Result`: Custom result object extending `DeepEvalBaseBenchmarkResult`

**Exit Criteria**: ✅ Task enum defined, template structure in place, result object extends base class

### Phase 2: Template Implementation ✅ **COMPLETED**
**Scope**: Implement template class following existing benchmark patterns
**Files created**:
- ✅ `deepeval/benchmarks/persona_drift_v2/template.py` (updated)

**Template Structure** (following SQuAD/LogiQA pattern):
- ✅ `PersonaDriftV2Template`: Static class with methods for:
  - ✅ `generate_neutral_message()`: Generate neutral conversation messages
  - ✅ `generate_probe_questions()`: Generate probe questions for persona
  - ✅ `build_conversation_messages()`: Build message list for agent model
  - ✅ `build_probe_messages()`: Build message list for branch probe
  - ✅ `create_persona_adherence_geval()`: Create G-Eval metric for persona adherence scoring
- ✅ **Note**: User provides persona system prompts directly; no predefined catalog

**System Prompts** (hardcoded in template class):
- ✅ **Neutral Conversation**: "Generate neutral, topical conversation messages that don't reinforce any persona. Avoid asking questions that would reveal or test persona characteristics." (Includes persona system prompt context to avoid reinforcing it)
- ✅ **Probe Generation**: "Generate probe questions that test the persona's voice, style, perspective, and character consistency. Focus on persona embodiment rather than specific knowledge recall."
- ✅ **G-Eval Judge**: Uses proper G-Eval framework with comprehensive criteria covering persona embodiment, voice consistency, and character alignment

**Exit Criteria**: ✅ Template class implemented with all message building methods

### Phase 3: Result Object Implementation ✅ **COMPLETED**
**Scope**: Implement custom result object following existing benchmark patterns
**Files created**:
- ✅ `deepeval/benchmarks/persona_drift_v2/result.py` (updated)

**Result Structure** (following existing patterns):
- ✅ `PersonaDriftV2Result`: Extends `DeepEvalBaseBenchmarkResult` with:
  - ✅ `overall_accuracy`: Inherited from base class
  - ✅ `n_drift`: First turn where persona is lost
  - ✅ `total_turns`: Number of conversation turns
  - ✅ `total_cost`: Total cost of benchmark run
  - ✅ `total_time_s`: Total time in seconds
  - ✅ `run_metadata`: Configuration and seed information

**Exit Criteria**: ✅ Result object extends base class, includes N_drift tracking

### Phase 4: Main Benchmark Implementation ✅ **COMPLETED**
**Scope**: Implement main benchmark class following existing patterns
**Files created**:
- ✅ `deepeval/benchmarks/persona_drift_v2/persona_drift_v2.py`

**Benchmark Structure** (following SQuAD/LogiQA pattern):
- ✅ `PersonaDriftV2`: Extends `DeepEvalBaseBenchmark[PersonaDriftV2Task]`
- ✅ Constructor parameters: `tasks`, `agent_model`, `user_model`, `persona_system_prompt`, `is_baked_in`, `turns`, `seed`, `verbose_mode`
- ✅ Methods: `evaluate()`, `predict()`, `load_benchmark_dataset()`, `print_verbose_logs()`

**Core Logic**:
- ✅ 1. Generate 10 probe questions at start using user model
- ✅ 2. For each turn: user generates neutral message → agent replies → branch probe → score
- ✅ 3. Track N_drift (first turn where score < 5.0/10 = 0.5)
- ✅ 4. Return `PersonaDriftV2Result`

**Exit Criteria**: ✅ Main benchmark class working, follows existing patterns

### Phase 5: Testing and Validation
**Scope**: Create comprehensive tests and example usage following existing patterns
**Files to create**:
- `examples/persona_drift_v2_example.py`
- `test_persona_drift_v2.py`
- `tests/test_persona_drift_v2/`

**Test Structure** (following existing patterns):
- Unit tests for each component (task, template, result)
- Integration tests for full benchmark
- Example with both baked-in and system-prompted models
- Validation of N_drift calculation
- Mock model testing for CI/CD

**Exit Criteria**: All tests passing, examples working, follows existing test patterns

### Phase 6: Documentation and Examples
**Scope**: Add documentation and usage examples following existing patterns
**Files to create**:
- `docs/benchmarks/persona_drift_v2.md`
- `examples/persona_drift_v2_example.py`

**Components**:
- Comprehensive documentation following existing benchmark docs
- Usage examples and best practices
- Integration examples with existing DeepEval patterns

**Exit Criteria**: Documentation complete, examples working, follows existing patterns

## Key Design Decisions

### Persona Embodiment vs Knowledge Recall
- **Decision**: Focus on persona embodiment (voice, style, perspective) rather than knowledge recall
- **Rationale**: For baked-in persona models, the goal is to test if the model maintains its character and voice, not just factual accuracy. This makes the benchmark more meaningful for evaluating persona drift.

### Branch Probing vs Inline Probing
- **Decision**: Use branch probing exclusively
- **Rationale**: Prevents probe context from affecting main conversation, more realistic evaluation

### Three-Model Architecture
- **Decision**: Separate agent and user models with user model handling multiple roles
- **Rationale**: Allows for objective evaluation while maintaining realistic conversation flow

### G-Eval Scoring
- **Decision**: Use G-Eval judge instead of deterministic functions
- **Rationale**: More flexible and can handle complex persona characteristics, focusing on persona embodiment rather than knowledge recall

### Baked-in Model Support
- **Decision**: Explicit flag for baked-in models
- **Rationale**: Clear distinction between system-prompted and baked-in personas

### Probe Generation
- **Decision**: Generate probes once at start, rotate through set
- **Rationale**: Consistent evaluation while avoiding probe repetition, focusing on persona embodiment rather than knowledge recall

## Technical Specifications

### Model Requirements
- **Agent Model**: Must support `chat_generate` method
- **User Model**: Must support `chat_generate` method, used for multiple roles
- **G-Eval Judge**: User model with specific system prompt for scoring

### System Prompts
- **Neutral Conversation**: Generic, non-persona-reinforcing, knows what to avoid
- **Probe Generation**: Persona-aware, designed to test persona embodiment and character consistency
- **G-Eval Judge**: Persona-aware, objective scoring focused on voice, style, and perspective alignment

### Result Object
- **N_drift**: First turn where persona is lost
- **Basic Statistics**: Essential metrics only (no full conversation logs)
- **Metadata**: Configuration and run information

### Reproducibility
- **Seeds**: Same seed used for all random operations in a single run
- **Configuration**: All parameters logged for reproducibility
- **Minimal Logging**: Only essential data for reproducibility (no full conversation logs)

## Practical Changes Needed

### 1. Probe Generation Updates
- **Current**: "Generate questions that test specific knowledge, experiences, or characteristics the persona should have"
- **New**: "Generate questions that test the persona's voice, style, perspective, and character consistency"
- **Focus**: Persona embodiment rather than factual accuracy

### 2. G-Eval Criteria Updates
- **Current**: Focus on factual accuracy and knowledge demonstration
- **New**: Focus on persona embodiment, voice consistency, and character alignment
- **Scoring**: "Does this response sound like it came from this persona?" rather than "Does this response demonstrate expertise?"

### 3. Scoring Philosophy
- **Current**: Tests knowledge recall and expertise demonstration
- **New**: Tests persona maintenance, voice consistency, and character alignment
- **Goal**: Detect when the model loses its persona's voice/perspective, not just factual knowledge

## Success Criteria

1. **Functional**: Benchmark runs end-to-end with both baked-in and system-prompted models
2. **Accurate**: N_drift calculation correctly identifies persona drift points
3. **Reproducible**: Same configuration produces same results
4. **Extensible**: Easy to add new personas and probe types
5. **Well-tested**: Comprehensive test coverage
6. **Documented**: Clear usage examples and documentation (no CLI required)
7. **Persona-Focused**: Tests persona embodiment rather than knowledge recall

## File Structure Summary

Following existing benchmark patterns, the final structure will be:

```
deepeval/benchmarks/persona_drift_v2/
├── __init__.py              # Module exports
├── task.py                  # PersonaDriftV2Task enum
├── template.py              # PersonaDriftV2Template static class
├── result.py                # PersonaDriftV2Result class
└── persona_drift_v2.py      # PersonaDriftV2 main benchmark class
```

## Timeline
- **Phases 1-3**: Core architecture and data structures (1-2 days)
- **Phase 4**: Main benchmark implementation (2-3 days)
- **Phase 5**: Testing and validation (1-2 days)
- **Phase 6**: Documentation and examples (1 day)
- **Total**: 5-8 days for complete implementation
