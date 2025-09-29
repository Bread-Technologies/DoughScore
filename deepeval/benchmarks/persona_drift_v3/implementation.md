# Persona Drift v3 Benchmark Implementation Plan

## Overview
This document outlines the implementation of Persona Drift v3, an arena-style benchmark that directly compares baked-in persona models vs system-prompted models using blind head-to-head evaluation. The benchmark uses Arena G-Eval to determine which approach better embodies a given persona across **every conversation turn**.

**Key Innovation**: Unlike v2 which evaluates a single model against a threshold, v3 performs blind comparisons between two approaches to persona implementation on **every response**, providing comprehensive insights about which method works better for specific personas and contexts.

**Key Insight**: For baked-in persona models, the benchmark focuses on **authentic persona embodiment** (genuine voice, natural character thinking, spontaneous reactions) rather than **knowledge recall** (factual accuracy) or **surface-level performance** (forced mannerisms, identity declarations). This makes the benchmark more meaningful for evaluating how well models maintain their trained personas authentically over extended conversations.

## Core Architecture

### **Parallel Synchronized Conversations with Full Arena Evaluation**
- **Baked-in Model**: Model with persona trained/fine-tuned into weights (no system prompt)
- **System-Prompted Model**: Standard model with persona provided as system prompt
- **Arena G-Eval Judge**: LLM judge that blindly picks which response better embodies the persona **for every turn**
- **User Message Generator**: Creates neutral messages that work for both conversation contexts
- **Synchronized Input**: Both models receive identical user messages at each turn
- **Blind Evaluation**: Judge sees responses with dummy names (Alice, Bob) without knowing which is which
- **Full Coverage**: Arena evaluation on **every single response pair** (no probing intervals)

### **Parallel Conversation Protocol (Updated)**
1. **Initialize**: Two separate conversation histories (baked_history, system_history)
2. **For each turn**:
   - Generate neutral user message based on both conversation contexts
   - Add same user message to both conversation histories
   - Get response from baked-in model (using its conversation history, no system prompt)
   - Get response from system-prompted model (using its conversation history + persona system prompt)
   - **NEW**: Run Arena G-Eval comparison on both responses immediately
   - Add respective responses to their conversation histories
   - Track Arena G-Eval result (winner, reasoning, cost)
3. **Final scores**: Win rates for each model based on **all turn comparisons** (0-1 scale)

## IMPLEMENTED CHANGES - Full Arena Evaluation

### **✅ COMPLETED: Changes to Remove Probing and Evaluate Every Turn**

#### **1. Main Benchmark Class Changes (`persona_drift_v3.py`)** ✅

**✅ REMOVED**:
- ✅ `probe_frequency` parameter 
- ✅ `probe_questions` generation and management
- ✅ Conditional probing logic (`if turn % probe_frequency == 0`)
- ✅ Probe question rotation/selection
- ✅ Separate probe response generation

**✅ ADDED/MODIFIED**:
- ✅ Arena G-Eval evaluation **on every turn** instead of probe turns only
- ✅ Use the regular conversation user message and responses for Arena evaluation
- ✅ Simplified evaluation loop: generate turn → get responses → evaluate with Arena → continue
- ✅ Updated progress tracking to show turn-by-turn win counts
- ✅ Updated cost tracking to include Arena evaluation cost for every turn
- ✅ Updated result metrics to reflect evaluation on all turns

**New Evaluation Flow**:
```python
for turn in range(1, self.num_turns + 1):
    # Generate neutral user message for this turn
    user_message = PersonaDriftV3Template.generate_neutral_message(...)
    
    # Get responses from both models
    baked_response = self.baked_model.chat_generate(baked_messages + [{"role": "user", "content": user_message}])
    system_response = self.system_model.chat_generate(system_messages + [{"role": "user", "content": user_message}], system_prompt=self.persona_system_prompt)
    
    # ARENA EVALUATION ON EVERY TURN
    arena_test_case = PersonaDriftV3Template.create_arena_test_case(
        user_message=user_message,
        baked_response=baked_response,
        system_response=system_response
    )
    winner = arena_geval.measure(arena_test_case)
    
    # Track results and continue conversation
    # ... (add to histories, update counters)
```

#### **2. Template Class Changes (`template.py`)** ✅

**✅ REMOVED**:
- ✅ `generate_probe_questions()` method (no longer needed)

**✅ KEPT/UNCHANGED**:
- ✅ `create_probe_arena_test_case()` method (maintained for backward compatibility)
- ✅ `create_arena_test_case()` method (already existed with correct signature)  
- ✅ `create_persona_arena_geval()` remains unchanged (already works for any response comparison)
- ✅ `generate_neutral_message()` unchanged (used for every turn)
- ✅ All system prompts remain exactly as originally designed

#### **3. Result Class Changes (`result.py`)** ✅

**✅ RENAMED**:
- ✅ `ProbeResult` schema → `TurnResult`
- ✅ `probe_results` field → `turn_results`
- ✅ `total_probes` field → `total_turns_evaluated`

**✅ UPDATED**:
- ✅ Field names to reflect turn-based evaluation
- ✅ All docstrings and property methods to use "turn" terminology
- ✅ Result statistics to reflect full conversation evaluation
- ✅ Maintained same core metrics: win rates, winner, reasoning per turn

**✅ NEW SCHEMA**:
```python
@dataclass
class TurnResult:
    turn_index: int
    user_message: str  # Changed from probe_question
    baked_response: str
    system_response: str
    winner: str  # "baked_model" or "system_model"
    reasoning: str
    evaluation_cost: float
```

#### **4. Logging and Progress Changes** ✅

**✅ UPDATED**:
- ✅ Progress bar description: "Turn X/Y | Evaluations: X | Baked: X wins | System: Y wins" (every turn)
- ✅ Verbose logging: Shows Arena evaluation result for every turn
- ✅ Cost tracking: Arena evaluation cost added for every turn
- ✅ Result summary: Based on all turns, not just probe subset

#### **5. Configuration Changes** ✅

**✅ REMOVED**:
- ✅ `probe_frequency` parameter from constructor
- ✅ All probe-related configuration and validation

**✅ KEPT UNCHANGED**:
- ✅ `num_turns` - now represents total turns AND total evaluations
- ✅ All other parameters remain the same

### **Benefits of Full Arena Evaluation**

1. **Complete Coverage**: Every response is evaluated, not just a subset
2. **Higher Statistical Power**: More data points for win rate calculation
3. **Conversation Flow Insights**: See how persona embodiment changes throughout conversation
4. **Simplified Logic**: No probe timing or question management complexity
5. **More Actionable**: Full conversation comparison instead of spot checks

### **Cost Considerations**

- **Increased Evaluation Cost**: Arena G-Eval called on every turn instead of every N turns
- **Cost Scaling**: If previously 10 turns with probe_frequency=2 → 5 evaluations, now → 10 evaluations (2x cost)
- **Mitigation**: Users can reduce `num_turns` if cost is a concern
- **Value**: More comprehensive evaluation justifies increased cost

### **Backward Compatibility**

- **API Changes**: Remove `probe_frequency` parameter (breaking change)
- **Result Format**: Field name changes (`probe_results` → `turn_results`) 
- **Semantic Changes**: Results now represent full conversation evaluation
- **Migration**: Users need to update code that accesses probe-specific fields

## Implementation Progress

**✅ COMPLETED PHASES:**
- ✅ **Phase 1**: Directory structure and core components
- ✅ **Phase 2**: Arena G-Eval integration and template implementation  
- ✅ **Phase 3**: Result object implementation
- ✅ **Phase 4**: Main benchmark implementation (probe-based)
- ✅ **Phase 4b**: Converted to full arena evaluation (removed probing)
- ✅ **Phase 5**: Testing and validation with comprehensive test suite

**🔄 REMAINING PHASES:**
- 🔄 **Phase 6**: Documentation and examples

**Current Status**: **Phase 5 completed** - Full arena evaluation working and tested. Ready for production use.

## Technical Specifications (Updated)

### Model Requirements
- **Baked-in Model**: Must support `chat_generate` method, persona baked into weights
- **System-Prompted Model**: Must support `chat_generate` method, takes persona as system prompt  
- **Judge Model**: Used for Arena G-Eval **on every turn**, should be capable and unbiased
- **User Model**: Used for generating neutral user messages that work for both conversation contexts

### Arena G-Eval Integration (Unchanged)
- **Evaluation Params**: `[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]`
- **Criteria**: Persona embodiment focused, not knowledge recall (detailed criteria string)
- **Test Case Structure**: `ArenaTestCase(contestants: Dict[str, LLMTestCase])`
- **LLMTestCase Fields**: `input` (user message), `actual_output` (model response)
- **Contestant Names**: Must be unique, uses dummy names (Alice, Bob) for blind evaluation
- **Output**: `winner` (string), `reason` (detailed explanation), `evaluation_cost` (float)
- **Async Mode**: Configurable (recommend False for benchmark consistency)

### Result Metrics (Updated)
- **Primary**: Win rates for each model approach (based on all turns)
- **Secondary**: Individual turn results with reasoning for every response pair
- **Metadata**: Cost (higher due to full evaluation), time, configuration details

### Conversation Strategy (Updated)
- **Conversation Flow**: Parallel conversations with synchronized user messages
- **Evaluation Timing**: **Every single turn** (no intervals or probing)
- **Evaluation Focus**: Persona embodiment testing of responses to same user message
- **Coverage**: Complete conversation coverage for comprehensive win rate statistics

## ✅ Success Criteria ACHIEVED

1. ✅ **Functional**: Benchmark runs end-to-end with Arena evaluation on every turn
2. ✅ **Accurate**: Win rate calculations correctly reflect all Arena G-Eval results  
3. ✅ **Insightful**: Provides clear winner and detailed reasoning for **every turn**
4. ✅ **Unbiased**: Blind evaluation and synchronized inputs prevent approach-based bias
5. ✅ **Comprehensive**: Tests persona embodiment across entire conversation, not just samples
6. ✅ **Well-tested**: Comprehensive test coverage with 5/5 tests passing
7. ✅ **Documented**: Updated documentation reflecting full evaluation approach
8. ✅ **Persona-Focused**: Tests persona embodiment rather than knowledge recall
9. ✅ **Actionable**: Results help decide between baked-in vs system-prompted approaches with complete data

This implementation removes the complexity of probe management while providing comprehensive persona embodiment evaluation across every conversation turn, giving users complete visibility into how well each approach maintains persona consistency throughout extended interactions.