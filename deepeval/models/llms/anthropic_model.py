from typing import Optional, Tuple, Union, Dict, List, get_origin, get_args
from typing import Optional, Tuple, Union, Dict, List
from anthropic import Anthropic, AsyncAnthropic
from pydantic import BaseModel
import os
import warnings

from deepeval.models import DeepEvalBaseLLM
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models.utils import parse_model_name

model_pricing = {
    "claude-opus-4-20250514": {"input": 15.00 / 1e6, "output": 75.00 / 1e6},
    "claude-sonnet-4-20250514": {"input": 3.00 / 1e6, "output": 15.00 / 1e6},
    "claude-3-7-sonnet-latest": {"input": 3.00 / 1e6, "output": 15.00 / 1e6},
    "claude-3-5-haiku-latest": {"input": 0.80 / 1e6, "output": 4.00 / 1e6},
    "claude-3-5-sonnet-latest": {"input": 3.00 / 1e6, "output": 15.00 / 1e6},
    "claude-3-opus-latest": {"input": 15.00 / 1e6, "output": 75.00 / 1e6},
    "claude-3-sonnet-20240229": {"input": 3.00 / 1e6, "output": 15.00 / 1e6},
    "claude-3-haiku-20240307": {"input": 0.25 / 1e6, "output": 1.25 / 1e6},
    "claude-instant-1.2": {"input": 0.80 / 1e6, "output": 2.40 / 1e6},
}


class AnthropicModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: str = "claude-3-7-sonnet-latest",
        temperature: float = 0,
        _anthropic_api_key: Optional[str] = None,
        generation_kwargs: Optional[Dict] = None,
        enable_thinking: bool = False,
        thinking_budget_tokens: int = 1024,
        **kwargs,
    ):
        model_name = parse_model_name(model)
        self._anthropic_api_key = _anthropic_api_key

        if temperature < 0:
            raise ValueError("Temperature must be >= 0.")
        self.temperature = temperature

        self.enable_thinking = enable_thinking
        self.thinking_budget_tokens = thinking_budget_tokens
        
        # Warn if thinking is enabled but temperature is not 1
        if enable_thinking and temperature != 1:
            warnings.warn(
                "When thinking is enabled, temperature will be automatically set to 1 "
                "as required by Anthropic's extended thinking feature.",
                UserWarning
            )
        
        self.kwargs = kwargs
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(model_name)

    ###############################################
    # Generate functions
    ###############################################

    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        chat_model = self.load_model()
        message = chat_model.messages.create(
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": enhanced_prompt,
                }
            ],
            model=self.model_name,
            temperature=self.temperature,
            **self.generation_kwargs,
        )
        cost = self.calculate_cost(
            message.usage.input_tokens, message.usage.output_tokens
        )
        if schema is None:
            return message.content[0].text, cost
        else:
            json_output = trim_and_load_json(message.content[0].text)
            return schema.model_validate(json_output), cost

    async def a_chat_generate(
        self, messages: List[Dict[str, str]], schema: Optional[BaseModel] = None
    ) -> Tuple[str, float]:
        chat_model = self.load_model(async_mode=True)
        message = await chat_model.messages.create(
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_name,
            temperature=self.temperature,
            **self.generation_kwargs,
        )
        cost = self.calculate_cost(
            message.usage.input_tokens, message.usage.output_tokens
        )
        if schema is None:
            return message.content[0].text, cost
        else:
            json_output = trim_and_load_json(message.content[0].text)

            return schema.model_validate(json_output), cost

    def chat_generate(
        self, messages: List[Dict[str, str]], schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        chat_model = self.load_model()
        message = chat_model.messages.create(
            max_tokens=1024,
            messages=messages,
            model=self.model_name,
            temperature=self.temperature,
            **self.generation_kwargs,
        )
        cost = self.calculate_cost(
            message.usage.input_tokens, message.usage.output_tokens
        )
        if schema is None:
            return message.content[0].text, cost
        else:
            json_output = trim_and_load_json(message.content[0].text)
            return schema.model_validate(json_output), cost

    async def a_chat_generate(
        self, messages: List[Dict[str, str]], schema: Optional[BaseModel] = None
    ) -> Tuple[str, float]:
        chat_model = self.load_model(async_mode=True)
        message = await chat_model.messages.create(
            max_tokens=1024,
            messages=messages,
            model=self.model_name,
            temperature=self.temperature,
            **self.generation_kwargs,
        )
        cost = self.calculate_cost(
            message.usage.input_tokens, message.usage.output_tokens
        )
        if schema is None:
            return message.content[0].text, cost
        else:
            json_output = trim_and_load_json(message.content[0].text)
            return schema.model_validate(json_output), cost

    ###############################################
    # Utilities
    ###############################################

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = model_pricing.get(self.model_name)

        if pricing is None:
            # Calculate average cost from all known models
            avg_input_cost = sum(
                p["input"] for p in model_pricing.values()
            ) / len(model_pricing)
            avg_output_cost = sum(
                p["output"] for p in model_pricing.values()
            ) / len(model_pricing)
            pricing = {"input": avg_input_cost, "output": avg_output_cost}

            warnings.warn(
                f"[Warning] Pricing not defined for model '{self.model_name}'. "
                "Using average input/output token costs from existing model_pricing."
            )

        input_cost = input_tokens * pricing["input"]
        output_cost = output_tokens * pricing["output"]
        return input_cost + output_cost

    ###############################################
    # Model
    ###############################################

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY")
                or self._anthropic_api_key,
                **self.kwargs,
            )
        else:
            return AsyncAnthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY")
                or self._anthropic_api_key,
                **self.kwargs,
            )

    def get_model_name(self):
        return f"{self.model_name}"
    
    def _get_schema_instruction(self, schema: BaseModel) -> str:
        """Generate explicit JSON format instructions based on the schema structure."""
        try:
            # Get the schema's JSON schema to understand the structure
            json_schema = schema.model_json_schema()
            
            # Extract the main field info
            properties = json_schema.get('properties', {})
            if len(properties) != 1:
                return "CRITICAL: You must respond with a valid JSON object that matches the expected schema. Do not provide any explanation, reasoning, or additional text. Just the JSON object."
            
            field_name = list(properties.keys())[0]
            field_schema = properties[field_name]
            
            # Handle different field types based on JSON schema
            return self._generate_instruction_from_json_schema(field_name, field_schema)
                
        except Exception as e:
            print(f"DEBUG: Schema analysis failed: {e}")
            return "CRITICAL: You must respond with a valid JSON object that matches the expected schema. Do not provide any explanation, reasoning, or additional text. Just the JSON object."
    
    def _generate_instruction_from_json_schema(self, field_name: str, field_schema: dict) -> str:
        """Generate instruction based on JSON schema field definition."""
        field_type = field_schema.get('type', 'string')
        
        # Handle enum/constraint types
        if 'enum' in field_schema:
            enum_values = field_schema['enum']
            return self._generate_enum_instruction(field_name, enum_values)
        
        # Handle string types
        elif field_type == 'string':
            return f"CRITICAL: You must respond with a JSON object in this exact format: {{\"{field_name}\": \"your_string_here\"}}. You may think through the problem, but your final response must be a JSON object."
        
        # Handle integer types
        elif field_type == 'integer':
            return f"CRITICAL: You must respond with a JSON object in this exact format: {{\"{field_name}\": 42}} where the answer is a number. You may think through the problem, but your final response must be a JSON object."
        
        # Handle number types
        elif field_type == 'number':
            return f"CRITICAL: You must respond with a JSON object in this exact format: {{\"{field_name}\": 3.14}} where the answer is a number. You may think through the problem, but your final response must be a JSON object."
        
        # Handle boolean types
        elif field_type == 'boolean':
            return f"CRITICAL: You must respond with a JSON object in this exact format: {{\"{field_name}\": true}} or {{\"{field_name}\": false}}. You may think through the problem, but your final response must be a JSON object."
        
        # Handle array types
        elif field_type == 'array':
            return f"CRITICAL: You must respond with a JSON object in this exact format: {{\"{field_name}\": [...]}} where the answer is a list. You may think through the problem, but your final response must be a JSON object."
        
        # Fallback
        else:
            return f"CRITICAL: You must respond with a JSON object in this exact format: {{\"{field_name}\": \"your_answer_here\"}}. You may think through the problem, but your final response must be a JSON object."
    
    def _generate_enum_instruction(self, field_name: str, enum_values: list) -> str:
        """Generate instruction for enum/constraint types."""
        values_str = ", ".join(f'"{v}"' for v in enum_values)
        
        # Special handling for multiple choice schemas (single letters)
        if all(isinstance(v, str) and len(v) == 1 and v.isalpha() for v in enum_values):
            return f"CRITICAL: You must respond with a JSON object in this exact format: {{\"{field_name}\": \"A\"}} where the answer is ONLY the letter ({', '.join(enum_values)}) - NOT the full answer text. You may think through the problem, but your final response must be a JSON object. For example, if the correct answer is 'A) Paris', you should respond with {{\"{field_name}\": \"A\"}}, not {{\"{field_name}\": \"A) Paris\"}}. Extract just the letter from your choice."
        
        # Special handling for parenthesized choices
        elif all(isinstance(v, str) and v.startswith('(') and v.endswith(')') for v in enum_values):
            return f"CRITICAL: You must respond with a JSON object in this exact format: {{\"{field_name}\": \"(A)\"}} where the answer is one of: {values_str}. You may think through the problem, but your final response must be a JSON object."
        
        # Special handling for Yes/No
        elif all(v in ['Yes', 'No'] for v in enum_values):
            return f"CRITICAL: You must respond with a JSON object in this exact format: {{\"{field_name}\": \"Yes\"}} or {{\"{field_name}\": \"No\"}}. You may think through the problem, but your final response must be a JSON object."
        
        # Special handling for lowercase yes/no
        elif all(v in ['yes', 'no'] for v in enum_values):
            return f"CRITICAL: You must respond with a JSON object in this exact format: {{\"{field_name}\": \"yes\"}} or {{\"{field_name}\": \"no\"}}. You may think through the problem, but your final response must be a JSON object."
        
        # Special handling for True/False
        elif all(v in ['True', 'False'] for v in enum_values):
            return f"CRITICAL: You must respond with a JSON object in this exact format: {{\"{field_name}\": \"True\"}} or {{\"{field_name}\": \"False\"}}. You may think through the problem, but your final response must be a JSON object."
        
        # Special handling for valid/invalid
        elif all(v in ['valid', 'invalid'] for v in enum_values):
            return f"CRITICAL: You must respond with a JSON object in this exact format: {{\"{field_name}\": \"valid\"}} or {{\"{field_name}\": \"invalid\"}}. You may think through the problem, but your final response must be a JSON object."
        
        # Generic enum handling
        else:
            return f"CRITICAL: You must respond with a JSON object in this exact format: {{\"{field_name}\": \"{enum_values[0]}\"}} where the answer is one of: {values_str}. You may think through the problem, but your final response must be a JSON object."
    
    def _extract_json_from_thinking_model(self, text: str) -> dict:
        """Extract JSON from thinking model responses that may include reasoning."""
        import re
        import json
        
        # First, try to parse the entire text as JSON (most common case)
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        # If that fails, try to find JSON objects in the text
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested JSON object
            r'\{.*\}',  # Any JSON object (non-greedy)
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    # Clean up the match
                    cleaned = match.strip()
                    # Validate that braces are balanced
                    if cleaned.count('{') == cleaned.count('}'):
                        return json.loads(cleaned)
                except json.JSONDecodeError:
                    continue
        
        # If no JSON found, try the original trim_and_load_json approach
        return trim_and_load_json(text)
