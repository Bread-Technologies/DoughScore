from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class GenerationParams(BaseModel):
    temperature: float = Field(default=1.0)
    top_p: float = Field(default=0.9, alias="topP")
    max_tokens: int = Field(default=128, ge=1, alias="maxTokens")


class PersonaDriftConfig(BaseModel):
    model_name: str = Field(description="Model identifier for LiteLLMModel")
    turns: int = Field(default=8, ge=1, description="Number of turns per conversation")
    seed: int = Field(default=42, description="Random seed for run metadata")
    topic: Optional[str] = Field(default=None, description="Conversation topic prompt")
    agent_persona_id: Optional[str] = Field(default=None)
    user_persona_id: Optional[str] = Field(default=None)
    generation_params: GenerationParams = Field(default_factory=GenerationParams)


class PersonaTurnRecord(BaseModel):
    turn_index: int = Field(ge=1)
    reply: str
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    cost: Optional[float] = Field(default=None, ge=0.0)
    wall_time_s: Optional[float] = Field(default=None, ge=0.0)


class PersonaDriftRunArtifact(BaseModel):
    run_metadata: Dict = Field(default_factory=dict)
    persona_info: Dict = Field(default_factory=dict)
    per_turn: List[PersonaTurnRecord] = Field(default_factory=list)
    aggregates: Dict = Field(default_factory=dict)


