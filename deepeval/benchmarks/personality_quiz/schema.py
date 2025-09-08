from pydantic import BaseModel, Field
from typing import Optional


class PersonalityResponseSchema(BaseModel):
    """Schema for personality quiz responses using Likert scale."""
    
    rating: int = Field(
        ge=1, 
        le=5, 
        description="Numeric rating from 1-5 on the Likert scale"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0, 
        le=1.0, 
        description="Confidence in the response (0.0 to 1.0)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "rating": 4,
                "confidence": 0.8
            }
        }
