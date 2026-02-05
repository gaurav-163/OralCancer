"""
LangGraph State Management
"""
from typing import TypedDict, Optional, Any
from pydantic import BaseModel, Field
import numpy as np


class PatientInfo(TypedDict, total=False):
    """Patient information dictionary"""
    age: Optional[int]
    gender: str
    tobacco_use: str
    alcohol_use: str
    symptoms: list[str]
    duration: str


class PredictionResult(TypedDict):
    """ML model prediction result"""
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]


class AnalysisState(BaseModel):
    """
    State object for the LangGraph analysis workflow.
    
    This maintains the state as it flows through different nodes
    in the analysis graph.
    """
    # Input data
    image: Any = Field(default=None, description="Preprocessed image array")
    patient_info: dict = Field(default_factory=dict, description="Patient information")
    
    # ML Prediction
    prediction: dict = Field(default_factory=dict, description="ML model prediction results")
    
    # LLM Analysis
    analysis: str = Field(default="", description="LLM-generated analysis text")
    risk_level: str = Field(default="Unknown", description="Overall risk assessment")
    recommendations: list = Field(default_factory=list, description="List of recommendations")
    
    # Workflow control
    analysis_complete: bool = Field(default=False, description="Whether analysis is complete")
    error: Optional[str] = Field(default=None, description="Error message if any")
    
    class Config:
        arbitrary_types_allowed = True
    
    def to_dict(self) -> dict:
        """Convert state to dictionary for output"""
        return {
            "prediction": self.prediction,
            "analysis": self.analysis,
            "risk_level": self.risk_level,
            "recommendations": self.recommendations,
            "analysis_complete": self.analysis_complete,
            "error": self.error
        }
