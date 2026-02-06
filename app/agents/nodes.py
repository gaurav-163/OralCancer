"""
LangGraph Workflow Nodes
"""
from typing import Any
import os
import logging

from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage, SystemMessage

from .state import AnalysisState
from utils.prompts import get_analysis_prompt, get_recommendation_prompt
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def create_llm(config: dict) -> ChatCohere:
    """Create Cohere LLM instance from config"""
    # Get API key from environment or Streamlit secrets
    api_key = os.getenv("COHERE_API_KEY")
    
    # Fallback to Streamlit secrets if available
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets["COHERE_API_KEY"]
        except Exception as e:
            logger.debug(f"Could not access Streamlit secrets: {e}")
    
    if not api_key:
        raise ValueError("COHERE_API_KEY not found in environment variables or Streamlit secrets")
    
    return ChatCohere(
        cohere_api_key=api_key,
        model=config["llm"]["model"],
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"]
    )


def analyze_prediction_node(state: AnalysisState, config: dict) -> AnalysisState:
    """
    Node that analyzes the ML prediction using LLM.
    """
    try:
        llm = create_llm(config)
        
        # Build the analysis prompt
        prompt = get_analysis_prompt(
            prediction=state.prediction,
            patient_info=state.patient_info
        )
        
        messages = [
            SystemMessage(content="""You are an expert medical AI assistant specializing in oral pathology. 
            Your role is to analyze ML model predictions for oral lesions and provide clear, 
            professional explanations. Always maintain a balanced tone - be informative but not alarmist.
            Keep your response concise (2-3 paragraphs). Remember to always recommend professional medical consultation."""),
            HumanMessage(content=prompt)
        ]
        
        logger.info("Calling Cohere LLM for analysis...")
        response = llm.invoke(messages)
        state.analysis = response.content
        logger.info("Analysis generated successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        state.error = f"Analysis failed: {str(e)}"
        # Provide a meaningful fallback analysis based on the prediction
        prediction = state.prediction
        pred_class = prediction.get("predicted_class", "Unknown")
        confidence = prediction.get("confidence", 0)
        
        if pred_class == "Oral Cancer" and confidence > 0.7:
            state.analysis = f"""The AI classification model has detected characteristics consistent with oral cancer with {confidence*100:.1f}% confidence. 
            
This finding warrants immediate attention from a qualified healthcare professional. The model's high confidence level suggests visible indicators that should be evaluated through clinical examination and potentially biopsy.

Important: This is an AI-assisted screening tool and not a definitive diagnosis. Please consult with an oral pathologist or oncologist for proper evaluation and treatment planning."""
        elif pred_class == "Oral Cancer":
            state.analysis = f"""The AI classification model has identified some characteristics that may be associated with oral abnormalities ({confidence*100:.1f}% confidence for {pred_class}).

Given these findings, we recommend scheduling an appointment with a dental or medical professional for a thorough clinical examination. Early detection and professional evaluation are key to proper management.

This is an AI-assisted screening tool and should not replace professional medical advice."""
        else:
            state.analysis = """The image has been analyzed by our AI classification model. Based on the results shown in the classification section, please review the probability scores for each category.

We recommend discussing these findings with a healthcare professional, especially if you have any symptoms or concerns about oral health. Regular dental checkups are important for maintaining oral health."""
    
    return state


def assess_risk_node(state: AnalysisState, config: dict) -> AnalysisState:
    """
    Node that assesses overall risk level based on prediction and patient factors.
    """
    try:
        prediction = state.prediction
        patient_info = state.patient_info
        
        # Get the highest probability class (excluding Normal)
        class_probs = prediction.get("class_probabilities", {})
        normal_prob = class_probs.get("Normal", 0)
        
        # Calculate risk based on prediction confidence
        max_abnormal_prob = max(
            [v for k, v in class_probs.items() if k != "Normal"],
            default=0
        )
        
        # Adjust risk based on patient factors
        risk_multiplier = 1.0
        
        if patient_info.get("tobacco_use") == "Current":
            risk_multiplier += 0.3
        elif patient_info.get("tobacco_use") == "Former":
            risk_multiplier += 0.1
            
        if patient_info.get("alcohol_use") in ["Regular", "Heavy"]:
            risk_multiplier += 0.2
            
        if patient_info.get("duration") == "More than 3 months":
            risk_multiplier += 0.2
        elif patient_info.get("duration") == "1-3 months":
            risk_multiplier += 0.1
        
        # Age factor
        age = patient_info.get("age")
        if age and age > 50:
            risk_multiplier += 0.1
        
        # Calculate final risk score
        risk_score = max_abnormal_prob * risk_multiplier
        
        # Determine risk level
        if risk_score >= 0.7 or max_abnormal_prob >= 0.8:
            state.risk_level = "High"
        elif risk_score >= 0.4 or max_abnormal_prob >= 0.5:
            state.risk_level = "Moderate"
        elif normal_prob >= 0.7:
            state.risk_level = "Low"
        else:
            state.risk_level = "Moderate"
            
    except Exception as e:
        logger.error(f"Risk assessment failed: {str(e)}")
        state.error = f"Risk assessment failed: {str(e)}"
        state.risk_level = "Unknown"
    
    return state


def generate_recommendations_node(state: AnalysisState, config: dict) -> AnalysisState:
    """
    Node that generates specific recommendations based on analysis.
    """
    # Generate recommendations based on risk level (no LLM needed for reliability)
    risk_level = state.risk_level
    prediction = state.prediction
    patient_info = state.patient_info
    
    recommendations = []
    
    if risk_level == "High":
        recommendations = [
            {"priority": "high", "text": "Seek immediate consultation with an oral pathologist or oncologist"},
            {"priority": "high", "text": "Request a comprehensive oral examination and possible biopsy"},
            {"priority": "medium", "text": "Document any changes in symptoms and bring records to your appointment"},
            {"priority": "medium", "text": "Avoid tobacco and alcohol until professional evaluation"}
        ]
    elif risk_level == "Moderate":
        recommendations = [
            {"priority": "high", "text": "Schedule an appointment with a dental professional within 1-2 weeks"},
            {"priority": "medium", "text": "Monitor the area for any changes in size, color, or symptoms"},
            {"priority": "medium", "text": "Take photos to track any visible changes over time"},
            {"priority": "low", "text": "Maintain good oral hygiene practices"}
        ]
    else:
        recommendations = [
            {"priority": "medium", "text": "Continue regular dental checkups every 6 months"},
            {"priority": "low", "text": "Maintain good oral hygiene with regular brushing and flossing"},
            {"priority": "low", "text": "Avoid tobacco products and limit alcohol consumption"},
            {"priority": "low", "text": "Report any new symptoms to your healthcare provider"}
        ]
    
    # Add patient-specific recommendations
    if patient_info.get("tobacco_use") == "Current":
        recommendations.insert(1, {"priority": "high", "text": "Consider tobacco cessation programs - tobacco significantly increases oral cancer risk"})
    
    if patient_info.get("symptoms") and len(patient_info["symptoms"]) > 0:
        recommendations.insert(0, {"priority": "high", "text": f"Discuss your symptoms ({', '.join(patient_info['symptoms'])}) with a healthcare provider"})
    
    state.recommendations = recommendations
    return state


def finalize_node(state: AnalysisState, config: dict) -> AnalysisState:
    """
    Final node that marks the analysis as complete.
    """
    state.analysis_complete = True
    return state
