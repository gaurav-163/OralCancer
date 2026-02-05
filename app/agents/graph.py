"""
LangGraph Workflow Definition
"""
from typing import Any
from langgraph.graph import StateGraph, END
from functools import partial

from .state import AnalysisState
from .nodes import (
    analyze_prediction_node,
    assess_risk_node,
    generate_recommendations_node,
    finalize_node
)


def create_analysis_graph(config: dict) -> StateGraph:
    """
    Create the LangGraph workflow for oral cancer analysis.
    
    The workflow consists of the following steps:
    1. Analyze Prediction - LLM analyzes the ML model output
    2. Assess Risk - Calculate overall risk level
    3. Generate Recommendations - Create actionable recommendations
    4. Finalize - Mark analysis as complete
    
    Args:
        config: Application configuration dictionary
        
    Returns:
        Compiled StateGraph ready for invocation
    """
    # Create the graph with AnalysisState as the state schema
    workflow = StateGraph(AnalysisState)
    
    # Add nodes with config bound
    workflow.add_node(
        "analyze_prediction",
        partial(analyze_prediction_node, config=config)
    )
    workflow.add_node(
        "assess_risk",
        partial(assess_risk_node, config=config)
    )
    workflow.add_node(
        "generate_recommendations",
        partial(generate_recommendations_node, config=config)
    )
    workflow.add_node(
        "finalize",
        partial(finalize_node, config=config)
    )
    
    # Define the workflow edges
    workflow.set_entry_point("analyze_prediction")
    
    workflow.add_edge("analyze_prediction", "assess_risk")
    workflow.add_edge("assess_risk", "generate_recommendations")
    workflow.add_edge("generate_recommendations", "finalize")
    workflow.add_edge("finalize", END)
    
    # Compile the graph
    compiled_graph = workflow.compile()
    
    return compiled_graph


def run_analysis(graph: StateGraph, initial_state: AnalysisState) -> dict:
    """
    Run the analysis workflow.
    
    Args:
        graph: Compiled LangGraph workflow
        initial_state: Initial state with image and prediction
        
    Returns:
        Dictionary containing final analysis results
    """
    try:
        # Invoke the graph
        final_state = graph.invoke(initial_state)
        
        # Convert to output dictionary
        if isinstance(final_state, AnalysisState):
            return final_state.to_dict()
        elif isinstance(final_state, dict):
            return final_state
        else:
            return {"error": "Unexpected state type"}
            
    except Exception as e:
        return {
            "error": str(e),
            "analysis_complete": False,
            "risk_level": "Unknown",
            "analysis": "Analysis failed. Please try again or consult a healthcare professional.",
            "recommendations": [
                {"priority": "high", "text": "Please consult with a healthcare professional"}
            ]
        }
