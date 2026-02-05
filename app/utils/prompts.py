"""
LLM Prompt Templates
"""
from typing import Optional


def get_analysis_prompt(
    prediction: dict,
    patient_info: Optional[dict] = None
) -> str:
    """
    Generate the analysis prompt for the LLM.
    
    Args:
        prediction: ML model prediction results
        patient_info: Optional patient information
        
    Returns:
        Formatted prompt string
    """
    # Build prediction summary
    class_probs = prediction.get("class_probabilities", {})
    pred_class = prediction.get("predicted_class", "Unknown")
    confidence = prediction.get("confidence", 0)
    
    prob_summary = "\n".join([
        f"- {name}: {prob*100:.1f}%"
        for name, prob in sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
    ])
    
    # Build patient info summary
    patient_summary = "No patient information provided."
    if patient_info:
        info_parts = []
        if patient_info.get("age"):
            info_parts.append(f"Age: {patient_info['age']}")
        if patient_info.get("gender") and patient_info["gender"] != "Not specified":
            info_parts.append(f"Gender: {patient_info['gender']}")
        if patient_info.get("tobacco_use") and patient_info["tobacco_use"] != "Not specified":
            info_parts.append(f"Tobacco use: {patient_info['tobacco_use']}")
        if patient_info.get("alcohol_use") and patient_info["alcohol_use"] != "Not specified":
            info_parts.append(f"Alcohol consumption: {patient_info['alcohol_use']}")
        if patient_info.get("symptoms"):
            info_parts.append(f"Symptoms: {', '.join(patient_info['symptoms'])}")
        if patient_info.get("duration") and patient_info["duration"] != "Not specified":
            info_parts.append(f"Symptom duration: {patient_info['duration']}")
        
        if info_parts:
            patient_summary = "\n".join(info_parts)
    
    prompt = f"""
Analyze the following oral lesion image classification results and provide a detailed, professional analysis.

## ML Model Classification Results:
Primary Prediction: {pred_class} ({confidence*100:.1f}% confidence)

Class Probabilities:
{prob_summary}

## Patient Information:
{patient_summary}

## Please provide:
1. **Clinical Interpretation**: Explain what the classification results suggest about the lesion
2. **Key Observations**: Based on the confidence levels and class probabilities, what patterns are notable?
3. **Risk Factor Analysis**: If patient information is available, how do their risk factors relate to the findings?
4. **Important Considerations**: What should the healthcare provider consider when reviewing these results?

Please maintain a balanced, professional tone. Do not cause unnecessary alarm, but be clear about any concerning findings.
Remember to emphasize that AI analysis supplements but does not replace professional medical evaluation.
"""
    
    return prompt


def get_recommendation_prompt(
    prediction: dict,
    risk_level: str,
    patient_info: Optional[dict] = None
) -> str:
    """
    Generate the recommendations prompt for the LLM.
    
    Args:
        prediction: ML model prediction results
        risk_level: Assessed risk level (Low/Moderate/High)
        patient_info: Optional patient information
        
    Returns:
        Formatted prompt string
    """
    pred_class = prediction.get("predicted_class", "Unknown")
    confidence = prediction.get("confidence", 0)
    
    # Build patient context
    risk_factors = []
    if patient_info:
        if patient_info.get("tobacco_use") == "Current":
            risk_factors.append("active tobacco use")
        if patient_info.get("alcohol_use") in ["Regular", "Heavy"]:
            risk_factors.append("significant alcohol consumption")
        if patient_info.get("duration") == "More than 3 months":
            risk_factors.append("prolonged symptom duration")
        if patient_info.get("age") and patient_info["age"] > 50:
            risk_factors.append("age over 50")
    
    risk_factors_text = ", ".join(risk_factors) if risk_factors else "none identified"
    
    prompt = f"""
Based on the following analysis, generate specific, actionable recommendations:

## Analysis Summary:
- Predicted Classification: {pred_class}
- Confidence: {confidence*100:.1f}%
- Overall Risk Level: {risk_level}
- Identified Risk Factors: {risk_factors_text}

## Generate recommendations in the following JSON format:
[
    {{"priority": "high|medium|low", "text": "Specific recommendation"}},
    ...
]

## Guidelines:
- For HIGH risk: Include urgent referral recommendations and immediate follow-up actions
- For MODERATE risk: Include monitoring recommendations and scheduled follow-up
- For LOW risk: Include preventive care and routine monitoring recommendations
- Always include at least one recommendation about professional medical consultation
- Make recommendations specific and actionable
- Consider patient risk factors when applicable
- Limit to 4-6 most important recommendations

Return ONLY the JSON array, no additional text.
"""
    
    return prompt


def get_follow_up_prompt(
    previous_analysis: str,
    new_observation: str
) -> str:
    """
    Generate a follow-up question prompt.
    
    Args:
        previous_analysis: Previous analysis text
        new_observation: New observation or question
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""
Based on the previous analysis:

{previous_analysis}

The user has a follow-up question or observation:
{new_observation}

Please provide a helpful response that:
1. Addresses their specific question
2. Relates to the original analysis when relevant
3. Maintains appropriate medical disclaimer where necessary
4. Suggests professional consultation if the question requires clinical judgment
"""
    
    return prompt


def get_comparison_prompt(
    current_prediction: dict,
    previous_prediction: dict
) -> str:
    """
    Generate a comparison analysis prompt for follow-up images.
    
    Args:
        current_prediction: Current image prediction results
        previous_prediction: Previous image prediction results
        
    Returns:
        Formatted prompt string
    """
    current_class = current_prediction.get("predicted_class", "Unknown")
    current_conf = current_prediction.get("confidence", 0)
    previous_class = previous_prediction.get("predicted_class", "Unknown")
    previous_conf = previous_prediction.get("confidence", 0)
    
    prompt = f"""
Compare the following two analysis results from different time points:

## Previous Analysis:
- Classification: {previous_class}
- Confidence: {previous_conf*100:.1f}%

## Current Analysis:
- Classification: {current_class}
- Confidence: {current_conf*100:.1f}%

Please provide:
1. **Change Analysis**: Has there been any notable change between the two assessments?
2. **Trend Interpretation**: What might these changes (or lack thereof) suggest?
3. **Clinical Significance**: Are these changes clinically meaningful?
4. **Recommendations**: What follow-up actions should be considered based on this comparison?

Remember to emphasize that AI-based comparison has limitations and professional evaluation is essential.
"""
    
    return prompt
