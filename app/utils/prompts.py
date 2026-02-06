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

- Make recommendations specific and actionable
- Consider patient risk factors when applicable
- Limit to 4-6 most important recommendations

Return ONLY the JSON array, no additional text.
"""
    
    return prompt


def get_system_prompt(patient_name: Optional[str] = None) -> str:
    """
    Return a concise system prompt template for clinical analysis that addresses
    the patient by name. Use `{{patient_name}}` placeholder internally; callers
    should substitute or pass `patient_name` to this helper.
    """
    name = patient_name or "{{patient_name}}"
    prompt = f"""
You are a concise, evidence-oriented medical assistant specializing in oral pathology. Address the patient directly by name and keep the output clinical and actionable.

- Begin with: "ShortSummary: Patient: {name} —"
- Provide a one-line clinical summary of the model prediction and confidence.
- Provide 2 short bullets under "Observations" describing key visual features supporting the classification.
- Provide 1–3 numbered "Recommendations" with clear timeframes (urgent vs routine).
- Provide one short "Rationale" sentence explaining the reasoning.
- If clinical details are missing (history, symptom duration, biopsy), state clearly what is needed.
- Do NOT give definitive diagnoses; use neutral phrasing such as "suggests", "consistent with", "warrants evaluation".
- Keep total length ≤ 150 words and use the exact output sections below.

Output format (strict; return only these sections):
ShortSummary: <one-line summary>
Observations:
- <bullet 1>
- <bullet 2>
Recommendations:
1. <text with timeframe>
2. <optional additional item>
Rationale: <one short sentence>

This is an AI screening tool and not a diagnosis.
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
3. Suggests professional consultation if the question requires clinical judgment
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


"""
    
    return prompt
