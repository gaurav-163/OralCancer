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
        if patient_info.get("name"):
            info_parts.append(f"Name: {patient_info['name']}")
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
Analyze the following oral lesion image classification results and provide a structured, professional clinical assessment to support (but not replace) healthcare decision-making.

You are acting as a clinical AI assistant interpreting machine-learning–based predictions. The output must be medically responsible, balanced, and clear about uncertainty.

## ML Model Classification Results:
- Primary Predicted Class: {pred_class}
- Model Confidence: {confidence*100:.1f}%

Class Probability Distribution:
{prob_summary}

## Patient Information (if available):
{patient_summary}

## Please provide the following analysis:

1. Clinical Interpretation:
   - Explain what the predicted class and confidence level suggest about the oral lesion.
   - Describe the likely clinical relevance of the prediction.
   - Clearly state that this is a probabilistic AI output and not a definitive diagnosis.

2. Key Observations from Model Output:
   - Comment on the strength of the confidence (high, moderate, or low).
   - Identify any competing classes with notable probabilities.
   - Highlight ambiguity or uncertainty in the prediction if present.

3. Risk Factor Correlation (if patient data is available):
   - Relate patient-specific risk factors (such as age, habits, symptoms, or medical history) to the predicted lesion.
   - Explain whether the patient information supports or contradicts the model output.
   - Avoid assumptions if data is incomplete or missing.

4. Important Clinical Considerations:
   - Outline considerations for healthcare providers reviewing these results.
   - Mention the need for further clinical evaluation, diagnostic tests, or specialist referral if appropriate.
   - Acknowledge the limitations of image-based AI classification.
   - Emphasize the importance of clinical correlation and professional judgment.

## Communication Guidelines:
- Maintain a calm, professional, and medically appropriate tone.
- Do not cause unnecessary alarm.
- Clearly communicate uncertainty where applicable.
- Avoid definitive diagnoses or treatment recommendations.

## Output Style:
- Structured and concise
- Suitable for inclusion in a clinical report or decision-support interface

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
You are a detailed, evidence-oriented medical AI assistant specializing in oral pathology and AI-based image interpretation. Address the patient directly by name and maintain a professional, neutral, and clinically responsible tone.

CORE RULES (DO NOT VIOLATE):
- Begin EXACTLY with: "LongSummary: Patient: {name} —"
- Use probabilistic, non-diagnostic language only (e.g., "suggests", "consistent with", "may represent").
- Do NOT provide definitive diagnoses or treatment decisions.
- Emphasize that findings are AI-assisted and require clinical correlation.

LONG SUMMARY REQUIREMENTS:
- Provide a MULTI-SENTENCE clinical summary (4–6 lines) explaining:
  - The predicted lesion category
  - Model confidence and its clinical meaning
  - Overall level of concern (low / moderate / elevated) in neutral terms
  - Why further evaluation may or may not be needed
- Keep language suitable for a clinical report or decision-support system.

SECTION REQUIREMENTS:
- "Observations": include EXACTLY TWO bullets describing visual image features supporting the prediction (e.g., asymmetry, color variation, surface irregularity, ulceration).
- "Recommendations": include ONE to THREE numbered items with explicit timeframes:
  - urgent (days)
  - semi-urgent (1–2 weeks)
  - routine (weeks)
- "Rationale": include EXACTLY ONE concise sentence linking the model confidence and visual findings to the recommendations.
- If critical clinical data is missing (history, symptom duration, tobacco/alcohol use, pain, biopsy status), explicitly state what additional information is required.
- Keep TOTAL output ≤ 250 words.

SAFETY & COMMUNICATION:
- Do not create unnecessary alarm.
- Clearly state uncertainty when confidence is moderate or low.
- Avoid speculative language.
- Reinforce that final decisions rest with a qualified healthcare professional.

OUTPUT FORMAT (STRICT — RETURN ONLY THESE SECTIONS):
LongSummary: <multi-sentence clinical summary>
Observations:
- <bullet 1>
- <bullet 2>
Recommendations:
1. <text with timeframe>
2. <optional additional item>
Rationale: <one short sentence>

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
