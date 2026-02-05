"""
Results Display Component - Professional Design
"""
import streamlit as st
from typing import Any


def render_results(result: Any, config: dict) -> None:
    """
    Render the analysis results with professional visualizations.
    """
    if result is None:
        return
    
    # Extract data from result
    if hasattr(result, 'prediction'):
        prediction = result.prediction if result.prediction else {}
        analysis = result.analysis if hasattr(result, 'analysis') else ""
        recommendations = result.recommendations if hasattr(result, 'recommendations') else []
        risk_level = result.risk_level if hasattr(result, 'risk_level') else "Unknown"
    else:
        prediction = result.get("prediction", {})
        analysis = result.get("analysis", "")
        recommendations = result.get("recommendations", [])
        risk_level = result.get("risk_level", "Unknown")
    
    # Risk Level Configuration
    risk_config = {
        "Low": {
            "color": "#22c55e",
            "bg": "#f0fdf4",
            "border": "#22c55e",
            "icon": "fa-check-circle"
        },
        "Moderate": {
            "color": "#f59e0b",
            "bg": "#fffbeb",
            "border": "#fbbf24",
            "icon": "fa-exclamation-circle"
        },
        "High": {
            "color": "#ef4444",
            "bg": "#fef2f2",
            "border": "#ef4444",
            "icon": "fa-exclamation-triangle"
        },
        "Unknown": {
            "color": "#6b7280",
            "bg": "#f3f4f6",
            "border": "#d1d5db",
            "icon": "fa-question-circle"
        }
    }
    
    rc = risk_config.get(risk_level, risk_config["Unknown"])
    
    # Risk Level Card
    st.markdown(f"""
    <div style="
        background: {rc['bg']};
        border: 1px solid {rc['border']};
        border-left: 4px solid {rc['color']};
        padding: 1.25rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    ">
        <div style="display: flex; align-items: center; gap: 12px;">
            <i class="fas {rc['icon']}" style="font-size: 1.5rem; color: {rc['color']};"></i>
            <div>
                <div style="font-size: 0.8rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px;">Risk Assessment</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: {rc['color']};">{risk_level} Risk</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Classification Results
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <div class="card-icon">
                <i class="fas fa-chart-pie"></i>
            </div>
            <h4 class="card-title">Classification Results</h4>
        </div>
    """, unsafe_allow_html=True)
    
    class_probs = prediction.get("class_probabilities", {})
    
    if class_probs:
        for class_name, prob in sorted(class_probs.items(), key=lambda x: x[1], reverse=True):
            # Color based on probability
            if prob >= 0.7:
                bar_color = "#ef4444" if class_name != "Normal" else "#22c55e"
            elif prob >= 0.4:
                bar_color = "#f59e0b"
            else:
                bar_color = "#94a3b8"
            
            st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.35rem;">
                    <span style="font-size: 0.9rem; color: #374151; font-weight: 500;">{class_name}</span>
                    <span style="font-size: 0.9rem; font-weight: 600; color: {bar_color};">{prob*100:.1f}%</span>
                </div>
                <div style="
                    background: #e5e7eb;
                    border-radius: 6px;
                    height: 8px;
                    overflow: hidden;
                ">
                    <div style="
                        background: {bar_color};
                        height: 100%;
                        width: {prob*100}%;
                        border-radius: 6px;
                        transition: width 0.5s ease;
                    "></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <p style="color: #6b7280; font-size: 0.9rem;">No classification data available</p>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # AI Analysis
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <div class="card-icon">
                <i class="fas fa-brain"></i>
            </div>
            <h4 class="card-title">AI Analysis</h4>
        </div>
    """, unsafe_allow_html=True)
    
    if analysis:
        st.markdown(f"""
        <div style="
            background: #f8fafc;
            padding: 1rem;
            border-radius: 8px;
            font-size: 0.9rem;
            color: #4a5568;
            line-height: 1.6;
        ">
            {analysis}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <p style="color: #6b7280; font-size: 0.9rem;">Analysis could not be generated. Please review classification results.</p>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <div class="card-icon">
                <i class="fas fa-clipboard-list"></i>
            </div>
            <h4 class="card-title">Recommendations</h4>
        </div>
    """, unsafe_allow_html=True)
    
    if recommendations:
        priority_icons = {
            "high": ("fa-exclamation-circle", "#ef4444"),
            "medium": ("fa-info-circle", "#f59e0b"),
            "low": ("fa-check-circle", "#22c55e")
        }
        
        for rec in recommendations:
            if isinstance(rec, dict):
                priority = rec.get("priority", "medium").lower()
                text = rec.get("text", "")
            else:
                priority = "medium"
                text = str(rec)
            
            icon, color = priority_icons.get(priority, priority_icons["medium"])
            
            st.markdown(f"""
            <div style="
                background: #f8fafc;
                padding: 0.875rem 1rem;
                border-radius: 8px;
                margin-bottom: 0.5rem;
                display: flex;
                align-items: flex-start;
                gap: 10px;
                border-left: 3px solid {color};
            ">
                <i class="fas {icon}" style="color: {color}; margin-top: 2px;"></i>
                <span style="font-size: 0.9rem; color: #374151;">{text}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <p style="color: #6b7280; font-size: 0.9rem;">No specific recommendations available.</p>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Download Report Button
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    
    if st.button("Download Report", key="download_report"):
        report = generate_report(result, config)
        st.download_button(
            label="Save Report as Text File",
            data=report,
            file_name="oral_cancer_analysis_report.txt",
            mime="text/plain",
            key="download_txt"
        )


def generate_report(result: Any, config: dict) -> str:
    """Generate a text report from the analysis results."""
    
    if hasattr(result, 'prediction'):
        prediction = result.prediction if result.prediction else {}
        analysis = result.analysis if hasattr(result, 'analysis') else ""
        recommendations = result.recommendations if hasattr(result, 'recommendations') else []
        risk_level = result.risk_level if hasattr(result, 'risk_level') else "Unknown"
    else:
        prediction = result.get("prediction", {})
        analysis = result.get("analysis", "")
        recommendations = result.get("recommendations", [])
        risk_level = result.get("risk_level", "Unknown")
    
    report = f"""
================================================================================
                    ORAL CANCER DETECTION AI - ANALYSIS REPORT
================================================================================

Generated by: {config['app']['title']} v{config['app']['version']}

--------------------------------------------------------------------------------
RISK ASSESSMENT
--------------------------------------------------------------------------------
Overall Risk Level: {risk_level}

--------------------------------------------------------------------------------
CLASSIFICATION RESULTS
--------------------------------------------------------------------------------
"""
    
    class_probs = prediction.get("class_probabilities", {})
    if class_probs:
        for class_name, prob in sorted(class_probs.items(), key=lambda x: x[1], reverse=True):
            report += f"  - {class_name}: {prob*100:.1f}%\n"
    
    report += f"""
--------------------------------------------------------------------------------
AI ANALYSIS
--------------------------------------------------------------------------------
{analysis if analysis else 'Analysis not available'}

--------------------------------------------------------------------------------
RECOMMENDATIONS
--------------------------------------------------------------------------------
"""
    
    for i, rec in enumerate(recommendations, 1):
        if isinstance(rec, dict):
            text = rec.get('text', str(rec))
            priority = rec.get('priority', '').upper()
            report += f"  {i}. [{priority}] {text}\n"
        else:
            report += f"  {i}. {rec}\n"
    
    report += """
--------------------------------------------------------------------------------
DISCLAIMER
--------------------------------------------------------------------------------
This report is generated by an AI system and is intended for informational 
purposes only. It should not be used as a substitute for professional medical 
diagnosis. Please consult with a qualified healthcare provider for proper 
diagnosis and treatment.

================================================================================
"""
    
    return report
