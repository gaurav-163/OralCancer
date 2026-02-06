"""
Oral Cancer Detection AI - Professional Streamlit Application
"""
import streamlit as st
import yaml
from pathlib import Path
from dotenv import load_dotenv
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from components import render_sidebar, render_image_upload, render_results
from agents import create_analysis_graph, AnalysisState
from model import OralCancerPredictor
from utils import preprocess_image, validate_image

# Load environment variables
load_dotenv()

# Load configuration
def load_config():
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Page Configuration
st.set_page_config(
    page_title=config["app"]["title"],
    page_icon="assets/favicon.ico" if Path("assets/favicon.ico").exists() else None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Font Awesome and Custom CSS
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }
    
    /* Header Styles */
    .app-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    }
    
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .app-subtitle {
        color: #a0aec0;
        font-size: 1rem;
        margin: 0;
        font-weight: 400;
    }
    
    .app-icon {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.5rem;
    }
    
    /* Card Styles */
    .card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    .card-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a202c !important;
        margin: 0;
    }
    
    .card-icon {
        width: 32px;
        height: 32px;
        background: #f0f4ff;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #667eea;
        font-size: 0.9rem;
    }
    
    /* Alert Styles */
    .alert {
        padding: 1rem 1.25rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    }
    
    .alert-warning {
        background: #fffbeb;
        border: 1px solid #fbbf24;
        color: #92400e;
    }
    
    .alert-info {
        background: #eff6ff;
        border: 1px solid #3b82f6;
        color: #1e40af;
    }
    
    .alert-success {
        background: #f0fdf4;
        border: 1px solid #22c55e;
        color: #166534;
    }
    
    .alert-error {
        background: #fef2f2;
        border: 1px solid #ef4444;
        color: #991b1b;
    }
    
    .alert-icon {
        font-size: 1.1rem;
        margin-top: 2px;
    }
    
    .alert-content {
        flex: 1;
    }
    
    .alert-title {
        font-weight: 600;
        margin-bottom: 4px;
    }
    
    /* Button Styles */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.875rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.35);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.45);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Upload Area */
    .upload-area {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        background: #f8fafc;
        transition: all 0.2s ease;
    }
    
    .upload-area:hover {
        border-color: #667eea;
        background: #f0f4ff;
    }
    
    .upload-icon {
        font-size: 3rem;
        color: #94a3b8;
        margin-bottom: 1rem;
    }
    
    .upload-text {
        color: #64748b;
        font-size: 0.95rem;
    }
    
    .upload-hint {
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }
    
    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1rem;
    }
    
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1a202c;
        margin: 0;
    }
    
    .section-icon {
        color: #667eea;
        font-size: 1.1rem;
    }
    
    /* Footer */
    .app-footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.85rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 3rem;
    }
    
    /* Force dark text for content inside white/light containers */
    .card, .card *, .metric-card, .metric-card *, .upload-area, .upload-area * {
        color: #1a202c !important;
    }

    .metric-card {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar */
    .css-1d391kg {
        background: #f8fafc;
    }
    
    section[data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if "predictor" not in st.session_state:
        st.session_state.predictor = OralCancerPredictor(config)
    if "analysis_graph" not in st.session_state:
        st.session_state.analysis_graph = create_analysis_graph(config)
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None


def main():
    """Main application entry point"""
    initialize_session_state()
    
    # Render sidebar
    settings = render_sidebar(config)
    
    # Header
    st.markdown("""
    <div class="app-header">
        <div class="app-title">
            <div class="app-icon">
                <i class="fas fa-microscope"></i>
            </div>
            Oral Cancer Detection AI
        </div>
        <p class="app-subtitle">Advanced AI-powered screening and risk assessment for oral lesions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Medical Disclaimer

    
    # Two-column layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div class="section-header">
            <i class="fas fa-cloud-upload-alt section-icon"></i>
            <h3 class="section-title">Upload Image</h3>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_image = render_image_upload(config)
        
        if uploaded_image is not None:
            st.session_state.uploaded_image = uploaded_image
            
            # Validate
            is_valid, message = validate_image(uploaded_image)
            if not is_valid:
                st.markdown(f"""
                <div class="alert alert-error">
                    <i class="fas fa-times-circle alert-icon"></i>
                    <div class="alert-content">{message}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert alert-success">
                    <i class="fas fa-check-circle alert-icon"></i>
                    <div class="alert-content">Image uploaded and validated successfully</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Analyze button
                if st.button("Analyze Image", key="analyze_btn"):
                    # Step 1: Keras Model Prediction
                    with st.spinner("🧠 Running Keras model prediction..."):
                        # Preprocess
                        processed_image = preprocess_image(
                            uploaded_image, 
                            target_size=tuple(config["model"]["input_size"])
                        )
                        
                        # Get ML prediction from Keras model (gauravvv7/Oralcancer)
                        prediction = st.session_state.predictor.predict(processed_image)
                        st.success(f"✅ Model prediction: {prediction['predicted_class']} ({prediction['confidence']:.1%})")
                    
                    # Step 2: Cohere AI Analysis
                    with st.spinner("🤖 Generating AI analysis with Cohere..."):
                        
                        # Create analysis state
                        initial_state = AnalysisState(
                            image=processed_image,
                            prediction=prediction,
                            patient_info=settings.get("patient_info", {}),
                            analysis_complete=False
                        )
                        
                        # Run LangGraph workflow
                        try:
                            result = st.session_state.analysis_graph.invoke(initial_state)
                            st.session_state.analysis_result = result
                        except Exception as e:
                            # Fallback if LLM fails
                            st.session_state.analysis_result = {
                                "prediction": prediction,
                                "risk_level": "Moderate" if prediction.get("confidence", 0) > 0.5 else "Low",
                                "analysis": "AI analysis could not be completed. Please review the classification results.",
                                "recommendations": [
                                    {"priority": "high", "text": "Review classification confidence levels"},
                                    {"priority": "medium", "text": "Check for visible abnormalities"},
                                    {"priority": "low", "text": "Monitor for changes over time"}
                                ]
                            }
    
    with col2:
        st.markdown("""
        <div class="section-header">
            <i class="fas fa-chart-bar section-icon"></i>
            <h3 class="section-title">Analysis Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.analysis_result is not None:
            render_results(st.session_state.analysis_result, config)
        else:
            st.markdown("""
            <div class="alert alert-info">
                <i class="fas fa-info-circle alert-icon"></i>
                <div class="alert-content">Upload an image and click 'Analyze' to see the results</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown(f"""
    <div class="app-footer">
        <p>{config['app']['title']} v{config['app']['version']} | Built with Streamlit and LangGraph</p>

    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
