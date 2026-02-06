"""
Sidebar Component - Professional Design
"""
import streamlit as st


def render_sidebar(config: dict) -> dict:
    """
    Render the sidebar with settings and patient information.
    """
    settings = {}
    
    with st.sidebar:
        # Sidebar Header
        st.markdown("""
        <div style="padding: 1rem 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <i class="fas fa-cog" style="color: #667eea; font-size: 1.2rem;"></i>
                <span style="font-size: 1.1rem; font-weight: 600; color: #1a202c;">Settings</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence threshold is hardcoded in the predictor (hidden from UI)
        # (No UI control shown.)
        
        # Patient Information Section
        st.markdown("""
        <div style="padding: 0.75rem 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <i class="fas fa-user-circle" style="color: #667eea; font-size: 1.2rem;"></i>
                <span style="font-size: 1.1rem; font-weight: 600; color: #1a202c;">Patient Information</span>
            </div>
            <p style="font-size: 0.75rem; color: #718096; margin: 0.25rem 0 0 0;">Optional - Helps provide more personalized analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        patient_info = {}
        
        # Name
        st.markdown("""
        <span style="font-size: 0.8rem; font-weight: 500; color: #4a5568;">
            <i class="fas fa-user" style="margin-right: 6px; color: #94a3b8;"></i>Name
        </span>
        """, unsafe_allow_html=True)
        patient_info["name"] = st.text_input(
            "Name",
            placeholder="Enter patient name",
            label_visibility="collapsed"
        )
        
        # Age
        st.markdown("""
        <span style="font-size: 0.8rem; font-weight: 500; color: #4a5568;">
            <i class="fas fa-calendar-alt" style="margin-right: 6px; color: #94a3b8;"></i>Age
        </span>
        """, unsafe_allow_html=True)
        patient_info["age"] = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=None,
            placeholder="Enter age",
            label_visibility="collapsed"
        )
        
        # Gender
        st.markdown("""
        <span style="font-size: 0.8rem; font-weight: 500; color: #4a5568;">
            <i class="fas fa-venus-mars" style="margin-right: 6px; color: #94a3b8;"></i>Gender
        </span>
        """, unsafe_allow_html=True)
        patient_info["gender"] = st.selectbox(
            "Gender",
            options=["Not specified", "Male", "Female", "Other"],
            label_visibility="collapsed"
        )
        
        # Tobacco Use
        st.markdown("""
        <span style="font-size: 0.8rem; font-weight: 500; color: #4a5568;">
            <i class="fas fa-smoking" style="margin-right: 6px; color: #94a3b8;"></i>Tobacco Use
        </span>
        """, unsafe_allow_html=True)
        patient_info["tobacco_use"] = st.selectbox(
            "Tobacco Use",
            options=["Not specified", "Never", "Former", "Current"],
            label_visibility="collapsed"
        )
        
        # Alcohol Use
        st.markdown("""
        <span style="font-size: 0.8rem; font-weight: 500; color: #4a5568;">
            <i class="fas fa-wine-glass-alt" style="margin-right: 6px; color: #94a3b8;"></i>Alcohol Consumption
        </span>
        """, unsafe_allow_html=True)
        patient_info["alcohol_use"] = st.selectbox(
            "Alcohol Consumption",
            options=["Not specified", "Never", "Occasional", "Regular", "Heavy"],
            label_visibility="collapsed"
        )
        
        # Symptoms
        st.markdown("""
        <span style="font-size: 0.8rem; font-weight: 500; color: #4a5568;">
            <i class="fas fa-notes-medical" style="margin-right: 6px; color: #94a3b8;"></i>Current Symptoms
        </span>
        """, unsafe_allow_html=True)
        patient_info["symptoms"] = st.multiselect(
            "Current Symptoms",
            options=[
                "Pain or discomfort",
                "Difficulty swallowing",
                "White/red patches",
                "Non-healing sore",
                "Lump or thickening",
                "Numbness",
                "Loose teeth",
                "Voice changes"
            ],
            label_visibility="collapsed"
        )
        
        # Duration
        st.markdown("""
        <span style="font-size: 0.8rem; font-weight: 500; color: #4a5568;">
            <i class="fas fa-clock" style="margin-right: 6px; color: #94a3b8;"></i>Symptom Duration
        </span>
        """, unsafe_allow_html=True)
        patient_info["duration"] = st.selectbox(
            "Symptom Duration",
            options=["Not specified", "Less than 2 weeks", "2-4 weeks", "1-3 months", "More than 3 months"],
            label_visibility="collapsed"
        )
        
        settings["patient_info"] = patient_info
        
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        
        # About Section
        st.markdown("""
        <div style="padding: 0.75rem 0; border-top: 1px solid #e2e8f0;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.75rem;">
                <i class="fas fa-info-circle" style="color: #667eea; font-size: 1rem;"></i>
                <span style="font-size: 0.95rem; font-weight: 600; color: #1a202c;">About</span>
            </div>
            <div style="background: #f0f4ff; padding: 1rem; border-radius: 8px; font-size: 0.8rem; color: #4a5568;">
                <strong style="color: #1a202c;">{title}</strong><br><br>
                Version: {version}<br><br>
                This AI tool uses deep learning to analyze oral cavity images and identify potential lesions.
            </div>
        </div>
        """.format(title=config['app']['title'], version=config['app']['version']), unsafe_allow_html=True)
        
        # Usage Guide
        with st.expander("How to Use", expanded=False):
            st.markdown("""
            <div style="font-size: 0.85rem; color: #4a5568; line-height: 1.6;">
                <p><strong>1. Upload an Image</strong><br>
                Use the upload area to submit a clear photo of the oral lesion</p>
                
                <p><strong>2. Provide Patient Info</strong><br>
                (Optional) Fill in patient details for better analysis</p>
                
                <p><strong>3. Analyze</strong><br>
                Click the analyze button to get AI predictions</p>
                
                <p><strong>4. Review Results</strong><br>
                Examine the analysis and recommendations</p>
                
                <p><strong>5. Consult a Professional</strong><br>
                Always verify with a healthcare provider</p>
            </div>
            """, unsafe_allow_html=True)
    
    return settings
