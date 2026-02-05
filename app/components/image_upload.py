"""
Image Upload Component - Professional Design
"""
import streamlit as st
from PIL import Image


def render_image_upload(config: dict) -> Image.Image | None:
    """
    Render the image upload component with professional styling.
    """
    max_size_mb = config["ui"]["max_upload_size_mb"]
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help=f"Upload a clear image of the oral cavity. Max size: {max_size_mb}MB",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Check file size
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        
        if file_size_mb > max_size_mb:
            st.markdown(f"""
            <div class="alert alert-error">
                <i class="fas fa-exclamation-circle alert-icon"></i>
                <div class="alert-content">
                    File size ({file_size_mb:.2f}MB) exceeds maximum allowed ({max_size_mb}MB)
                </div>
            </div>
            """, unsafe_allow_html=True)
            return None
        
        try:
            # Load image
            image = Image.open(uploaded_file)
            
            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Display the uploaded image in a card
            st.markdown("""
            <div class="card">
                <div class="card-header">
                    <div class="card-icon">
                        <i class="fas fa-image"></i>
                    </div>
                    <h4 class="card-title">Uploaded Image</h4>
                </div>
            """, unsafe_allow_html=True)
            
            st.image(image, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Image metadata
            st.markdown(f"""
            <div style="display: flex; gap: 1rem; margin-top: 1rem;">
                <div class="metric-card" style="flex: 1;">
                    <div class="metric-value">{image.width}</div>
                    <div class="metric-label">Width (px)</div>
                </div>
                <div class="metric-card" style="flex: 1;">
                    <div class="metric-value">{image.height}</div>
                    <div class="metric-label">Height (px)</div>
                </div>
                <div class="metric-card" style="flex: 1;">
                    <div class="metric-value">{file_size_mb:.1f}</div>
                    <div class="metric-label">Size (MB)</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            return image
            
        except Exception as e:
            st.markdown(f"""
            <div class="alert alert-error">
                <i class="fas fa-times-circle alert-icon"></i>
                <div class="alert-content">Error loading image: {str(e)}</div>
            </div>
            """, unsafe_allow_html=True)
            return None
    
    else:
        # Upload placeholder
        st.markdown("""
        <div class="upload-area">
            <div class="upload-icon">
                <i class="fas fa-cloud-upload-alt"></i>
            </div>
            <p class="upload-text">
                Drag and drop an image here, or click to browse
            </p>
            <p class="upload-hint">
                Supported formats: JPG, JPEG, PNG, BMP, WebP
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        return None
