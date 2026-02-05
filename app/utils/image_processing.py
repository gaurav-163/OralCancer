"""
Image Processing Utilities
"""
import numpy as np
from PIL import Image
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def preprocess_image(
    image: Image.Image,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Preprocess image for model input.
    
    Args:
        image: PIL Image object
        target_size: Target size (width, height)
        
    Returns:
        Preprocessed numpy array ready for model input
    """
    try:
        # Ensure RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize with high-quality resampling
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0
        
        return img_array
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise


def validate_image(image: Image.Image) -> Tuple[bool, str]:
    """
    Validate uploaded image for processing.
    
    Args:
        image: PIL Image object
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        # Check if image is None
        if image is None:
            return False, "No image provided"
        
        # Check minimum dimensions
        min_size = 100
        if image.width < min_size or image.height < min_size:
            return False, f"Image too small. Minimum size is {min_size}x{min_size} pixels"
        
        # Check maximum dimensions
        max_size = 10000
        if image.width > max_size or image.height > max_size:
            return False, f"Image too large. Maximum size is {max_size}x{max_size} pixels"
        
        # Check aspect ratio (shouldn't be too extreme)
        aspect_ratio = max(image.width, image.height) / min(image.width, image.height)
        if aspect_ratio > 10:
            return False, "Image aspect ratio is too extreme. Please use a more balanced image"
        
        # Check color mode
        if image.mode not in ["RGB", "RGBA", "L", "LA"]:
            return False, f"Unsupported image mode: {image.mode}"
        
        return True, "Image is valid"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def enhance_image(
    image: Image.Image,
    brightness: float = 1.0,
    contrast: float = 1.0,
    sharpness: float = 1.0
) -> Image.Image:
    """
    Apply basic image enhancements.
    
    Args:
        image: PIL Image object
        brightness: Brightness factor (1.0 = original)
        contrast: Contrast factor (1.0 = original)
        sharpness: Sharpness factor (1.0 = original)
        
    Returns:
        Enhanced PIL Image
    """
    from PIL import ImageEnhance
    
    # Apply brightness
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
    
    # Apply contrast
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
    
    # Apply sharpness
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness)
    
    return image


def crop_center(image: Image.Image, crop_size: Tuple[int, int]) -> Image.Image:
    """
    Crop the center of an image.
    
    Args:
        image: PIL Image object
        crop_size: Size to crop (width, height)
        
    Returns:
        Cropped PIL Image
    """
    width, height = image.size
    crop_width, crop_height = crop_size
    
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    
    return image.crop((left, top, right, bottom))


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    This can help enhance lesion visibility in oral images.
    
    Args:
        image: Input image array (RGB)
        clip_limit: Threshold for contrast limiting
        
    Returns:
        Enhanced image array
    """
    import cv2
    
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to RGB
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return result


def segment_roi(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment the region of interest (ROI) in an oral image.
    
    Uses basic color-based segmentation to identify potential lesion areas.
    
    Args:
        image: Input image array (RGB)
        
    Returns:
        Tuple of (segmented_image, mask)
    """
    import cv2
    
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Define range for oral tissue colors (pinkish-red)
    lower_bound = np.array([0, 30, 60])
    upper_bound = np.array([20, 255, 255])
    
    # Create mask
    mask1 = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Also include reddish colors
    lower_bound2 = np.array([160, 30, 60])
    upper_bound2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_bound2, upper_bound2)
    
    # Combine masks
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Apply mask to image
    segmented = cv2.bitwise_and(image, image, mask=mask)
    
    return segmented, mask
