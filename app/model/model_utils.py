"""
Model Utility Functions
"""
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values to [0, 1] range.
    
    Args:
        image: Input image array
        
    Returns:
        Normalized image array
    """
    if image.max() > 1.0:
        return image.astype(np.float32) / 255.0
    return image.astype(np.float32)


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Denormalize image from [0, 1] to [0, 255] range.
    
    Args:
        image: Normalized image array
        
    Returns:
        Denormalized image array
    """
    return (image * 255).astype(np.uint8)


def apply_augmentation(
    image: np.ndarray,
    rotation: float = 0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    brightness: float = 1.0
) -> np.ndarray:
    """
    Apply basic augmentation to an image.
    
    Args:
        image: Input image array
        rotation: Rotation angle in degrees
        flip_horizontal: Whether to flip horizontally
        flip_vertical: Whether to flip vertically
        brightness: Brightness factor (1.0 = no change)
        
    Returns:
        Augmented image array
    """
    import cv2
    
    result = image.copy()
    
    # Apply rotation
    if rotation != 0:
        h, w = result.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
        result = cv2.warpAffine(result, matrix, (w, h))
    
    # Apply flips
    if flip_horizontal:
        result = cv2.flip(result, 1)
    if flip_vertical:
        result = cv2.flip(result, 0)
    
    # Apply brightness
    if brightness != 1.0:
        result = np.clip(result * brightness, 0, 255).astype(np.uint8)
    
    return result


def calculate_class_weights(class_counts: dict) -> dict:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        class_counts: Dictionary mapping class names to sample counts
        
    Returns:
        Dictionary mapping class names to weights
    """
    total = sum(class_counts.values())
    n_classes = len(class_counts)
    
    weights = {}
    for class_name, count in class_counts.items():
        weights[class_name] = total / (n_classes * count)
    
    return weights


def get_gradcam_heatmap(
    model,
    image: np.ndarray,
    last_conv_layer_name: str,
    pred_index: Optional[int] = None
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for model interpretability.
    
    Args:
        model: Trained Keras model
        image: Input image array
        last_conv_layer_name: Name of the last convolutional layer
        pred_index: Index of the class to visualize (None for top prediction)
        
    Returns:
        Heatmap array
    """
    try:
        import tensorflow as tf
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            model.inputs,
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Pool gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
        
    except Exception as e:
        logger.error(f"Grad-CAM failed: {str(e)}")
        return None


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = None
) -> np.ndarray:
    """
    Overlay heatmap on original image.
    
    Args:
        image: Original image array
        heatmap: Heatmap array
        alpha: Transparency of heatmap overlay
        colormap: OpenCV colormap constant
        
    Returns:
        Image with heatmap overlay
    """
    import cv2
    
    if colormap is None:
        colormap = cv2.COLORMAP_JET
    
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert to uint8
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Overlay
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay
