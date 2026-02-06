"""
ML Model Predictor for Oral Cancer Detection
Uses gauravvv7/Oralcancer model from Hugging Face
Model output: 0 = Oral Cancer, 1 = Normal (inverted from typical convention)
"""
import numpy as np
from pathlib import Path
from typing import Optional
import logging
import os

logger = logging.getLogger(__name__)


class OralCancerPredictor:
    """
    Wrapper class for the gauravvv7/Oralcancer model.
    Input: 300x300 RGB images
    Output: Binary classification where:
        - Output close to 0 = Oral Cancer
        - Output close to 1 = Normal
    """
    
    def __init__(self, config: dict):
        """Initialize the predictor with configuration."""
        self.config = config
        self.model = None
        self.classes = config["model"]["classes"]  # ["Normal", "Oral Cancer"]
        self.confidence_threshold = config["model"]["confidence_threshold"]
        
        # HuggingFace model info
        self.hf_repo_id = config["model"].get("huggingface_repo_id", "gauravvv7/Oralcancer")
        self.hf_filename = config["model"].get("huggingface_filename", "oral-cancer-model.h5")
        self.model_path = Path(config["model"]["path"])
        
        # Model expects 300x300 input
        self.input_size = (300, 300)
        self.model_type = None
        
        # Load the model
        self._load_model()
    
    def _download_from_hf(self) -> Optional[Path]:
        """Download model from Hugging Face."""
        try:
            from huggingface_hub import hf_hub_download
            
            print("="*80)
            print(f"🔽 DOWNLOADING MODEL FROM HUGGING FACE")
            print(f"   Repository: {self.hf_repo_id}")
            print(f"   Filename: {self.hf_filename}")
            print("="*80)
            
            models_dir = self.model_path.parent
            models_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Download directory: {models_dir}")
            
            downloaded = hf_hub_download(
                repo_id=self.hf_repo_id,
                filename=self.hf_filename,
                local_dir=str(models_dir),
                local_dir_use_symlinks=False
            )
            
            print("="*80)
            print(f"✅ MODEL DOWNLOADED SUCCESSFULLY")
            print(f"   Path: {downloaded}")
            print(f"   Size: {Path(downloaded).stat().st_size / (1024*1024):.2f} MB")
            print("="*80)
            return Path(downloaded)
            
        except Exception as e:
            print("="*80)
            print(f"❌ DOWNLOAD FAILED: {e}")
            print("="*80)
            logger.error(f"Download failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_model(self) -> None:
        """Load the Keras model."""
        print("\n" + "="*80)
        print("🚀 INITIALIZING MODEL LOADING")
        print("="*80)
        
        print(f"📍 Looking for model at: {self.model_path}")
        print(f"   File exists: {self.model_path.exists()}")
        
        # Download if needed
        if not self.model_path.exists():
            print("⚠️  Model file not found locally - initiating download...")
            downloaded = self._download_from_hf()
            if downloaded:
                self.model_path = downloaded
            else:
                print("❌ Failed to download model")
        else:
            print(f"✅ Model file found locally ({self.model_path.stat().st_size / (1024*1024):.2f} MB)")
        
        if not self.model_path.exists():
            print("="*80)
            print("❌ MODEL FILE NOT AVAILABLE - Using mock predictions")
            print("="*80)
            logger.error("Model file not found")
            self.model = None
            self.model_type = "mock"
            return
        
        try:
            import tensorflow as tf
            
            print("="*80)
            print(f"📥 LOADING MODEL FROM: {self.model_path}")
            print("="*80)
            
            self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
            self.model_type = "keras"
            
            # Get input size from model
            input_shape = self.model.input_shape
            if input_shape and len(input_shape) == 4:
                h, w = input_shape[1], input_shape[2]
                if h and w:
                    self.input_size = (h, w)
            
            print("="*80)
            print("✅ MODEL LOADED SUCCESSFULLY")
            print(f"   Model type: Keras/TensorFlow")
            print(f"   Input shape: {self.model.input_shape}")
            print(f"   Output shape: {self.model.output_shape}")
            print(f"   Expected input size: {self.input_size}")
            print(f"   Classes: {self.classes}")
            print(f"   ⚠️  Note: Model uses INVERTED labels (0=Cancer, 1=Normal)")
            print("="*80 + "\n")
            
        except Exception as e:
            print("="*80)
            print(f"❌ MODEL LOADING FAILED: {e}")
            print("="*80)
            logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.model_type = "mock"
            print("⚠️  Falling back to mock predictions")
    
    def predict(self, image: np.ndarray) -> dict:
        """Make a prediction on the input image."""
        print("\n" + "="*80)
        print("🔮 STARTING PREDICTION")
        print(f"   Model type: {self.model_type}")
        print(f"   Input image shape: {image.shape}")
        print("="*80)
        
        if self.model is None:
            print("⚠️  No model loaded - using mock prediction")
            return self._mock_prediction(image)
        
        return self._keras_prediction(image)
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input."""
        from PIL import Image as PILImage
        
        # Ensure image is in right format
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                img_uint8 = (image * 255).astype(np.uint8)
            else:
                img_uint8 = image.astype(np.uint8)
        else:
            img_uint8 = image
        
        # Convert to PIL and resize to model's expected input
        pil_img = PILImage.fromarray(img_uint8)
        pil_img = pil_img.resize((self.input_size[1], self.input_size[0]), PILImage.Resampling.LANCZOS)
        
        # Convert back to numpy and normalize to [0, 1]
        processed = np.array(pil_img).astype(np.float32) / 255.0
        
        return processed
    
    def _keras_prediction(self, image: np.ndarray) -> dict:
        """Make prediction using Keras model."""
        try:
            print("📊 Preprocessing image...")
            # Preprocess
            processed = self._preprocess(image)
            print(f"   Processed shape: {processed.shape}")
            print(f"   Value range: [{processed.min():.3f}, {processed.max():.3f}]")
            
            # Add batch dimension
            batch_input = np.expand_dims(processed, axis=0)
            print(f"   Batch input shape: {batch_input.shape}")
            
            # Predict
            print("🧠 Running model inference...")
            prediction = self.model.predict(batch_input, verbose=0)
            raw_output = float(prediction[0][0])
            
            print(f"📈 Raw model output: {raw_output:.4f}")
            
            # IMPORTANT: Training labels are {"cancer": 0, "non_cancer": 1}
            # With sigmoid output, raw_output represents P(class=1) = P(non_cancer/Normal)
            # Therefore: cancer_prob = 1 - raw_output, normal_prob = raw_output
            
            cancer_prob = 1.0 - raw_output
            normal_prob = raw_output
            
            print(f"   Interpreted probabilities (label mapping):")
            print(f"      • Normal: {normal_prob:.2%}")
            print(f"      • Oral Cancer: {cancer_prob:.2%}")
            
            # Use explicit class names to avoid order-related mixups
            class_probs = {
                "Normal": normal_prob,
                "Oral Cancer": cancer_prob
            }
            
            # Determine prediction
            if cancer_prob >= 0.5:
                predicted_class = "Oral Cancer"
                confidence = cancer_prob
            else:
                predicted_class = "Normal"
                confidence = normal_prob
            
            print("="*80)
            print(f"✅ PREDICTION COMPLETE")
            print(f"   Predicted class: {predicted_class}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Above threshold ({self.confidence_threshold:.0%}): {confidence >= self.confidence_threshold}")
            print("="*80 + "\n")
            
            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "class_probabilities": class_probs,
                "above_threshold": confidence >= self.confidence_threshold,
                "model_type": "keras",
                "raw_output": raw_output
            }
            
        except Exception as e:
            print("="*80)
            print(f"❌ PREDICTION FAILED: {e}")
            print("="*80)
            logger.error(f"Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            print("⚠️  Falling back to mock prediction")
            return self._mock_prediction(image)
    
    def _mock_prediction(self, image: np.ndarray) -> dict:
        """Fallback mock prediction."""
        try:
            if len(image.shape) == 3 and image.shape[2] >= 3:
                red = np.mean(image[:,:,0])
                green = np.mean(image[:,:,1])
                total = red + green + 0.001
                red_ratio = red / total
                
                if red_ratio > 0.52:
                    cancer_prob = 0.6 + (red_ratio - 0.52) * 2.0
                else:
                    cancer_prob = 0.3
                
                cancer_prob = max(0.1, min(0.95, cancer_prob))
            else:
                cancer_prob = 0.5
                
        except:
            cancer_prob = 0.5
        
        normal_prob = 1.0 - cancer_prob
        
        class_probs = {
            self.classes[0]: normal_prob,
            self.classes[1]: cancer_prob
        }
        
        if cancer_prob >= 0.5:
            predicted_class = self.classes[1]
            confidence = cancer_prob
        else:
            predicted_class = self.classes[0]
            confidence = normal_prob
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_probabilities": class_probs,
            "above_threshold": confidence >= self.confidence_threshold,
            "model_type": "mock",
            "is_mock": True
        }
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_loaded": self.model is not None,
            "model_type": self.model_type,
            "model_path": str(self.model_path),
            "input_size": self.input_size,
            "classes": self.classes,
            "huggingface_repo": self.hf_repo_id,
            "note": "Model output 0=Cancer, 1=Normal (inverted)"
        }
