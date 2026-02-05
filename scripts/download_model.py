#!/usr/bin/env python3
"""
Model Download Script
Downloads oral cancer detection models from Hugging Face Hub
"""
import argparse
import os
import sys
from pathlib import Path


def download_from_huggingface(model_id: str, filename: str, save_dir: str = "models"):
    """Download a specific file from Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_download
        
        print(f"📥 Downloading {filename} from {model_id}...")
        
        downloaded_path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir=save_dir
        )
        
        print(f"✅ Model downloaded successfully to: {downloaded_path}")
        return downloaded_path
        
    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        return None


def download_transformers_model(model_id: str, save_dir: str = "models"):
    """Download a full Transformers model (for ViT, Swin, etc.)."""
    try:
        from transformers import AutoModelForImageClassification, AutoImageProcessor
        
        print(f"📥 Downloading Transformers model: {model_id}...")
        
        # Create save directory
        save_path = Path(save_dir) / model_id.replace("/", "_")
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Download model and processor
        model = AutoModelForImageClassification.from_pretrained(model_id)
        processor = AutoImageProcessor.from_pretrained(model_id)
        
        # Save locally
        model.save_pretrained(save_path)
        processor.save_pretrained(save_path)
        
        print(f"✅ Model saved to: {save_path}")
        return str(save_path)
        
    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        return None


def list_available_models():
    """List some known oral cancer / medical imaging models on Hugging Face."""
    print("\n📋 Available Oral Cancer / Medical Imaging Models on Hugging Face:\n")
    
    models = [
        {
            "id": "imfarzanansari/oral-cancer-detection-vit",
            "description": "Vision Transformer for oral cancer detection",
            "type": "transformers"
        },
        {
            "id": "google/vit-base-patch16-224",
            "description": "General ViT (can be fine-tuned for oral cancer)",
            "type": "transformers"
        },
        {
            "id": "microsoft/swin-tiny-patch4-window7-224",
            "description": "Swin Transformer (can be fine-tuned)",
            "type": "transformers"
        },
        {
            "id": "facebook/deit-base-patch16-224",
            "description": "Data-efficient Image Transformer",
            "type": "transformers"
        }
    ]
    
    for model in models:
        print(f"  • {model['id']}")
        print(f"    Description: {model['description']}")
        print(f"    Type: {model['type']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Download oral cancer detection models from Hugging Face"
    )
    
    parser.add_argument(
        "--model-id",
        type=str,
        help="Hugging Face model ID (e.g., 'username/model-name')"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Specific file to download (e.g., 'model.h5'). If not specified, downloads full Transformers model."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="Directory to save the model (default: models)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
        return
    
    if not args.model_id:
        print("❌ Please provide a model ID with --model-id")
        print("   Use --list to see available models")
        sys.exit(1)
    
    # Create save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    if args.filename:
        # Download specific file
        result = download_from_huggingface(
            args.model_id, 
            args.filename, 
            args.save_dir
        )
    else:
        # Download full Transformers model
        result = download_transformers_model(
            args.model_id,
            args.save_dir
        )
    
    if result:
        print("\n🎉 Download complete!")
        print("\nTo use this model, update your config/config.yaml:")
        print(f'  huggingface_model_id: "{args.model_id}"')
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
