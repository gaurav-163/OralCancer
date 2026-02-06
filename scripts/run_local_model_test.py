"""Simple script to test loading a local Keras model file specified in config.

Usage:
    python scripts/run_local_model_test.py

This script reads `config/config.yaml` for `model.path` and attempts to load
it with TensorFlow. If the file does not exist, it prints instructions.
"""
import yaml
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
config_path = ROOT / "config" / "config.yaml"

if not config_path.exists():
    print(f"Config file not found at {config_path}")
    sys.exit(2)

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

model_path = Path(ROOT / cfg["model"]["path"]) if cfg and "model" in cfg else None

print(f"Configured model path: {model_path}")

if model_path is None:
    print("No model path configured in config.")
    sys.exit(2)

if not model_path.exists():
    print("Model file not found locally.")
    print("Place your `oral_cancer_medical_final.h5` at:")
    print(f"  {model_path}")
    print("Or update `config/config.yaml` to point to the correct local path.")
    sys.exit(1)

print("Model file found — attempting to load with TensorFlow Keras...")
try:
    import tensorflow as tf
    model = tf.keras.models.load_model(str(model_path), compile=False)
    print("Model loaded successfully.")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
except Exception as e:
    print("Failed to load model:")
    import traceback
    traceback.print_exc()
    sys.exit(3)
