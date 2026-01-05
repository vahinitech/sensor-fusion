#!/usr/bin/env python3
"""
Character Recognition Model - Evaluation Script
Tests the trained model on test set and generates detailed metrics

Requirements:
- Trained model must exist at src/ai/models/character_model.h5
- Run 'train_model_only.py' first

What we achieve:
✓ Load saved model from file
✓ Test on held-out test dataset
✓ Calculate accuracy metrics
✓ Generate classification report
✓ Display per-character performance
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from ai.character_recognition.model import CharacterRecognitionModel
from ai.character_recognition.preprocessor import SensorDataPreprocessor
from ai.character_recognition.trainer import ModelTrainer
import json


def load_onhw_data(dataset_path):
    """Load OnHW dataset"""
    print(f"Loading test set from {dataset_path}...")

    X_test = np.load(os.path.join(dataset_path, "X_test.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(dataset_path, "y_test.npy"), allow_pickle=True)

    print(f"✓ Loaded {len(X_test)} test samples")

    X_test = list(X_test)
    y_test = list(y_test)

    return X_test, y_test


def main():
    print("=" * 70)
    print("CHARACTER RECOGNITION MODEL - EVALUATION PHASE")
    print("=" * 70)

    model_path = "src/ai/models/character_model.h5"
    metadata_path = "src/ai/models/model_metadata.json"
    dataset_path = "data/onhw-chars_2021-06-30/onhw2_upper_dep_0"

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ ERROR: Model not found at {model_path}")
        print(f"   Please run 'python train_model_only.py' first\n")
        return

    print(f"\n✓ Found trained model: {model_path}")
    print(f"  Size: {os.path.getsize(model_path) / (1024*1024):.2f} MB\n")

    # Load metadata
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print("📋 Training metadata:")
        print(f"  Model type: {metadata['model_type']}")
        print(f"  Training time: {metadata['training_time_minutes']:.1f} minutes")
        print(f"  Samples trained on: {metadata['training_samples']}")
        print(f"  Training date: {metadata['training_date'][:10]}\n")

    # Load configuration from metadata or use defaults
    timesteps = metadata.get("timesteps", 512)
    n_features = metadata.get("n_features", 13)
    num_classes = metadata.get("num_classes", 26)
    sampling_rate = metadata.get("sampling_rate", 104)

    print(f"[1/3] Loading test data...")
    X_test, y_test = load_onhw_data(dataset_path)

    print(f"\n[2/3] Initializing preprocessor and loading model...")

    preprocessor = SensorDataPreprocessor(max_timesteps=timesteps, sampling_rate=sampling_rate)

    # Load the trained model
    model = CharacterRecognitionModel(timesteps=timesteps, n_features=n_features, num_classes=num_classes)
    model.load_model(model_path)

    # Create trainer for label encoding
    trainer = ModelTrainer(model, preprocessor)

    # Process test data
    print(f"Processing {len(X_test)} test samples...")
    X_test_processed = preprocessor.batch_process(X_test, normalize=False)

    # For evaluation, we need to normalize using the training stats
    # This is a limitation - ideally we'd save the scaler
    # For now, just normalize the test set independently
    X_test_norm = preprocessor.normalize_data(X_test_processed, fit=True)

    # Encode test labels
    y_test_encoded = trainer.encode_labels(y_test)

    print(f"✓ Test data shape: {X_test_norm.shape}")

    print(f"\n[3/3] Evaluating model on test set...")
    print(f"  Test samples: {len(X_test_norm)}")
    print(f"  Classes: 26 (A-Z)")
    print(f"  Features: {n_features} sensor channels")
    print(f"  Max sequence length: {timesteps}\n")

    # Evaluate using the loaded model
    try:
        metrics = trainer.evaluate(X_test_norm, y_test_encoded)

        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print(f"\n📊 Test Accuracy: {metrics['accuracy']:.2%}")
        print(f"   ({int(metrics['accuracy'] * len(X_test))} / {len(X_test)} samples correct)\n")

        print("Per-Character Performance:")
        print(metrics["report"])

        print("\n" + "=" * 70)
        print("✅ EVALUATION COMPLETE")
        print("=" * 70)
        print(f"\n🎯 Model Performance: {metrics['accuracy']:.1%} accuracy")
        print(f"📁 Model path: {model_path}")
        print(f"🚀 Ready to use in GUI: ./run.py\n")

    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
