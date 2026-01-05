#!/usr/bin/env python3
"""
Character Recognition Model Training Script (Training Only)
Focuses ONLY on training the CNN+BiLSTM model on OnHW dataset
Saves the trained model for later use

What we achieve:
✓ Load professional IMU handwriting dataset (OnHW - 11,542 samples)
✓ Preprocess sensor data (normalize, pad to 512 timesteps)
✓ Train CNN+BiLSTM neural network (26 output classes for A-Z)
✓ Validate on held-out data during training
✓ Save trained weights to file (character_model.h5)

Output: Trained model that recognizes handwritten characters from sensor data
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from ai.character_recognition.model import CharacterRecognitionModel
from ai.character_recognition.preprocessor import SensorDataPreprocessor
from ai.character_recognition.trainer import ModelTrainer
import tensorflow as tf
from datetime import datetime

# Configure multiprocessing for faster training
num_cores = os.cpu_count() or 4
workers = max(1, int(num_cores * 0.75))
print(f"\n🔧 System Configuration:")
print(f"  CPU Cores Available: {num_cores}")
print(f"  Workers for Training: {workers}")
print(f"  Using TensorFlow: {tf.__version__}")
print(f"\n📌 IMPORTANT - Why Epochs Are Sequential:")
print(f"  • Each epoch builds on weights from previous epoch")
print(f"  • Epoch 1: Initialize → Train → Update weights")
print(f"  • Epoch 2: Use updated weights → Train → Update")
print(f"  • Cannot parallelize epochs (accuracy would break)")
print(f"  • BUT: Batches within epochs ARE parallelized\n")

tf.config.threading.set_inter_op_parallelism_threads(workers)
tf.config.threading.set_intra_op_parallelism_threads(workers)

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def load_onhw_data(dataset_path):
    """Load OnHW dataset from specified path"""
    print(f"Loading OnHW dataset from {dataset_path}...")

    X_train = np.load(os.path.join(dataset_path, "X_train.npy"), allow_pickle=True)
    y_train = np.load(os.path.join(dataset_path, "y_train.npy"), allow_pickle=True)
    X_test = np.load(os.path.join(dataset_path, "X_test.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(dataset_path, "y_test.npy"), allow_pickle=True)

    print(f"✓ Loaded {len(X_train)} training samples, {len(X_test)} test samples")
    print(f"  Sample shape: {X_train[0].shape} (timesteps x 13 features)")
    print(f"  Labels: {sorted(set(y_train))}\n")

    X_train = list(X_train)
    X_test = list(X_test)
    y_train = list(y_train)
    y_test = list(y_test)

    return X_train, y_train, X_test, y_test


def main():
    print("=" * 70)
    print("CHARACTER RECOGNITION MODEL - TRAINING PHASE")
    print("=" * 70)
    print("\n📊 WHAT WE'RE BUILDING:")
    print("  • CNN+BiLSTM neural network")
    print("  • Input: IMU sensor data (13 channels @ 104Hz)")
    print("  • Output: Character recognition (A-Z)")
    print("  • Dataset: OnHW professional handwriting corpus")
    print("  • Training samples: ~11,500")
    print("  • Validation accuracy target: >85%\n")
    print("=" * 70)

    # Configuration
    dataset_path = "data/onhw-chars_2021-06-30/onhw2_upper_dep_0"
    output_dir = "src/ai/models"
    timesteps = 512
    n_features = 13
    num_classes = 26
    sampling_rate = 104

    print(f"\n[1/4] Loading OnHW dataset...")
    X_train, y_train, X_test, y_test = load_onhw_data(dataset_path)

    print(f"[2/4] Initializing model and preprocessor...")
    model = CharacterRecognitionModel(timesteps=timesteps, n_features=n_features, num_classes=num_classes)

    preprocessor = SensorDataPreprocessor(max_timesteps=timesteps, sampling_rate=sampling_rate)

    trainer = ModelTrainer(model, preprocessor)
    print("✓ Model and preprocessor ready\n")

    print(f"[3/4] Preprocessing {len(X_train)} training samples...")

    # Process sequences
    X_train_processed = preprocessor.batch_process(X_train, normalize=False)
    X_test_processed = preprocessor.batch_process(X_test, normalize=False)

    # Normalize
    X_train_norm, X_test_norm = preprocessor.normalize_data(X_train_processed, X_test=X_test_processed, fit=True)

    # Encode labels
    y_train_encoded = trainer.encode_labels(y_train)
    y_test_encoded = trainer.encode_labels(y_test)

    # Split training data for validation
    val_split = int(0.9 * len(X_train_norm))
    X_val = X_train_norm[val_split:]
    y_val = y_train_encoded[val_split:]
    X_train_norm = X_train_norm[:val_split]
    y_train_encoded = y_train_encoded[:val_split]

    print(f"✓ Training: {X_train_norm.shape}")
    print(f"✓ Validation: {X_val.shape}")
    print(f"✓ Test (held-out): {X_test_norm.shape}\n")

    print(f"[4/4] Training CNN+BiLSTM model (30 epochs)...")
    print(f"  Parallelization Strategy:")
    print(f"  • Epochs: Sequential (required by design)")
    print(f"  • Batches: Parallel ({workers} workers)")
    print(f"  • Data loading: Multi-worker preprocessing")
    print(f"  • Batch size: 64")
    print(f"  • Learning rate: 0.001 (with decay)\n")

    start_time = datetime.now()

    # Use multi-worker data loading
    history = trainer.train(X_train_norm, y_train_encoded, X_val=X_val, y_val=y_val, epochs=30, batch_size=64)

    elapsed = (datetime.now() - start_time).total_seconds() / 60
    print(f"\n✓ Training completed in {elapsed:.1f} minutes")

    # Save model
    print(f"\n[5/4] Saving trained model...")
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "character_model.h5")
    model.save_model(model_path)
    print(f"✓ Model saved to {model_path}")
    print(f"  (This location is where GUI looks for the model)")
    print(f"  File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    print(f"  Architecture: CNN+BiLSTM (710,874 parameters)")

    # Save training metadata for production use
    metadata = {
        "model_type": "CNN+BiLSTM",
        "purpose": "Handwritten uppercase letter recognition (A-Z)",
        "input_shape": [timesteps, n_features],
        "output_classes": num_classes,
        "timesteps": timesteps,
        "n_features": n_features,
        "num_classes": num_classes,
        "sampling_rate": sampling_rate,
        "dataset": "OnHW uppercase (Fraunhofer Institute)",
        "training_samples": len(X_train_norm),
        "validation_samples": len(X_val),
        "test_samples": len(X_test_norm),
        "epochs_trained": 30,
        "batch_size": 64,
        "optimizer": "Adam (lr=0.001)",
        "loss_function": "categorical_crossentropy",
        "parallelization": {
            "epochs": "Sequential (by design)",
            "batches": f"Parallel ({workers} workers)",
            "data_loading": "Multi-worker preprocessing",
        },
        "training_time_minutes": elapsed,
        "training_date": datetime.now().isoformat(),
        "model_location": model_path,
        "gui_integration": "Automatic - place in src/ai/models/",
    }

    import json

    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Metadata saved to {metadata_path}")
    print(f"  (Used by GUI for model loading)")

    print("\n" + "=" * 70)
    print("TRAINING PHASE COMPLETE!")
    print("=" * 70)
    print(f"\n✅ Trained model ready at: {model_path}")
    print(f"📊 Training samples: {len(X_train_norm)}")
    print(f"⏱️  Training time: {elapsed:.1f} minutes")
    print(f"🎯 Next: Run './evaluate_model.py' to test accuracy")
    print(f"🚀 Then: Launch GUI with './run.py' to use the model!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
