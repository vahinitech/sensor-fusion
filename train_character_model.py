#!/usr/bin/env python3
"""
Training Script for Character Recognition Model
Trains the model on sensor data at 104Hz sampling rate
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from ai.character_recognition.model import CharacterRecognitionModel
from ai.character_recognition.preprocessor import SensorDataPreprocessor
from ai.character_recognition.trainer import ModelTrainer
from ai.utils.data_utils import DatasetLoader
import argparse


def generate_sample_data(num_samples=100, timesteps=512, num_features=13):
    """
    Generate synthetic training data for testing
    In real usage, this would come from OnHW dataset or sensor recordings

    Args:
        num_samples: Number of samples to generate
        timesteps: Time steps per sample (at 104Hz)
        num_features: Number of sensor features

    Returns:
        (X, y) training data
    """
    print(f"Generating {num_samples} synthetic samples...")

    X = []
    y = []
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for i in range(num_samples):
        # Generate random sensor sequence
        sample = np.random.randn(timesteps, num_features).astype(np.float32)

        # Add some structure to make it more realistic
        # Different characters have slightly different patterns
        char_idx = i % len(characters)
        char = characters[char_idx]

        # Add characteristic patterns for different letters
        if ord(char) % 3 == 0:
            sample[:, 0] *= 1.5  # Scale acceleration
        if ord(char) % 5 == 0:
            sample[:, 3:6] *= 0.8  # Scale gyroscope

        X.append(sample)
        y.append(char)

    print(f"Generated X shape: {len(X)}, X[0] shape: {X[0].shape}")
    print(f"Generated {len(y)} labels: {set(y)}")

    return X, y


def train_character_recognition_model(args):
    """
    Main training function

    Args:
        args: Command line arguments
    """

    print("=" * 70)
    print("CHARACTER RECOGNITION MODEL TRAINING")
    print("=" * 70)
    print(f"Sampling Rate: {args.sampling_rate} Hz (LSM6DSO)")
    print(f"Timesteps: {args.timesteps} (~{args.timesteps/args.sampling_rate:.1f}s per character)")
    print(f"Number of Features: {args.features}")
    print(f"Number of Classes: {args.num_classes}")
    print("=" * 70)

    # Initialize components
    print("\n[1/6] Initializing model components...")
    model = CharacterRecognitionModel(timesteps=args.timesteps, n_features=args.features, num_classes=args.num_classes)

    preprocessor = SensorDataPreprocessor(max_timesteps=args.timesteps, sampling_rate=args.sampling_rate)

    trainer = ModelTrainer(model, preprocessor)
    print("✓ Model and preprocessor initialized")

    # Load or generate data
    print("\n[2/6] Loading training data...")
    if args.dataset == "synthetic":
        X_raw, y_labels = generate_sample_data(
            num_samples=args.num_samples, timesteps=args.timesteps, num_features=args.features
        )
    elif args.dataset == "onhw":
        try:
            loader = DatasetLoader(sampling_rate=args.sampling_rate)
            X_train, y_train, X_test, y_test = loader.load_onhw_dataset(args.data_path, download=args.download)
            # Combine for training
            X_raw = np.concatenate([X_train, X_test], axis=0)
            y_labels = np.concatenate([y_train, y_test], axis=0)
        except Exception as e:
            print(f"Could not load OnHW dataset: {e}")
            print("Falling back to synthetic data...")
            X_raw, y_labels = generate_sample_data(args.num_samples)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"✓ Loaded {len(X_raw)} samples")

    # Prepare data
    print("\n[3/6] Preparing and preprocessing data...")
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = trainer.prepare_data(
        X_raw, y_labels, test_size=args.test_size, validate=True
    )

    # Preprocess
    X_train = preprocessor.batch_process(X_train_raw, normalize=False)
    X_val = preprocessor.batch_process(X_val_raw, normalize=False)
    X_test = preprocessor.batch_process(X_test_raw, normalize=False)

    # Normalize training data and apply to val/test
    X_train_norm = preprocessor.normalize_data(X_train, X_test=X_val, fit=True)
    X_val_norm = preprocessor.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_norm = preprocessor.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    print(f"✓ Training data: {X_train_norm.shape}")
    print(f"✓ Validation data: {X_val_norm.shape}")
    print(f"✓ Test data: {X_test_norm.shape}")

    # Build model
    print("\n[4/6] Building model architecture...")
    model.build_model()
    print("✓ Model built with CNN+BiLSTM architecture")
    print(f"  - Conv layers: 64 → 128 → 256 filters")
    print(f"  - BiLSTM: 128 → 64 units")
    print(f"  - Output: {args.num_classes} classes (A-Z)")

    # Train
    print("\n[5/6] Training model...")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.learning_rate}")

    history = trainer.train(
        X_train_norm, y_train, X_val=X_val_norm, y_val=y_val, epochs=args.epochs, batch_size=args.batch_size
    )
    print("✓ Training completed")

    # Evaluate
    print("\n[6/6] Evaluating model...")
    metrics = trainer.evaluate(X_test_norm, y_test)
    print(f"✓ Test Accuracy: {metrics['accuracy']:.4f}")
    print("\nClassification Report:")
    print(metrics["report"])

    # Save model
    print("\n[7/6] Saving model and metadata...")
    os.makedirs(args.output_dir, exist_ok=True)

    model_path = os.path.join(args.output_dir, "character_recognition_104hz.h5")
    model.save(model_path)
    print(f"✓ Model saved to {model_path}")

    # Save training info
    trainer.save_training_info(args.output_dir)
    print(f"✓ Training info saved to {args.output_dir}/training_info.json")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nModel Location: {model_path}")
    print(f"Sampling Rate: {args.sampling_rate} Hz")
    print(f"Timestamp Length: {args.timesteps/args.sampling_rate:.1f}s")
    print(f"Test Accuracy: {metrics['accuracy']:.2%}")

    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train Character Recognition Model at 104Hz")

    # Data arguments
    parser.add_argument(
        "--dataset", type=str, default="synthetic", choices=["synthetic", "onhw"], help="Dataset to use for training"
    )
    parser.add_argument("--data-path", type=str, default="./data", help="Path to dataset")
    parser.add_argument("--download", action="store_true", help="Download OnHW dataset if not found")

    # Model arguments
    parser.add_argument("--timesteps", type=int, default=512, help="Timesteps per sample (at sampling_rate Hz)")
    parser.add_argument("--features", type=int, default=13, help="Number of sensor features")
    parser.add_argument("--num-classes", type=int, default=26, help="Number of output classes (A-Z)")
    parser.add_argument("--sampling-rate", type=int, default=104, help="Sensor sampling rate (Hz)")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data to use for testing")

    # Other arguments
    parser.add_argument("--num-samples", type=int, default=500, help="Number of synthetic samples to generate")
    parser.add_argument("--output-dir", type=str, default="./src/ai/models", help="Directory to save trained model")

    args = parser.parse_args()

    train_character_recognition_model(args)


if __name__ == "__main__":
    main()
