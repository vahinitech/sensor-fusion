# This file has been replaced by separate scripts:
# - train_model_only.py (Training phase)
# - evaluate_model.py (Evaluation phase)
# 
# For production use, run:
# 1. python train_model_only.py
# 2. python evaluate_model.py
# 3. ./run.py
#
# See TRAINING_GUIDE.md for detailed documentation

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from ai.character_recognition.model import CharacterRecognitionModel
from ai.character_recognition.preprocessor import SensorDataPreprocessor
from ai.character_recognition.trainer import ModelTrainer

# Configure multiprocessing for faster training
import tensorflow as tf

# Get number of available CPU cores
num_cores = os.cpu_count() or 4
# Use 75% of available cores to avoid overloading
workers = max(1, int(num_cores * 0.75))
print(f"\n🔧 System Configuration:")
print(f"  CPU Cores Available: {num_cores}")
print(f"  Workers for Training: {workers}")
print(f"  Using TensorFlow: {tf.__version__}\n")

# Configure TensorFlow for optimal performance
tf.config.threading.set_inter_op_parallelism_threads(workers)
tf.config.threading.set_intra_op_parallelism_threads(workers)

# Enable memory growth to prevent OOM errors
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def load_onhw_data(dataset_path):
    """Load OnHW dataset from specified path"""
    print(f"Loading OnHW dataset from {dataset_path}...")
    
    X_train = np.load(os.path.join(dataset_path, 'X_train.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(dataset_path, 'y_train.npy'), allow_pickle=True)
    X_test = np.load(os.path.join(dataset_path, 'X_test.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(dataset_path, 'y_test.npy'), allow_pickle=True)
    
    print(f"✓ Loaded {len(X_train)} training samples, {len(X_test)} test samples")
    print(f"  Sample shape: {X_train[0].shape} (timesteps x 13 features)")
    print(f"  Labels: {sorted(set(y_train))}")
    
    # Convert to list format (variable length sequences)
    X_train = list(X_train)
    X_test = list(X_test)
    y_train = list(y_train)
    y_test = list(y_test)
    
    return X_train, y_train, X_test, y_test


def main():
    print("=" * 70)
    print("CHARACTER RECOGNITION TRAINING WITH OnHW DATASET")
    print("=" * 70)
    
    # Configuration
    dataset_path = 'data/onhw-chars_2021-06-30/onhw2_both_dep_0'  # Both upper and lowercase
    output_dir = 'src/ai/models'
    timesteps = 512  # Maximum timesteps (will pad/truncate)
    n_features = 13  # OnHW has 13 sensor features
    num_classes = 52  # A-Z + a-z (both upper and lowercase)
    sampling_rate = 104  # Hz
    
    # For faster training, use uppercase only
    use_uppercase_only = True
    
    if use_uppercase_only:
        dataset_path = 'data/onhw-chars_2021-06-30/onhw2_upper_dep_0'
        num_classes = 26  # A-Z only
        print(f"\nUsing UPPERCASE letters only (26 classes)")
    else:
        print(f"\nUsing BOTH upper and lowercase letters (52 classes)")
    
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}/character_model.h5")
    print(f"Max timesteps: {timesteps} (~{timesteps/sampling_rate:.1f}s)")
    print(f"Features: {n_features}")
    print(f"Classes: {num_classes}")
    print("=" * 70)
    
    # Load data
    print("\n[1/5] Loading OnHW dataset...")
    X_train, y_train, X_test, y_test = load_onhw_data(dataset_path)
    
    # Initialize components
    print("\n[2/5] Initializing model and preprocessor...")
    model = CharacterRecognitionModel(
        timesteps=timesteps,
        n_features=n_features,
        num_classes=num_classes
    )
    
    preprocessor = SensorDataPreprocessor(
        max_timesteps=timesteps,
        sampling_rate=sampling_rate
    )
    
    trainer = ModelTrainer(model, preprocessor)
    print("✓ Model and preprocessor ready")
    
    # Prepare data (no splitting needed, already have train/test)
    print("\n[3/5] Preprocessing data...")
    
    # Process sequences (pad/truncate to fixed length)
    X_train_processed = preprocessor.batch_process(X_train, normalize=False)
    X_test_processed = preprocessor.batch_process(X_test, normalize=False)
    
    # Normalize - returns tuple when X_test is provided
    X_train_norm, X_test_norm = preprocessor.normalize_data(X_train_processed, X_test=X_test_processed, fit=True)
    
    # Encode labels to one-hot
    y_train_encoded = trainer.encode_labels(y_train)
    y_test_encoded = trainer.encode_labels(y_test)
    
    # Use 10% of training data for validation
    val_split = int(0.9 * len(X_train_norm))
    X_val = X_train_norm[val_split:]
    y_val = y_train_encoded[val_split:]
    X_train_norm = X_train_norm[:val_split]
    y_train_encoded = y_train_encoded[:val_split]
    
    print(f"✓ Training: {X_train_norm.shape}")
    print(f"✓ Validation: {X_val.shape}")
    print(f"✓ Test: {X_test_norm.shape}")
    
    # Build model
    print("\n[4/5] Building and training model...")
    model.build_model()
    print("✓ CNN+BiLSTM architecture built")
    
    # Train (fewer epochs for quick results)
    print(f"\n[4/5] Training model with {workers} parallel workers...")
    history = trainer.train(
        X_train_norm, y_train_encoded,
        X_val=X_val, y_val=y_val,
        epochs=30,  # Increase to 50-100 for better results
        batch_size=64
    )
    print("✓ Training completed")
    
    # Evaluate
    print("\n[5/5] Evaluating model...")
    metrics = trainer.evaluate(X_test_norm, y_test_encoded)
    print(f"✓ Test Accuracy: {metrics['accuracy']:.2%}")
    print("\nTop-5 Classification Report:")
    lines = metrics['report'].split('\n')
    for line in lines[:10]:  # Show first 10 lines
        print(line)
    
    # Save model
    print(f"\n[6/5] Saving model...")
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'character_model.h5')
    model.save(model_path)
    print(f"✓ Model saved to {model_path}")
    
    trainer.save_training_info(output_dir)
    print(f"✓ Training info saved")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)
    print(f"\nYour trained model is ready at:")
    print(f"  {model_path}")
    print(f"\nTest Accuracy: {metrics['accuracy']:.2%}")
    print(f"\nNow launch the GUI with: ./run.py")
    print("The model will be automatically loaded!")
    print("=" * 70)


if __name__ == '__main__':
    main()
