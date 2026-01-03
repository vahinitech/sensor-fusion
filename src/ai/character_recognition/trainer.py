"""
Model Training Pipeline for Character Recognition
Handles training, validation, and model persistence
"""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import os
import json
from datetime import datetime


class ModelTrainer:
    """Train and evaluate character recognition models"""
    
    def __init__(self, model, preprocessor, character_set='ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        """
        Initialize trainer
        
        Args:
            model: CharacterRecognitionModel instance
            preprocessor: SensorDataPreprocessor instance
            character_set: Characters to recognize
        """
        self.model = model
        self.preprocessor = preprocessor
        self.character_set = character_set
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(list(character_set))
        
        self.history = None
        self.train_metrics = None
        self.val_metrics = None
    
    def encode_labels(self, y):
        """
        Encode character labels to one-hot
        
        Args:
            y: List/array of character labels
        
        Returns:
            One-hot encoded labels
        """
        y_encoded = self.label_encoder.transform(y)
        y_categorical = to_categorical(y_encoded, num_classes=len(self.character_set))
        return y_categorical
    
    def decode_labels(self, y_encoded):
        """
        Decode one-hot back to characters
        
        Args:
            y_encoded: One-hot encoded labels
        
        Returns:
            Character labels
        """
        y_indices = np.argmax(y_encoded, axis=1)
        y_chars = self.label_encoder.inverse_transform(y_indices)
        return y_chars
    
    def prepare_data(self, X_raw, y_labels, test_size=0.2, validate=True):
        """
        Prepare data for training
        
        Args:
            X_raw: Raw sensor data list
            y_labels: Character labels
            test_size: Fraction for testing
            validate: Whether to use validation split
        
        Returns:
            (X_train, y_train, X_test, y_test) or (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        # Process and pad data
        X_processed = self.preprocessor.batch_process(X_raw, normalize=False)
        
        # Normalize
        X_normalized = self.preprocessor.normalize_data(
            X_processed, X_test=None, fit=True
        )
        
        # Encode labels
        y_encoded = self.encode_labels(y_labels)
        
        # Split data
        n_samples = len(X_normalized)
        n_test = int(n_samples * test_size)
        n_train = n_samples - n_test
        
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        X_train = X_normalized[train_idx]
        y_train = y_encoded[train_idx]
        X_test = X_normalized[test_idx]
        y_test = y_encoded[test_idx]
        
        if validate:
            # Further split training into train/val
            n_val = int(n_train * 0.2)
            n_train_actual = n_train - n_val
            
            train_indices = np.random.permutation(n_train)
            train_actual_idx = train_indices[:n_train_actual]
            val_idx = train_indices[n_train_actual:]
            
            X_val = X_train[val_idx]
            y_val = y_train[val_idx]
            X_train_final = X_train[train_actual_idx]
            y_train_final = y_train[train_actual_idx]
            
            return X_train_final, y_train_final, X_val, y_val, X_test, y_test
        
        return X_train, y_train, X_test, y_test
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the model
        
        Args:
            X_train: Training data (N, T, C)
            y_train: Training labels (one-hot)
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            epochs: Number of epochs
            batch_size: Batch size
            validation_split: Validation split ratio
        
        Returns:
            Training history
        """
        # Early stopping callback
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        # Reduce learning rate callback
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        callbacks = [early_stop, reduce_lr]
        
        if X_val is not None and y_val is not None:
            val_data = (X_val, y_val)
        else:
            val_data = None
        
        print(f"Training on {len(X_train)} samples...")
        self.history = self.model.train(
            X_train, y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set
        
        Args:
            X_test: Test data
            y_test: Test labels (one-hot)
        
        Returns:
            Metrics dictionary
        """
        # Get predictions
        y_pred_probs = self.model.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_labels, y_pred)
        
        y_test_chars = self.label_encoder.inverse_transform(y_test_labels)
        y_pred_chars = self.label_encoder.inverse_transform(y_pred)
        
        # Fix target_names - split string into individual characters
        target_names = list(self.character_set)
        
        report = classification_report(
            y_test_chars, y_pred_chars,
            target_names=target_names
        )
        
        conf_matrix = confusion_matrix(y_test_chars, y_pred_chars)
        
        self.val_metrics = {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': conf_matrix.tolist()
        }
        
        return self.val_metrics
    
    def save_training_info(self, save_dir):
        """
        Save training information and metadata
        
        Args:
            save_dir: Directory to save files
        """
        os.makedirs(save_dir, exist_ok=True)
        
        info = {
            'timestamp': datetime.now().isoformat(),
            'character_set': self.character_set,
            'num_classes': len(self.character_set),
            'model_config': {
                'timesteps': self.model.timesteps,
                'n_features': self.model.n_features,
                'num_classes': self.model.num_classes
            },
            'preprocessing_config': {
                'max_timesteps': self.preprocessor.max_timesteps,
                'sampling_rate': self.preprocessor.sampling_rate
            }
        }
        
        if self.history:
            info['training_history'] = {
                'loss': [float(l) for l in self.history.history.get('loss', [])],
                'accuracy': [float(a) for a in self.history.history.get('accuracy', [])],
                'val_loss': [float(l) for l in self.history.history.get('val_loss', [])],
                'val_accuracy': [float(a) for a in self.history.history.get('val_accuracy', [])]
            }
        
        if self.val_metrics:
            info['validation_metrics'] = {
                'accuracy': float(self.val_metrics['accuracy']),
                'report': self.val_metrics['report']
            }
        
        with open(os.path.join(save_dir, 'training_info.json'), 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"Training info saved to {save_dir}/training_info.json")
