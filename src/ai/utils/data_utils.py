"""
Data Utilities for Loading and Managing Datasets
Handles dataset loading, preprocessing, and augmentation
"""

import numpy as np
import os
import json
from typing import Tuple, List, Optional
import zipfile
import urllib.request


class DatasetLoader:
    """Load and manage training datasets"""
    
    def __init__(self, sampling_rate=104):
        """
        Initialize dataset loader
        
        Args:
            sampling_rate: Sampling rate of sensor data (Hz)
        """
        self.sampling_rate = sampling_rate
        self.datasets = {}
    
    def load_onhw_dataset(self, path_or_url: str, download=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load OnHW (Online Handwriting) dataset
        
        OnHW dataset format:
        - X_train: (N_train, T, C) - training samples with variable timesteps
        - X_test: (N_test, T, C) - test samples
        - y_train: (N_train,) - character labels (0-25 for A-Z)
        - y_test: (N_test,) - test labels
        
        Args:
            path_or_url: Local path or URL to dataset
            download: Whether to download if not found
        
        Returns:
            (X_train, y_train, X_test, y_test)
        """
        
        # If URL and download requested
        if path_or_url.startswith('http') and download:
            path = self._download_dataset(path_or_url)
        else:
            path = path_or_url
        
        # Load numpy files
        if path.endswith('.zip'):
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)
                
                # Find .npy files
                X_train = np.load(os.path.join(tmpdir, 'X_train.npy'), allow_pickle=True)
                y_train = np.load(os.path.join(tmpdir, 'y_train.npy'), allow_pickle=True)
                X_test = np.load(os.path.join(tmpdir, 'X_test.npy'), allow_pickle=True)
                y_test = np.load(os.path.join(tmpdir, 'y_test.npy'), allow_pickle=True)
        else:
            X_train = np.load(os.path.join(path, 'X_train.npy'), allow_pickle=True)
            y_train = np.load(os.path.join(path, 'y_train.npy'), allow_pickle=True)
            X_test = np.load(os.path.join(path, 'X_test.npy'), allow_pickle=True)
            y_test = np.load(os.path.join(path, 'y_test.npy'), allow_pickle=True)
        
        # Convert string labels to integers (A=0, B=1, ..., Z=25)
        if y_train.dtype == object or isinstance(y_train[0], str):
            y_train = self._encode_character_labels(y_train)
            y_test = self._encode_character_labels(y_test)
        
        self.datasets['onhw'] = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'source': 'OnHW'
        }
        
        print(f"Loaded OnHW dataset:")
        print(f"  X_train: {X_train.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_test: {y_test.shape}")
        
        return X_train, y_train, X_test, y_test
    
    def load_csv_dataset(self, csv_path: str, label_col='label') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset from CSV files
        
        Expected format:
        - Each row is one sample
        - Columns contain features
        - label_col contains character label
        
        Args:
            csv_path: Path to CSV file
            label_col: Name of label column
        
        Returns:
            (X, y) where X is (N, T, C) and y is (N,)
        """
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        
        # Extract labels
        y = df[label_col].values
        y = self._encode_character_labels(y)
        
        # Extract features
        feature_cols = [c for c in df.columns if c != label_col]
        X = df[feature_cols].values
        
        # Reshape to (N, 1, C) - each row is one sample with C features
        X = X.reshape(X.shape[0], 1, X.shape[1])
        
        print(f"Loaded CSV dataset from {csv_path}")
        print(f"  X: {X.shape}")
        print(f"  y: {y.shape}")
        
        return X, y
    
    def load_sensor_recordings(self, recording_dir: str) -> Tuple[List[np.ndarray], List[str]]:
        """
        Load sensor recordings from directory
        
        Expected structure:
        recording_dir/
            character_A_1.npy
            character_A_2.npy
            character_B_1.npy
            ...
        
        Args:
            recording_dir: Directory containing .npy files
        
        Returns:
            (X_list, y_list) where X_list is list of sensor arrays, y_list is character labels
        """
        X_list = []
        y_list = []
        
        for filename in sorted(os.listdir(recording_dir)):
            if not filename.endswith('.npy'):
                continue
            
            filepath = os.path.join(recording_dir, filename)
            data = np.load(filepath, allow_pickle=True)
            
            # Extract character from filename (e.g., "character_A_1" -> "A")
            char = filename.split('_')[1].upper()
            
            X_list.append(data)
            y_list.append(char)
        
        print(f"Loaded {len(X_list)} recordings from {recording_dir}")
        
        return X_list, y_list
    
    def create_train_val_test_split(self, X: np.ndarray, y: np.ndarray,
                                   train_ratio=0.7, val_ratio=0.15, 
                                   test_ratio=0.15,
                                   stratified=True) -> Tuple[np.ndarray, np.ndarray, 
                                                             np.ndarray, np.ndarray,
                                                             np.ndarray, np.ndarray]:
        """
        Create train/val/test split
        
        Args:
            X: Features
            y: Labels
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            stratified: Whether to use stratified split
        
        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, \
            "Ratios must sum to 1.0"
        
        n_samples = len(X)
        n_val = int(n_samples * val_ratio)
        n_test = int(n_samples * test_ratio)
        n_train = n_samples - n_val - n_test
        
        if stratified:
            from sklearn.model_selection import train_test_split
            
            # First split: train/temp
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, train_size=train_ratio, stratify=y, random_state=42
            )
            
            # Second split: val/test
            test_size_in_temp = test_ratio / (val_ratio + test_ratio)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=test_size_in_temp, 
                stratify=y_temp, random_state=42
            )
        else:
            indices = np.random.permutation(n_samples)
            
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train+n_val]
            test_idx = indices[n_train+n_val:]
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            X_test, y_test = X[test_idx], y[test_idx]
        
        print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def augment_data(self, X: np.ndarray, y: np.ndarray,
                    augmentation_factor=2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment dataset with noise and time-warping
        
        Args:
            X: Input features (N, T, C)
            y: Labels (N,)
            augmentation_factor: How many augmented copies per sample
        
        Returns:
            (X_augmented, y_augmented)
        """
        X_augmented = [X]
        y_augmented = [y]
        
        for _ in range(augmentation_factor - 1):
            # Add Gaussian noise
            noise = np.random.normal(0, 0.01, X.shape)
            X_aug = X + noise
            
            # Optional: Time warping (scaling)
            scale = np.random.uniform(0.95, 1.05)
            X_aug = X_aug * scale
            
            X_augmented.append(X_aug)
            y_augmented.append(y)
        
        X_result = np.concatenate(X_augmented, axis=0)
        y_result = np.concatenate(y_augmented, axis=0)
        
        print(f"Augmented from {len(X)} to {len(X_result)} samples")
        
        return X_result, y_result
    
    def _encode_character_labels(self, y: np.ndarray) -> np.ndarray:
        """Convert character labels to integers"""
        if y.dtype in [np.int32, np.int64]:
            return y
        
        # Map A-Z to 0-25
        char_to_int = {chr(65+i): i for i in range(26)}
        
        y_encoded = np.zeros(len(y), dtype=np.int32)
        for i, label in enumerate(y):
            if isinstance(label, str):
                y_encoded[i] = char_to_int.get(label.upper(), 0)
            else:
                y_encoded[i] = int(label)
        
        return y_encoded
    
    def _download_dataset(self, url: str) -> str:
        """Download dataset from URL"""
        import tempfile
        
        tmpdir = tempfile.gettempdir()
        filename = os.path.basename(url)
        filepath = os.path.join(tmpdir, filename)
        
        if not os.path.exists(filepath):
            print(f"Downloading {url}...")
            urllib.request.urlretrieve(url, filepath)
        
        return filepath
    
    def save_dataset(self, X: np.ndarray, y: np.ndarray, save_dir: str, prefix=''):
        """
        Save dataset to disk
        
        Args:
            X: Features
            y: Labels
            save_dir: Directory to save
            prefix: Filename prefix
        """
        os.makedirs(save_dir, exist_ok=True)
        
        np.save(os.path.join(save_dir, f'{prefix}X.npy'), X)
        np.save(os.path.join(save_dir, f'{prefix}y.npy'), y)
        
        print(f"Saved dataset to {save_dir}")
