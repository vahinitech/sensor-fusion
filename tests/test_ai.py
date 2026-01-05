"""
Test suite for AI character recognition functionality
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from ai.character_recognition.preprocessor import SensorDataPreprocessor


class TestSensorDataPreprocessor:
    """Test cases for SensorDataPreprocessor"""

    def test_initialization(self):
        """Test preprocessor initialization"""
        preprocessor = SensorDataPreprocessor(max_timesteps=512, sampling_rate=208)
        assert preprocessor.max_timesteps == 512
        assert preprocessor.sampling_rate == 208

    def test_pad_sequence(self):
        """Test sequence padding"""
        preprocessor = SensorDataPreprocessor(max_timesteps=100, sampling_rate=208)
        
        # Create a short sequence
        sequence = np.random.randn(50, 13)
        padded = preprocessor.pad_sequence(sequence, max_len=100)
        
        assert padded.shape == (100, 13)
        # Check that the end is padded with zeros
        assert np.all(padded[50:] == 0)

    def test_truncate_sequence(self):
        """Test sequence truncation"""
        preprocessor = SensorDataPreprocessor(max_timesteps=100, sampling_rate=208)
        
        # Create a long sequence
        sequence = np.random.randn(150, 13)
        truncated = preprocessor.pad_sequence(sequence, max_len=100)
        
        assert truncated.shape == (100, 13)

    def test_normalize_features(self):
        """Test feature normalization"""
        preprocessor = SensorDataPreprocessor(max_timesteps=100, sampling_rate=208)
        
        # Create training data
        X_train = [np.random.randn(50, 13) for _ in range(5)]
        
        # Use batch_process to normalize
        X_processed = preprocessor.batch_process(X_train, normalize=True)
        
        # Check shape is preserved
        assert X_processed.shape[0] == 5
        assert X_processed.shape[2] == 13

    def test_prepare_sequence(self):
        """Test full sequence preparation pipeline"""
        preprocessor = SensorDataPreprocessor(max_timesteps=100, sampling_rate=208)
        
        # Create variable length sequences
        sequences = [
            np.random.randn(50, 13),
            np.random.randn(100, 13),
            np.random.randn(150, 13)
        ]
        
        prepared = preprocessor.batch_process(sequences, normalize=False)
        
        # All sequences should be same length
        assert prepared.shape[0] == 3
        assert prepared.shape[1] == 100
        assert prepared.shape[2] == 13


class TestCharacterRecognitionWorkflow:
    """Integration tests for character recognition workflow"""

    def test_data_collection_buffer(self):
        """Test sensor data collection buffer"""
        buffer_size = 512
        buffer = {
            'top_accel_x': [],
            'top_accel_y': [],
            'top_accel_z': [],
            'top_gyro_x': [],
            'top_gyro_y': [],
            'top_gyro_z': [],
        }
        
        # Simulate data collection
        for i in range(100):
            buffer['top_accel_x'].append(np.random.randn())
            buffer['top_accel_y'].append(np.random.randn())
            buffer['top_accel_z'].append(np.random.randn())
            buffer['top_gyro_x'].append(np.random.randn())
            buffer['top_gyro_y'].append(np.random.randn())
            buffer['top_gyro_z'].append(np.random.randn())
        
        assert len(buffer['top_accel_x']) == 100
        
        # Convert to numpy array for processing
        data_array = np.column_stack([
            buffer['top_accel_x'],
            buffer['top_accel_y'],
            buffer['top_accel_z'],
            buffer['top_gyro_x'],
            buffer['top_gyro_y'],
            buffer['top_gyro_z'],
        ])
        
        assert data_array.shape == (100, 6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
