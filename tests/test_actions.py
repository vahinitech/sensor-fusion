"""
Test suite for action detection functionality
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from actions.action_detector import ActionDetector


class TestActionDetector:
    """Test cases for ActionDetector"""

    def test_initialization(self):
        """Test action detector initialization"""
        detector = ActionDetector(buffer_size=30)
        assert detector.buffer_size == 30
        assert detector.current_action == "idle"

    def test_update_with_idle_data(self):
        """Test detection with idle/stationary data"""
        detector = ActionDetector(buffer_size=30)
        
        # Send stationary data (low acceleration, low gyro)
        for _ in range(35):  # More than buffer size to ensure detection
            detector.update(
                top_accel={'x': 0, 'y': 0, 'z': -9800},  # Gravity only
                rear_accel={'x': 0, 'y': 0, 'z': -9800},
                top_gyro={'x': 0, 'y': 0, 'z': 0},
                rear_gyro={'x': 0, 'y': 0, 'z': 0},
                mag={'x': 5000, 'y': 0, 'z': 0},
                force={'x': 2000, 'y': 2000, 'z': 2000}
            )
        
        # Should have a current action state
        assert detector.current_action is not None

    def test_update_with_movement_data(self):
        """Test detection with movement data"""
        detector = ActionDetector(buffer_size=30)
        
        # Send data with significant acceleration changes
        for i in range(35):
            detector.update(
                top_accel={'x': 5000 * np.sin(i), 'y': 5000 * np.cos(i), 'z': -9800},
                rear_accel={'x': 4000 * np.sin(i), 'y': 4000 * np.cos(i), 'z': -9800},
                top_gyro={'x': 500, 'y': 300, 'z': 200},
                rear_gyro={'x': 450, 'y': 280, 'z': 180},
                mag={'x': 5000, 'y': 0, 'z': 0},
                force={'x': 2000, 'y': 2000, 'z': 2000}
            )
        
        # Should detect some form of movement
        assert detector.current_action != "idle"

    def test_get_display_text(self):
        """Test display text generation"""
        detector = ActionDetector(buffer_size=30)
        
        # Update with some data
        detector.update(
            top_accel={'x': 0, 'y': 0, 'z': -9800},
            rear_accel={'x': 0, 'y': 0, 'z': -9800},
            top_gyro={'x': 0, 'y': 0, 'z': 0},
            rear_gyro={'x': 0, 'y': 0, 'z': 0},
            mag={'x': 5000, 'y': 0, 'z': 0},
            force={'x': 2000, 'y': 2000, 'z': 2000}
        )
        
        text = detector.get_display_text()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_reset(self):
        """Test resetting the detector"""
        detector = ActionDetector(buffer_size=30)
        
        # Add some data
        for _ in range(10):
            detector.update(
                top_accel={'x': 1000, 'y': 1000, 'z': -9800},
                rear_accel={'x': 1000, 'y': 1000, 'z': -9800},
                top_gyro={'x': 100, 'y': 100, 'z': 100},
                rear_gyro={'x': 100, 'y': 100, 'z': 100},
                mag={'x': 5000, 'y': 0, 'z': 0},
                force={'x': 2000, 'y': 2000, 'z': 2000}
            )
        
        # Reset
        detector.reset()
        
        # Should be back to idle
        assert detector.current_action == "idle"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
