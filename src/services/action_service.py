"""
Action Service - Handles pen action detection
Framework independent
"""

from typing import Dict, Callable, Optional
from actions.action_detector import ActionDetector


class ActionService:
    """Service for pen action detection and management"""

    def __init__(self, buffer_size: int = 30):
        """
        Initialize action service

        Args:
            buffer_size: Buffer size for action detection
        """
        self.detector = ActionDetector(buffer_size=buffer_size)
        self.current_action = "idle"
        self.prev_action = "idle"

        # Callbacks for action changes
        self._action_callbacks = []

    def register_action_callback(self, callback: Callable[[str, str], None]):
        """
        Register callback for action changes

        Args:
            callback: Function(current_action, previous_action)
        """
        self._action_callbacks.append(callback)

    def update(self, sensor_data: Dict):
        """
        Update action detector with new sensor data

        Args:
            sensor_data: Dictionary with sensor values
        """
        # Extract data for action detector
        top_accel = {
            "x": sensor_data.get("top_accel_x", 0),
            "y": sensor_data.get("top_accel_y", 0),
            "z": sensor_data.get("top_accel_z", 0),
        }

        rear_accel = {
            "x": sensor_data.get("rear_accel_x", 0),
            "y": sensor_data.get("rear_accel_y", 0),
            "z": sensor_data.get("rear_accel_z", 0),
        }

        top_gyro = {
            "x": sensor_data.get("top_gyro_x", 0),
            "y": sensor_data.get("top_gyro_y", 0),
            "z": sensor_data.get("top_gyro_z", 0),
        }

        rear_gyro = {
            "x": sensor_data.get("rear_gyro_x", 0),
            "y": sensor_data.get("rear_gyro_y", 0),
            "z": sensor_data.get("rear_gyro_z", 0),
        }

        mag = {"x": sensor_data.get("mag_x", 0), "y": sensor_data.get("mag_y", 0), "z": sensor_data.get("mag_z", 0)}

        force = {"x": sensor_data.get("force_raw", 0)}

        # Update detector
        self.detector.update(top_accel, rear_accel, top_gyro, rear_gyro, mag, force)

        # Get current action
        self.prev_action = self.current_action
        self.current_action = self.detector.get_action()

        # Notify callbacks if action changed
        if self.current_action != self.prev_action:
            for callback in self._action_callbacks:
                callback(self.current_action, self.prev_action)

    def get_action(self) -> str:
        """Get current pen action"""
        return self.current_action

    def get_display_text(self) -> str:
        """Get formatted text for display"""
        return self.detector.get_display_text()

    def is_writing(self) -> bool:
        """Check if pen is currently writing"""
        return self.current_action == "writing"

    def is_hovering(self) -> bool:
        """Check if pen is hovering"""
        return self.current_action == "hovering"

    def is_idle(self) -> bool:
        """Check if pen is idle"""
        return self.current_action == "idle"

    def reset(self):
        """Reset action detector"""
        self.detector.reset()
        self.current_action = "idle"
        self.prev_action = "idle"
