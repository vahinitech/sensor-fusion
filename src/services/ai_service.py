"""
AI Service - Handles character recognition and AI inference
Framework independent
"""

from typing import Optional, Tuple, Dict, Callable
import os
from gui.character_recognition_integration import CharacterRecognitionIntegration


class AIService:
    """Service for AI character recognition"""

    def __init__(self, model_path: Optional[str] = None, buffer_size: int = 512):
        """
        Initialize AI service

        Args:
            model_path: Path to trained model file
            buffer_size: Buffer size for character data collection
        """
        self.char_recognition = CharacterRecognitionIntegration(model_path=model_path, buffer_size=buffer_size)

        # Callbacks for recognition events
        self._recognition_callbacks = []
        self._writing_callbacks = []

        self.is_collecting = False
        self.last_recognized_char = None
        self.last_confidence = 0.0

    def register_recognition_callback(self, callback: Callable[[str, float], None]):
        """
        Register callback for character recognition

        Args:
            callback: Function(character, confidence)
        """
        self._recognition_callbacks.append(callback)

    def register_writing_callback(self, callback: Callable[[bool], None]):
        """
        Register callback for writing state changes

        Args:
            callback: Function(is_writing)
        """
        self._writing_callbacks.append(callback)

    def start_writing(self):
        """Signal that writing has started"""
        self.char_recognition.start_writing()
        self.is_collecting = True

        # Notify callbacks
        for callback in self._writing_callbacks:
            callback(True)

    def stop_writing(self):
        """Signal that writing has stopped"""
        self.char_recognition.stop_writing()

    def add_sensor_data(self, sensor_data: Dict):
        """Add sensor data to writing buffer"""
        if self.char_recognition.is_writing:
            self.char_recognition.add_sensor_data(sensor_data)

    def check_pause_complete(self) -> str:
        """
        Check if pause is complete and trigger recognition

        Returns:
            'complete', 'waiting', or 'none'
        """
        result = self.char_recognition.check_pause_complete()

        if result == "complete":
            # Trigger recognition
            char, confidence = self.char_recognition.end_writing()

            if char:
                self.last_recognized_char = char
                self.last_confidence = confidence
                self.is_collecting = False

                # Notify callbacks
                for callback in self._recognition_callbacks:
                    callback(char, confidence)

                # Notify writing stopped
                for callback in self._writing_callbacks:
                    callback(False)

        return result

    def get_buffer_length(self) -> int:
        """Get current writing buffer length"""
        if not self.char_recognition.writing_buffer:
            return 0
        return len(self.char_recognition.writing_buffer.get("top_accel_x", []))

    def get_display_state(self) -> Dict:
        """
        Get current display state for UI

        Returns:
            Dictionary with display information
        """
        if not self.char_recognition.model:
            if self.char_recognition.is_writing:
                buffer_len = self.get_buffer_length()
                return {
                    "text": f"Collecting...\n\n({buffer_len} samples)\n\nNo Model\nto Recognize",
                    "has_model": False,
                }
            else:
                return {"text": "No Model\n\nTrain first using:\ntrain_character_\nmodel.py", "has_model": False}

        # Model is loaded
        if self.char_recognition.is_writing:
            buffer_len = self.get_buffer_length()
            return {
                "text": f"✏️ Writing...\n\n📊 Collecting Data\n({buffer_len} samples)",
                "has_model": True,
                "is_writing": True,
            }
        elif self.last_recognized_char:
            return {
                "text": f"✓ Recognized:\n\n🔤 {self.last_recognized_char}\n\n📈 Confidence:\n{self.last_confidence:.1%}",
                "has_model": True,
                "recognized": True,
                "character": self.last_recognized_char,
                "confidence": self.last_confidence,
            }
        else:
            return {
                "text": "✓ Ready\n\n📦 Model: Loaded\n\n✍️  Write a character\nto recognize",
                "has_model": True,
                "ready": True,
            }

    def has_model(self) -> bool:
        """Check if AI model is loaded"""
        return self.char_recognition.model is not None

    def reset(self):
        """Reset AI service state"""
        self.is_collecting = False
        self.last_recognized_char = None
        self.last_confidence = 0.0
