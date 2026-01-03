"""
Character Recognition Integration for GUI
Handles model loading, real-time prediction, and display
"""

import numpy as np
import os
from typing import Optional, Dict, Tuple
import threading
from collections import deque


class CharacterRecognitionIntegration:
    """Integrates character recognition into the GUI"""
    
    def __init__(self, model_path: Optional[str] = None, buffer_size: int = 256):
        """
        Initialize character recognition integration
        
        Args:
            model_path: Path to trained model (.h5 file)
            buffer_size: Size of sensor buffer for recognition
        """
        self.model_path = model_path
        self.model = None
        self.preprocessor = None
        self.buffer_size = buffer_size
        self.lock = threading.Lock()
        
        # Sampling rate and timing
        self.target_sampling_rate = 104  # Hz - same as training data
        self.sample_interval = 1.0 / self.target_sampling_rate  # ~9.6 ms
        self.last_sample_time = None
        
        # Pause detection for character recognition
        self.pause_threshold = 0.5  # 500ms pause to trigger recognition
        self.writing_stopped_time = None  # When pen transitioned away from writing
        self.waiting_for_pause = False  # Waiting for pause to trigger recognition
        
        # Sensor data buffer for character collection
        self.writing_buffer = {
            'top_accel_x': deque(maxlen=buffer_size),
            'top_accel_y': deque(maxlen=buffer_size),
            'top_accel_z': deque(maxlen=buffer_size),
            'top_gyro_x': deque(maxlen=buffer_size),
            'top_gyro_y': deque(maxlen=buffer_size),
            'top_gyro_z': deque(maxlen=buffer_size),
            'rear_accel_x': deque(maxlen=buffer_size),
            'rear_accel_y': deque(maxlen=buffer_size),
            'rear_accel_z': deque(maxlen=buffer_size),
            'mag_x': deque(maxlen=buffer_size),
            'mag_y': deque(maxlen=buffer_size),
            'mag_z': deque(maxlen=buffer_size),
            'force_x': deque(maxlen=buffer_size),
        }
        
        # Character history
        self.recognized_characters = deque(maxlen=100)
        self.character_confidences = deque(maxlen=100)
        
        # Current writing state
        self.is_writing = False
        self.recognized_char = None
        self.recognized_confidence = 0.0
        
        self.characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load trained character recognition model
        
        Args:
            model_path: Path to .h5 model file
        """
        try:
            from ai.character_recognition.model import CharacterRecognitionModel
            from ai.character_recognition.preprocessor import SensorDataPreprocessor
            
            if not os.path.exists(model_path):
                # Silently skip if model doesn't exist yet
                return False
            
            # Load preprocessor with same config
            self.preprocessor = SensorDataPreprocessor(
                max_timesteps=512,
                sampling_rate=104
            )
            
            # Load model with proper initialization
            self.model = CharacterRecognitionModel(
                timesteps=512,
                n_features=13,
                num_classes=26,
                model_path=model_path
            )
            
            print(f"✓ Character recognition model loaded from {model_path}")
            return True
        
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            return False
    
    def add_sensor_data(self, sensor_dict: Dict):
        """
        Add new sensor data to writing buffer (only at 104Hz rate)
        
        Args:
            sensor_dict: Dictionary with sensor values and 'timestamp' key
        """
        if not self.is_writing:
            return
        
        import time
        
        with self.lock:
            current_time = time.time()
            
            # Initialize timing on first sample
            if self.last_sample_time is None:
                self.last_sample_time = current_time
                actual_interval = 0
            else:
                actual_interval = current_time - self.last_sample_time
            
            # Only add sample if enough time has passed (104Hz ≈ 9.6ms per sample)
            # Allow some tolerance: 8-11ms
            if actual_interval >= (self.sample_interval * 0.8):  # At least 80% of expected interval
                for key in self.writing_buffer:
                    if key in sensor_dict:
                        self.writing_buffer[key].append(sensor_dict[key])
                
                self.last_sample_time = current_time
    
    def start_writing(self):
        """Signal that pen is down and writing has started"""
        with self.lock:
            self.is_writing = True
            self.waiting_for_pause = False
            self.writing_stopped_time = None
            self.last_sample_time = None  # Reset timing for new character
            # Clear buffer for new character
            for buffer in self.writing_buffer.values():
                buffer.clear()
            # Keep last recognized character visible until new writing starts
            # Don't clear: self.recognized_char and self.recognized_confidence
    
    def stop_writing(self):
        """Signal that pen has stopped writing and start pause detection"""
        with self.lock:
            self.is_writing = False
            self.waiting_for_pause = True
            self.writing_stopped_time = None  # Will be set when pause begins
    
    def check_pause_complete(self) -> bool:
        """
        Check if sufficient pause has occurred to trigger recognition
        
        Returns:
            True if pause is complete and recognition should trigger
        """
        import time
        
        with self.lock:
            if not self.waiting_for_pause:
                return False
            
            if self.writing_stopped_time is None:
                # First call after stopping - record the time
                self.writing_stopped_time = time.time()
                return False
            
            # Check if pause duration exceeded threshold
            pause_duration = time.time() - self.writing_stopped_time
            
            if pause_duration >= self.pause_threshold:
                self.waiting_for_pause = False
                self.writing_stopped_time = None
                return True
            
            return False
    
    def end_writing(self) -> Tuple[Optional[str], float]:
        """
        Signal that pen is up and recognize the written character
        
        Returns:
            (recognized_character, confidence) or (None, 0.0)
        """
        with self.lock:
            self.is_writing = False
            
            if not self.model:
                print("❌ No model loaded!")
                return None, 0.0
            
            # Need minimum data
            buffer_len = len(self.writing_buffer['top_accel_x'])
            if buffer_len < 50:  # Minimum 50 samples (~0.5s at 104Hz)
                print(f"❌ Insufficient data: only {buffer_len} samples (need 50+)")
                return None, 0.0
            
            try:
                print(f"\n{'='*60}")
                print(f"🔍 CHARACTER RECOGNITION DEBUG")
                print(f"{'='*60}")
                
                # Convert buffer to numpy array
                sensor_data = np.zeros((buffer_len, 13), dtype=np.float32)
                sensor_data[:, 0] = list(self.writing_buffer['top_accel_x'])
                sensor_data[:, 1] = list(self.writing_buffer['top_accel_y'])
                sensor_data[:, 2] = list(self.writing_buffer['top_accel_z'])
                sensor_data[:, 3] = list(self.writing_buffer['top_gyro_x'])
                sensor_data[:, 4] = list(self.writing_buffer['top_gyro_y'])
                sensor_data[:, 5] = list(self.writing_buffer['top_gyro_z'])
                sensor_data[:, 6] = list(self.writing_buffer['rear_accel_x'])
                sensor_data[:, 7] = list(self.writing_buffer['rear_accel_y'])
                sensor_data[:, 8] = list(self.writing_buffer['rear_accel_z'])
                sensor_data[:, 9] = list(self.writing_buffer['mag_x'])
                sensor_data[:, 10] = list(self.writing_buffer['mag_y'])
                sensor_data[:, 11] = list(self.writing_buffer['mag_z'])
                sensor_data[:, 12] = list(self.writing_buffer['force_x'])
                
                print(f"📊 Raw Data Shape: {sensor_data.shape}")
                print(f"   Samples collected: {buffer_len}")
                print(f"   Duration: ~{buffer_len/104:.2f} seconds @ 104Hz")
                print(f"   Data range: [{sensor_data.min():.2f}, {sensor_data.max():.2f}]")
                print(f"   Data stats: mean={sensor_data.mean():.2f}, std={sensor_data.std():.2f}")
                
                # Preprocess
                print(f"\n⚙️  Preprocessing...")
                sensor_data_padded = self.preprocessor.pad_sequence(sensor_data)
                print(f"   After padding: {sensor_data_padded.shape}")
                
                sensor_data_normalized = self.preprocessor.normalize_data(
                    sensor_data_padded.reshape(1, 512, 13)
                )
                print(f"   After normalization: {sensor_data_normalized.shape}")
                print(f"   Normalized range: [{sensor_data_normalized.min():.4f}, {sensor_data_normalized.max():.4f}]")
                print(f"   Normalized stats: mean={sensor_data_normalized.mean():.4f}, std={sensor_data_normalized.std():.4f}")
                
                # Predict
                print(f"\n🤖 Running model prediction...")
                print(f"   Model type: {type(self.model.model)}")
                print(f"   Input shape to model: {sensor_data_normalized[0].shape}")
                
                # Get raw predictions (before argmax)
                input_batch = sensor_data_normalized[0:1]  # Keep batch dimension
                raw_predictions = self.model.model.predict(input_batch, verbose=0)
                print(f"   Raw model output shape: {raw_predictions.shape}")
                print(f"   Raw predictions: {raw_predictions[0]}")
                print(f"   Top 5 classes: {np.argsort(raw_predictions[0])[-5:][::-1]}")
                print(f"   Top 5 confidences: {np.sort(raw_predictions[0])[-5:][::-1]}")
                
                class_idx, confidence = self.model.predict_single(
                    sensor_data_normalized[0], 
                    return_confidence=True
                )
                
                # Convert class index (0-25) to character (A-Z)
                char = self.characters[class_idx]
                
                print(f"\n✅ RECOGNITION RESULT:")
                print(f"   Predicted Class Index: {class_idx}")
                print(f"   Predicted Character: '{char}'")
                print(f"   Confidence: {confidence:.2%}")
                
                # Check confidence threshold (80%)
                confidence_threshold = 0.80
                if confidence >= confidence_threshold:
                    print(f"   ✅ ACCEPTED - Above {confidence_threshold:.0%} threshold")
                    print(f"{'='*60}\n")
                    
                    self.recognized_char = char
                    self.recognized_confidence = confidence
                    self.recognized_characters.append(char)
                    self.character_confidences.append(confidence)
                    
                    return char, confidence
                else:
                    print(f"   ❌ REJECTED - Below {confidence_threshold:.0%} threshold")
                    print(f"{'='*60}\n")
                    
                    # Return None to indicate low confidence
                    return None, confidence
            
            except Exception as e:
                print(f"Error recognizing character: {e}")
                return None, 0.0
    
    def get_character_history(self, last_n: int = 10) -> str:
        """
        Get recognized character history as string
        
        Args:
            last_n: Number of last characters to show
        
        Returns:
            String of recognized characters
        """
        chars = list(self.recognized_characters)[-last_n:]
        return ''.join(chars) if chars else "[ No characters yet ]"
    
    def get_stats(self) -> Dict:
        """
        Get character recognition statistics
        
        Returns:
            Dictionary with stats
        """
        if self.character_confidences:
            avg_conf = np.mean(list(self.character_confidences))
            max_conf = np.max(list(self.character_confidences))
            min_conf = np.min(list(self.character_confidences))
        else:
            avg_conf = max_conf = min_conf = 0.0
        
        return {
            'total_recognized': len(self.recognized_characters),
            'avg_confidence': avg_conf,
            'max_confidence': max_conf,
            'min_confidence': min_conf,
            'is_writing': self.is_writing,
            'last_character': self.recognized_char,
            'last_confidence': self.recognized_confidence
        }
    
    def get_display_text(self) -> str:
        """
        Get text for GUI display
        
        Returns:
            Formatted display text
        """
        stats = self.get_stats()
        
        writing_indicator = "🖊️ WRITING" if stats['is_writing'] else "✋ IDLE"
        
        text = f"""CHARACTER RECOGNITION
━━━━━━━━━━━━━━━━━━━
Status: {writing_indicator}

LAST RECOGNIZED
━━━━━━━━━━━━━━━━━━━
Character: {stats['last_character'] or '-'}
Confidence: {stats['last_confidence']:.1%}

STATISTICS
━━━━━━━━━━━━━━━━━━━
Total Recognized: {stats['total_recognized']}
Avg Confidence: {stats['avg_confidence']:.1%}

HISTORY (Last 20)
━━━━━━━━━━━━━━━━━━━
{self.get_character_history(20)}
"""
        return text
    
    def reset(self):
        """Reset all buffers and history"""
        with self.lock:
            for buffer in self.writing_buffer.values():
                buffer.clear()
            self.recognized_characters.clear()
            self.character_confidences.clear()
            self.recognized_char = None
            self.recognized_confidence = 0.0
            self.is_writing = False
