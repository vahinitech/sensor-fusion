"""
Sensor Data Buffer Manager
Maintains circular buffers for real-time data
"""

from collections import deque
from typing import List

class SensorBuffers:
    """Manages all sensor data circular buffers"""
    
    def __init__(self, buffer_size: int = 256):
        """
        Initialize sensor buffers
        
        Args:
            buffer_size: Maximum number of samples to keep (256 = ~2.46s @ 104Hz)
        """
        self.buffer_size = buffer_size
        self._lock = __import__('threading').Lock()
        self._init_buffers()
    
    def _init_buffers(self):
        """Initialize all circular buffers"""
        # Timestamps
        self.timestamps = deque(maxlen=self.buffer_size)
        
        # Top IMU (LSM6DSO)
        self.top_accel_x = deque(maxlen=self.buffer_size)
        self.top_accel_y = deque(maxlen=self.buffer_size)
        self.top_accel_z = deque(maxlen=self.buffer_size)
        
        self.top_gyro_x = deque(maxlen=self.buffer_size)
        self.top_gyro_y = deque(maxlen=self.buffer_size)
        self.top_gyro_z = deque(maxlen=self.buffer_size)
        
        # Magnetometer (ST)
        self.mag_x = deque(maxlen=self.buffer_size)
        self.mag_y = deque(maxlen=self.buffer_size)
        self.mag_z = deque(maxlen=self.buffer_size)
        
        # Rear IMU (LSM6DSM)
        self.rear_accel_x = deque(maxlen=self.buffer_size)
        self.rear_accel_y = deque(maxlen=self.buffer_size)
        self.rear_accel_z = deque(maxlen=self.buffer_size)
        
        self.rear_gyro_x = deque(maxlen=self.buffer_size)
        self.rear_gyro_y = deque(maxlen=self.buffer_size)
        self.rear_gyro_z = deque(maxlen=self.buffer_size)
        
        # Force Sensor (HLP A04)
        self.force_x = deque(maxlen=self.buffer_size)
        self.force_y = deque(maxlen=self.buffer_size)
        self.force_z = deque(maxlen=self.buffer_size)
    
    def add_sample(self, sample_index: int, data: dict):
        """
        Add a complete sensor sample to buffers
        
        Args:
            sample_index: Sample counter
            data: Dictionary with keys for each sensor value
        """
        with self._lock:
            self.timestamps.append(sample_index)
        
        # Top IMU
        self.top_accel_x.append(data.get('top_accel_x', 0))
        self.top_accel_y.append(data.get('top_accel_y', 0))
        self.top_accel_z.append(data.get('top_accel_z', 0))
        
        self.top_gyro_x.append(data.get('top_gyro_x', 0))
        self.top_gyro_y.append(data.get('top_gyro_y', 0))
        self.top_gyro_z.append(data.get('top_gyro_z', 0))
        
        # Magnetometer
        self.mag_x.append(data.get('mag_x', 0))
        self.mag_y.append(data.get('mag_y', 0))
        self.mag_z.append(data.get('mag_z', 0))
        
        # Rear IMU
        self.rear_accel_x.append(data.get('rear_accel_x', 0))
        self.rear_accel_y.append(data.get('rear_accel_y', 0))
        self.rear_accel_z.append(data.get('rear_accel_z', 0))
        
        self.rear_gyro_x.append(data.get('rear_gyro_x', 0))
        self.rear_gyro_y.append(data.get('rear_gyro_y', 0))
        self.rear_gyro_z.append(data.get('rear_gyro_z', 0))
        
        # Force Sensor
        self.force_x.append(data.get('force_x', 0))
        self.force_y.append(data.get('force_y', 0))
        self.force_z.append(data.get('force_z', 0))
    
    def get_all(self) -> dict:
        """Get all buffers with thread safety (creates snapshots as lists)"""
        with self._lock:
            return {
                'timestamps': list(self.timestamps),
                'top_accel_x': list(self.top_accel_x),
                'top_accel_y': list(self.top_accel_y),
                'top_accel_z': list(self.top_accel_z),
                'top_gyro_x': list(self.top_gyro_x),
                'top_gyro_y': list(self.top_gyro_y),
                'top_gyro_z': list(self.top_gyro_z),
                'mag_x': list(self.mag_x),
                'mag_y': list(self.mag_y),
                'mag_z': list(self.mag_z),
                'rear_accel_x': list(self.rear_accel_x),
                'rear_accel_y': list(self.rear_accel_y),
                'rear_accel_z': list(self.rear_accel_z),
                'rear_gyro_x': list(self.rear_gyro_x),
                'rear_gyro_y': list(self.rear_gyro_y),
                'rear_gyro_z': list(self.rear_gyro_z),
                'force_x': list(self.force_x),
                'force_y': list(self.force_y),
                'force_z': list(self.force_z),
            }
    
    def clear(self):
        """Clear all buffers with thread safety"""
        with self._lock:
            for buffer in [
                self.timestamps,
                self.top_accel_x, self.top_accel_y, self.top_accel_z,
                self.top_gyro_x, self.top_gyro_y, self.top_gyro_z,
                self.mag_x, self.mag_y, self.mag_z,
                self.rear_accel_x, self.rear_accel_y, self.rear_accel_z,
                self.rear_gyro_x, self.rear_gyro_y, self.rear_gyro_z,
                self.force_x, self.force_y, self.force_z,
            ]:
                buffer.clear()
