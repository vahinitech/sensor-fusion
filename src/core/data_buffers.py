"""Circular buffers for sensor data storage"""
from collections import deque
import numpy as np
import threading


class SensorBuffers:
    """Manage circular buffers for all sensor data"""
    
    def __init__(self, buffer_size=256):
        self.buffer_size = buffer_size
        self._lock = threading.Lock()
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
        
        self.sample_count = 0
    
    def add_data(self, data, index):
        """Add parsed sensor data to buffers"""
        with self._lock:
            self.timestamps.append(index)
        
        # Top IMU
        self.top_accel_x.append(data['top_imu']['accel']['x'])
        self.top_accel_y.append(data['top_imu']['accel']['y'])
        self.top_accel_z.append(data['top_imu']['accel']['z'])
        self.top_gyro_x.append(data['top_imu']['gyro']['x'])
        self.top_gyro_y.append(data['top_imu']['gyro']['y'])
        self.top_gyro_z.append(data['top_imu']['gyro']['z'])
        
        # Magnetometer
        self.mag_x.append(data['magnetometer']['x'])
        self.mag_y.append(data['magnetometer']['y'])
        self.mag_z.append(data['magnetometer']['z'])
        
        # Rear IMU
        self.rear_accel_x.append(data['rear_imu']['accel']['x'])
        self.rear_accel_y.append(data['rear_imu']['accel']['y'])
        self.rear_accel_z.append(data['rear_imu']['accel']['z'])
        self.rear_gyro_x.append(data['rear_imu']['gyro']['x'])
        self.rear_gyro_y.append(data['rear_imu']['gyro']['y'])
        self.rear_gyro_z.append(data['rear_imu']['gyro']['z'])
        
        # Force Sensor
        self.force_x.append(data['force_sensor']['x'])
        self.force_y.append(data['force_sensor']['y'])
        self.force_z.append(data['force_sensor']['z'])
        
        self.sample_count += 1
    
    def calculate_magnitude(self, x_vals, y_vals, z_vals):
        """Calculate 3D vector magnitude"""
        if not x_vals or not y_vals or not z_vals:
            return []
        return [np.sqrt(x**2 + y**2 + z**2) 
                for x, y, z in zip(x_vals, y_vals, z_vals)]
    
    def get_data_arrays(self):
        """Get all data as arrays for plotting with thread safety"""
        with self._lock:
            return {
                'x_data': list(self.timestamps),
                'top_accel': (list(self.top_accel_x), list(self.top_accel_y), list(self.top_accel_z)),
                'top_gyro': (list(self.top_gyro_x), list(self.top_gyro_y), list(self.top_gyro_z)),
                'mag': (list(self.mag_x), list(self.mag_y), list(self.mag_z)),
                'rear_accel': (list(self.rear_accel_x), list(self.rear_accel_y), list(self.rear_accel_z)),
                'rear_gyro': (list(self.rear_gyro_x), list(self.rear_gyro_y), list(self.rear_gyro_z)),
                'force': (list(self.force_x), list(self.force_y), list(self.force_z))
            }
