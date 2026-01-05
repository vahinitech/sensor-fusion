"""
Sensor Service - Handles all sensor data processing
Framework independent - can be used with any UI
"""

from typing import Optional, Dict, Callable
import threading
from datetime import datetime

from core.sensor_parser import SensorParser
from core.data_reader import SerialReader
from utils.sensor_buffers import SensorBuffers
from utils.sensor_fusion_filters import SensorFusionManager


class SensorService:
    """Central service for sensor data management"""

    def __init__(self, buffer_size: int = 256):
        """
        Initialize sensor service

        Args:
            buffer_size: Size of circular buffers
        """
        self.buffers = SensorBuffers(buffer_size=buffer_size)
        self.parser = SensorParser()
        self.filter_manager = SensorFusionManager(filter_type="none")

        # Connection state
        self.data_reader = None
        self.running = False
        self.sample_count = 0
        self.start_time = None

        # Callbacks for UI updates
        self._data_callbacks = []
        self._connection_callbacks = []

        # Filter state
        self.apply_filter = False

    def register_data_callback(self, callback: Callable[[Dict], None]):
        """Register callback to be called when new data arrives"""
        self._data_callbacks.append(callback)

    def register_connection_callback(self, callback: Callable[[bool], None]):
        """Register callback to be called when connection state changes"""
        self._connection_callbacks.append(callback)

    def connect_serial(self, port: str, baudrate: int) -> bool:
        """
        Connect to serial port

        Args:
            port: COM port name
            baudrate: Baud rate

        Returns:
            True if connection successful
        """
        try:
            self.data_reader = SerialReader(port, baudrate, self._on_data_received)

            if not self.data_reader.connect():
                return False

            self.data_reader.start()
            self.running = True
            self.start_time = datetime.now()
            self.sample_count = 0

            # Notify connection callbacks
            for callback in self._connection_callbacks:
                callback(True)

            return True

        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def disconnect(self):
        """Disconnect from data source"""
        if self.data_reader:
            self.data_reader.stop()
            self.data_reader.close()
            self.data_reader = None

        self.running = False
        self.filter_manager.reset()

        # Notify connection callbacks
        for callback in self._connection_callbacks:
            callback(False)

    def _on_data_received(self, data):
        """Internal callback when data is received"""
        line = data if isinstance(data, str) else data.get("raw_line", "")
        if not line:
            return

        parsed = self.parser.parse_line(line)
        if not parsed:
            return

        # Apply filters if enabled
        if self.apply_filter:
            # Filter IMU data only
            parsed = self._apply_filters(parsed)

        # Add to buffers
        self.buffers.add_sample(self.sample_count, parsed)
        self.sample_count += 1

        # Notify all data callbacks
        for callback in self._data_callbacks:
            callback(parsed)

    def _apply_filters(self, data: Dict) -> Dict:
        """Apply sensor fusion filters to IMU data"""
        # Filter top IMU
        filtered_top = self.filter_manager.process_imu_data(
            data["top_accel_x"],
            data["top_accel_y"],
            data["top_accel_z"],
            data["top_gyro_x"],
            data["top_gyro_y"],
            data["top_gyro_z"],
        )

        # Filter rear IMU
        filtered_rear = self.filter_manager.process_imu_data(
            data["rear_accel_x"],
            data["rear_accel_y"],
            data["rear_accel_z"],
            data["rear_gyro_x"],
            data["rear_gyro_y"],
            data["rear_gyro_z"],
        )

        # Update data with filtered values
        data.update(
            {
                "top_accel_x": filtered_top["accel_x"],
                "top_accel_y": filtered_top["accel_y"],
                "top_accel_z": filtered_top["accel_z"],
                "top_gyro_x": filtered_top["gyro_x"],
                "top_gyro_y": filtered_top["gyro_y"],
                "top_gyro_z": filtered_top["gyro_z"],
                "rear_accel_x": filtered_rear["accel_x"],
                "rear_accel_y": filtered_rear["accel_y"],
                "rear_accel_z": filtered_rear["accel_z"],
                "rear_gyro_x": filtered_rear["gyro_x"],
                "rear_gyro_y": filtered_rear["gyro_y"],
                "rear_gyro_z": filtered_rear["gyro_z"],
            }
        )

        return data

    def set_filter_type(self, filter_type: str):
        """
        Set sensor fusion filter type

        Args:
            filter_type: 'none', 'kf', 'ekf', or 'cf'
        """
        if filter_type == "none":
            self.apply_filter = False
            self.filter_manager.set_filter_type("none")
        else:
            self.apply_filter = True
            self.filter_manager.set_filter_type(filter_type)
            self.filter_manager.reset()

    def get_buffer_data(self) -> Dict:
        """Get all buffered sensor data"""
        return self.buffers.get_all()

    def get_stats(self) -> Dict:
        """Get sensor statistics"""
        elapsed = 0
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()

        rate = self.sample_count / elapsed if elapsed > 0 else 0

        return {"sample_count": self.sample_count, "elapsed": elapsed, "rate": rate, "running": self.running}

    def clear_buffers(self):
        """Clear all sensor buffers"""
        self.buffers.clear()
        self.sample_count = 0
