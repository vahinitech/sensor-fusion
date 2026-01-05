"""
Test suite for core sensor functionality
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from core.sensor_parser import SensorParser
from utils.sensor_buffers import SensorBuffers
from utils.battery_utils import BatteryConverter


class TestSensorParser:
    """Test cases for SensorParser"""

    def test_parse_valid_line(self):
        """Test parsing a valid sensor data line"""
        line = "5430303,2651,50,-3099,-10,-13,-2,5033,226,-6074,7,4,-7,-160,89,399,3209,1660,2821"
        result = SensorParser.parse_line(line)
        
        assert result is not None
        assert result['timestamp'] == 5430303
        assert result['top_accel_x'] == 2651
        assert result['top_accel_y'] == 50
        assert result['top_accel_z'] == -3099
        assert result['mag_x'] == 5033
        assert result['force_raw'] == 3209

    def test_parse_invalid_line(self):
        """Test parsing an invalid line returns None"""
        line = "invalid,data,line"
        result = SensorParser.parse_line(line)
        assert result is None

    def test_parse_empty_line(self):
        """Test parsing an empty line returns None"""
        result = SensorParser.parse_line("")
        assert result is None

    def test_parse_insufficient_fields(self):
        """Test parsing a line with insufficient fields"""
        line = "5430303,2651,50,-3099"
        result = SensorParser.parse_line(line)
        assert result is None


class TestSensorBuffers:
    """Test cases for SensorBuffers"""

    def test_buffer_initialization(self):
        """Test buffer initialization"""
        buffers = SensorBuffers(buffer_size=100)
        assert len(buffers.timestamps) == 0
        assert buffers.buffer_size == 100

    def test_add_sample(self):
        """Test adding a sample to buffers"""
        buffers = SensorBuffers(buffer_size=100)
        sample_data = {
            'timestamp': 5430303,
            'top_accel_x': 2651,
            'top_accel_y': 50,
            'top_accel_z': -3099,
            'top_gyro_x': -10,
            'top_gyro_y': -13,
            'top_gyro_z': -2,
            'mag_x': 5033,
            'mag_y': 226,
            'mag_z': -6074,
            'rear_accel_x': 7,
            'rear_accel_y': 4,
            'rear_accel_z': -7,
            'rear_gyro_x': -160,
            'rear_gyro_y': 89,
            'rear_gyro_z': 399,
            'force_x': 3209,
            'force_y': 1660,
            'force_z': 2821,
            'force_raw': 3209,
            'battery_mv': 7200,
            'force_mv': 3000
        }
        
        buffers.add_sample(0, sample_data)
        assert len(buffers.timestamps) == 1
        assert buffers.timestamps[0] == 0

    def test_buffer_overflow(self):
        """Test buffer maintains max size"""
        buffers = SensorBuffers(buffer_size=5)
        sample_data = {
            'timestamp': 0, 'top_accel_x': 0, 'top_accel_y': 0, 'top_accel_z': 0,
            'top_gyro_x': 0, 'top_gyro_y': 0, 'top_gyro_z': 0,
            'mag_x': 0, 'mag_y': 0, 'mag_z': 0,
            'rear_accel_x': 0, 'rear_accel_y': 0, 'rear_accel_z': 0,
            'rear_gyro_x': 0, 'rear_gyro_y': 0, 'rear_gyro_z': 0,
            'force_x': 0, 'force_y': 0, 'force_z': 0,
            'force_raw': 0, 'battery_mv': 0, 'force_mv': 0
        }
        
        # Add 10 samples to a buffer of size 5
        for i in range(10):
            sample_data['timestamp'] = i
            buffers.add_sample(i, sample_data)
        
        # Should only keep the last 5
        assert len(buffers.timestamps) == 5
        assert buffers.timestamps[-1] == 9

    def test_clear_buffers(self):
        """Test clearing all buffers"""
        buffers = SensorBuffers(buffer_size=100)
        sample_data = {
            'timestamp': 0, 'top_accel_x': 0, 'top_accel_y': 0, 'top_accel_z': 0,
            'top_gyro_x': 0, 'top_gyro_y': 0, 'top_gyro_z': 0,
            'mag_x': 0, 'mag_y': 0, 'mag_z': 0,
            'rear_accel_x': 0, 'rear_accel_y': 0, 'rear_accel_z': 0,
            'rear_gyro_x': 0, 'rear_gyro_y': 0, 'rear_gyro_z': 0,
            'force_x': 0, 'force_y': 0, 'force_z': 0,
            'force_raw': 0, 'battery_mv': 0, 'force_mv': 0
        }
        buffers.add_sample(0, sample_data)
        buffers.clear()
        assert len(buffers.timestamps) == 0


class TestBatteryConverter:
    """Test cases for BatteryConverter"""

    def test_voltage_to_percentage(self):
        """Test voltage to percentage conversion"""
        converter = BatteryConverter(min_mv=6000, max_mv=8400)
        
        # Test min voltage
        assert converter.voltage_to_percentage(6000) == 0.0
        
        # Test max voltage
        assert converter.voltage_to_percentage(8400) == 100.0
        
        # Test mid voltage
        mid_voltage = (6000 + 8400) / 2
        percentage = converter.voltage_to_percentage(mid_voltage)
        assert 45.0 <= percentage <= 55.0

    def test_battery_health_status(self):
        """Test battery health status"""
        converter = BatteryConverter(min_mv=6000, max_mv=8400)
        
        # Full battery
        status, color, icon = converter.get_battery_health_status(8400)
        assert "Full" in status or "Excellent" in status
        
        # Low battery
        status, color, icon = converter.get_battery_health_status(6200)
        assert "Low" in status or "Critical" in status

    def test_battery_icon(self):
        """Test battery icon selection"""
        icon_100 = BatteryConverter.get_battery_icon(100)
        icon_50 = BatteryConverter.get_battery_icon(50)
        icon_10 = BatteryConverter.get_battery_icon(10)
        
        # Icons should contain the battery emoji
        assert "🔋" in icon_100 or "🪫" in icon_100
        assert "🔋" in icon_50 or "🪫" in icon_50
        assert "🔋" in icon_10 or "🪫" in icon_10 or "⚠️" in icon_10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
