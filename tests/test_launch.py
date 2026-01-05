"""
Test suite for application launch and basic functionality
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest


class TestApplicationLaunch:
    """Test application launch and initialization"""

    def test_imports_core_modules(self):
        """Test that core modules can be imported"""
        from core import config, sensor_parser, serial_config

        assert config is not None
        assert sensor_parser is not None
        assert serial_config is not None

    def test_imports_utils_modules(self):
        """Test that utility modules can be imported"""
        from utils import battery_utils, sensor_buffers, sensor_fusion_filters

        assert battery_utils is not None
        assert sensor_buffers is not None
        assert sensor_fusion_filters is not None

    def test_imports_action_modules(self):
        """Test that action detection modules can be imported"""
        from actions import action_detector

        assert action_detector is not None

    def test_imports_service_modules(self):
        """Test that service modules can be imported"""
        try:
            from services import (
                action_service,
                ai_service,
                battery_service,
                sensor_service,
            )

            assert action_service is not None
            assert ai_service is not None
            assert battery_service is not None
            assert sensor_service is not None
        except ImportError as e:
            # Some service dependencies might not be available
            if "serial_reader" not in str(e) and "data_reader" not in str(e):
                raise
            pytest.skip(f"Service module imports skipped: {e}")

    def test_run_py_imports_gui(self):
        """Test that run.py can import GUI module"""
        # This tests the main entry point
        try:
            from gui.gui_app import SensorDashboardGUI

            assert SensorDashboardGUI is not None
        except ImportError as e:
            # GUI might have dependencies like tkinter or serial_reader
            if "tkinter" not in str(e) and "serial_reader" not in str(e):
                raise
            pytest.skip(f"GUI module import skipped: {e}")

    def test_config_loads(self):
        """Test that default config can be loaded"""
        from core.config import Config

        config = Config("data/config.json")
        assert config.config is not None
        assert "dashboard" in config.config or "data_source" in config.config

    def test_sensor_parser_works(self):
        """Test that sensor parser can parse data"""
        from core.sensor_parser import SensorParser

        # Valid line
        line = "5430303,2651,50,-3099,-10,-13,-2,5033,226,-6074,7,4,-7,-160,89,399,3209,1660,2821"
        result = SensorParser.parse_line(line)
        assert result is not None
        assert "timestamp" in result

    def test_battery_utils_initialized(self):
        """Test battery utilities can be initialized"""
        from utils.battery_utils import BatteryConverter

        converter = BatteryConverter(min_mv=6000, max_mv=8400)
        assert converter is not None
        percentage = converter.voltage_to_percentage(7200)
        assert 0 <= percentage <= 100

    def test_sensor_buffers_initialized(self):
        """Test sensor buffers can be initialized"""
        from utils.sensor_buffers import SensorBuffers

        buffers = SensorBuffers(buffer_size=256)
        assert buffers is not None
        assert len(buffers.timestamps) == 0

    def test_action_detector_initialized(self):
        """Test action detector can be initialized"""
        from actions.action_detector import ActionDetector

        detector = ActionDetector(buffer_size=30)
        assert detector is not None
        assert detector.current_action is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
