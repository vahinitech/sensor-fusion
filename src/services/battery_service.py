"""
Battery Service - Handles battery monitoring and percentage calculation
Framework independent
"""

from typing import Dict, Tuple
from utils.battery_utils import BatteryConverter


class BatteryService:
    """Service for battery monitoring and status"""

    def __init__(self, min_mv: int = 6000, max_mv: int = 8400, battery_type: str = "lipo"):
        """
        Initialize battery service

        Args:
            min_mv: Minimum battery voltage in mV
            max_mv: Maximum battery voltage in mV
            battery_type: Type of battery
        """
        self.converter = BatteryConverter(min_mv=min_mv, max_mv=max_mv, battery_type=battery_type)
        self.current_voltage = 0
        self.current_percentage = 0
        self.current_status = "Unknown"
        self.current_color = "gray"

    def update(self, voltage_mv: float) -> Dict:
        """
        Update battery status with new voltage reading

        Args:
            voltage_mv: Battery voltage in millivolts

        Returns:
            Dictionary with battery info
        """
        self.current_voltage = voltage_mv
        self.current_percentage = self.converter.voltage_to_percentage(voltage_mv)
        self.current_status, self.current_color, _ = self.converter.get_battery_health_status(voltage_mv)

        return self.get_status()

    def get_status(self) -> Dict:
        """
        Get current battery status

        Returns:
            Dictionary with voltage, percentage, status, color, icon
        """
        icon = BatteryConverter.get_battery_icon(self.current_percentage)

        return {
            "voltage_mv": self.current_voltage,
            "percentage": self.current_percentage,
            "status": self.current_status,
            "color": self.current_color,
            "icon": icon,
            "display_text": f"{icon} {self.current_percentage:.0f}%",
        }

    def is_critical(self) -> bool:
        """Check if battery is in critical state"""
        return self.current_percentage < 10

    def is_low(self) -> bool:
        """Check if battery is low"""
        return self.current_percentage < 20

    def get_display_color(self) -> str:
        """Get color code for display"""
        return self.current_color
