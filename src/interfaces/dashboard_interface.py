"""
Dashboard Interface - Abstract base for any UI implementation
Defines the contract that any dashboard UI must implement
"""

from abc import ABC, abstractmethod
from typing import Dict


class DashboardInterface(ABC):
    """Abstract base class for dashboard implementations"""

    @abstractmethod
    def initialize(self):
        """Initialize the dashboard UI"""
        pass

    @abstractmethod
    def update_sensor_plots(self, data: Dict):
        """
        Update sensor data plots

        Args:
            data: Dictionary with all sensor buffer data
        """
        pass

    @abstractmethod
    def update_battery_display(self, battery_info: Dict):
        """
        Update battery percentage display

        Args:
            battery_info: Dictionary with voltage, percentage, status, color, icon
        """
        pass

    @abstractmethod
    def update_action_display(self, action: str, display_text: str):
        """
        Update pen action display

        Args:
            action: Current action ('idle', 'hovering', 'writing')
            display_text: Formatted text for display
        """
        pass

    @abstractmethod
    def update_ai_display(self, display_state: Dict):
        """
        Update AI character recognition display

        Args:
            display_state: Dictionary with AI display information
        """
        pass

    @abstractmethod
    def update_stats(self, stats: Dict):
        """
        Update statistics display

        Args:
            stats: Dictionary with sample_count, rate, elapsed
        """
        pass

    @abstractmethod
    def show_connection_status(self, connected: bool):
        """
        Update connection status indicator

        Args:
            connected: True if connected, False otherwise
        """
        pass

    @abstractmethod
    def show_error(self, title: str, message: str):
        """
        Display error message to user

        Args:
            title: Error title
            message: Error message
        """
        pass

    @abstractmethod
    def show_info(self, title: str, message: str):
        """
        Display info message to user

        Args:
            title: Info title
            message: Info message
        """
        pass

    @abstractmethod
    def run(self):
        """Start the dashboard UI main loop"""
        pass

    @abstractmethod
    def stop(self):
        """Stop the dashboard and cleanup"""
        pass
