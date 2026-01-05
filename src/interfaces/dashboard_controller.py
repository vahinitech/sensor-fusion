"""
Dashboard Controller - Coordinates services and UI
Framework independent business logic orchestration
"""

from typing import Optional
from interfaces.dashboard_interface import DashboardInterface
from services import SensorService, BatteryService, ActionService, AIService


class DashboardController:
    """
    Central controller that coordinates all services and UI
    This is the bridge between business logic and any UI implementation
    """

    def __init__(self, ui: DashboardInterface, model_path: Optional[str] = None):
        """
        Initialize dashboard controller

        Args:
            ui: Dashboard UI implementation
            model_path: Path to AI model (optional)
        """
        self.ui = ui

        # Initialize services
        self.sensor_service = SensorService(buffer_size=256)
        self.battery_service = BatteryService(min_mv=6000, max_mv=8400)
        self.action_service = ActionService(buffer_size=30)
        self.ai_service = AIService(model_path=model_path, buffer_size=512)

        # Register callbacks
        self._register_callbacks()

        # State tracking
        self.prev_pen_action = "idle"

    def _register_callbacks(self):
        """Register all service callbacks"""
        # Sensor data updates
        self.sensor_service.register_data_callback(self._on_sensor_data)

        # Connection status changes
        self.sensor_service.register_connection_callback(self._on_connection_change)

        # Action changes
        self.action_service.register_action_callback(self._on_action_change)

        # AI recognition
        self.ai_service.register_recognition_callback(self._on_character_recognized)
        self.ai_service.register_writing_callback(self._on_writing_state_change)

    def _on_sensor_data(self, data: dict):
        """Handle new sensor data"""
        # Update battery service
        battery_info = self.battery_service.update(data.get("battery_mv", 0))

        # Update action service
        self.action_service.update(data)

        # Update AI service if writing
        if self.ai_service.char_recognition.is_writing:
            self.ai_service.add_sensor_data(data)

        # Check for pause completion (character recognition trigger)
        pause_result = self.ai_service.check_pause_complete()

        # Update UI with latest data periodically (not every sample to avoid overhead)
        # This is handled by the UI's update loop

    def _on_connection_change(self, connected: bool):
        """Handle connection status change"""
        self.ui.show_connection_status(connected)

        if not connected:
            # Reset services
            self.action_service.reset()
            self.ai_service.reset()

    def _on_action_change(self, current_action: str, prev_action: str):
        """Handle pen action change"""
        # Writing started
        if current_action == "writing" and prev_action != "writing":
            print(f"\n✏️  Pen down - Writing started")
            self.ai_service.start_writing()

        # Writing stopped
        elif prev_action == "writing" and current_action != "writing":
            sample_count = self.ai_service.get_buffer_length()
            duration = sample_count / 208.0
            print(f"⏸️  Pen stopped writing (transitioned to: {current_action})")
            print(f"📊 Collected {sample_count} samples (~{duration:.2f}s of writing)")
            self.ai_service.stop_writing()
            print(f"⏳ Waiting for pause to trigger recognition...")

        self.prev_pen_action = current_action

    def _on_character_recognized(self, character: str, confidence: float):
        """Handle character recognition result"""
        print(f"\n{'='*60}")
        print(f"🎯 ✨ RECOGNIZED: '{character}' (confidence: {confidence:.1%}) ✨")
        print(f"{'='*60}\n")

    def _on_writing_state_change(self, is_writing: bool):
        """Handle writing state change"""
        pass  # Currently handled by action change

    # Public API for UI

    def connect_serial(self, port: str, baudrate: int) -> bool:
        """Connect to serial port"""
        return self.sensor_service.connect_serial(port, baudrate)

    def disconnect(self):
        """Disconnect from data source"""
        self.sensor_service.disconnect()

    def set_filter_type(self, filter_type: str):
        """Set sensor fusion filter type"""
        self.sensor_service.set_filter_type(filter_type)

    def get_sensor_data(self) -> dict:
        """Get current sensor buffer data"""
        return self.sensor_service.get_buffer_data()

    def get_battery_status(self) -> dict:
        """Get current battery status"""
        return self.battery_service.get_status()

    def get_action_display(self) -> dict:
        """Get action display information"""
        return {"action": self.action_service.get_action(), "display_text": self.action_service.get_display_text()}

    def get_ai_display(self) -> dict:
        """Get AI display state"""
        return self.ai_service.get_display_state()

    def get_stats(self) -> dict:
        """Get sensor statistics"""
        return self.sensor_service.get_stats()

    def is_running(self) -> bool:
        """Check if system is running"""
        return self.sensor_service.running
