"""
Battery Voltage Conversion Utilities
Converts battery voltage measurements to percentage
"""


class BatteryConverter:
    """Converts battery mVolt measurements to percentage"""

    # Typical lithium-ion battery voltage ranges (adjust for your specific battery)
    # These values are for a 2S LiPo battery (7.4V nominal, 8.4V max, 6.0V min cutoff)
    DEFAULT_MIN_VOLTAGE = 6000  # mV (6.0V) - minimum safe voltage
    DEFAULT_MAX_VOLTAGE = 8400  # mV (8.4V) - fully charged
    DEFAULT_NOMINAL_VOLTAGE = 7400  # mV (7.4V) - nominal voltage

    def __init__(
        self, min_mv: int = DEFAULT_MIN_VOLTAGE, max_mv: int = DEFAULT_MAX_VOLTAGE, battery_type: str = "lipo"
    ):
        """
        Initialize battery converter

        Args:
            min_mv: Minimum voltage in mV (default: 6000 for 2S LiPo)
            max_mv: Maximum voltage in mV (default: 8400 for 2S LiPo)
            battery_type: Type of battery ('lipo', 'li-ion', 'alkaline')
        """
        self.min_mv = min_mv
        self.max_mv = max_mv
        self.battery_type = battery_type
        self.voltage_range = max_mv - min_mv

    def voltage_to_percentage(self, voltage_mv: float) -> float:
        """
        Convert voltage reading to battery percentage

        Args:
            voltage_mv: Voltage in millivolts

        Returns:
            Battery percentage (0-100)
        """
        if voltage_mv <= self.min_mv:
            return 0.0
        elif voltage_mv >= self.max_mv:
            return 100.0
        else:
            # Linear interpolation
            percentage = ((voltage_mv - self.min_mv) / self.voltage_range) * 100.0
            return max(0.0, min(100.0, percentage))

    def voltage_to_percentage_nonlinear(self, voltage_mv: float) -> float:
        """
        Convert voltage to percentage using non-linear curve
        (better matches actual LiPo discharge behavior)

        Args:
            voltage_mv: Voltage in millivolts

        Returns:
            Battery percentage (0-100)
        """
        # Normalize voltage to 0-1 range
        normalized = (voltage_mv - self.min_mv) / self.voltage_range
        normalized = max(0.0, min(1.0, normalized))

        # Non-linear mapping (exponential curve)
        # LiPo batteries have steep drop-off at low voltages
        percentage = (normalized**0.5) * 100.0
        return percentage

    def get_battery_health_status(self, voltage_mv: float) -> tuple:
        """
        Get battery health status based on voltage

        Args:
            voltage_mv: Voltage in millivolts

        Returns:
            (status_string, color_code, percentage)
        """
        percentage = self.voltage_to_percentage(voltage_mv)

        if percentage >= 80:
            return ("Excellent", "#00AA00", percentage)  # Green
        elif percentage >= 60:
            return ("Good", "#00FF00", percentage)  # Bright green
        elif percentage >= 40:
            return ("Fair", "#FFFF00", percentage)  # Yellow
        elif percentage >= 20:
            return ("Low", "#FF8800", percentage)  # Orange
        elif percentage > 0:
            return ("Critical", "#FF0000", percentage)  # Red
        else:
            return ("Dead", "#990000", percentage)  # Dark red

    @staticmethod
    def get_battery_icon(percentage: float) -> str:
        """
        Get a simple text icon for battery level

        Args:
            percentage: Battery percentage

        Returns:
            Text representation of battery level
        """
        if percentage >= 90:
            return "🔋 ████████"
        elif percentage >= 70:
            return "🔋 ███████"
        elif percentage >= 50:
            return "🔋 ██████"
        elif percentage >= 30:
            return "🔋 █████"
        elif percentage >= 10:
            return "⚠️  ████"
        else:
            return "🪫 ⚡"  # Critical


# Standard battery voltage ranges
BATTERY_CONFIGS = {
    "lipo_2s": {
        "name": "2S LiPo Battery",
        "min_mv": 6000,
        "max_mv": 8400,
        "nominal_mv": 7400,
    },
    "lipo_3s": {
        "name": "3S LiPo Battery",
        "min_mv": 9000,
        "max_mv": 12600,
        "nominal_mv": 11100,
    },
    "li_ion": {
        "name": "Li-Ion Battery (18650)",
        "min_mv": 2500,
        "max_mv": 4200,
        "nominal_mv": 3700,
    },
    "alkaline": {
        "name": "Alkaline Battery (AA)",
        "min_mv": 800,
        "max_mv": 1600,
        "nominal_mv": 1500,
    },
}


def create_battery_converter(battery_type: str = "lipo_2s") -> BatteryConverter:
    """
    Create a battery converter with predefined settings

    Args:
        battery_type: Battery type key from BATTERY_CONFIGS

    Returns:
        BatteryConverter instance
    """
    if battery_type not in BATTERY_CONFIGS:
        raise ValueError(f"Unknown battery type: {battery_type}")

    config = BATTERY_CONFIGS[battery_type]
    return BatteryConverter(min_mv=config["min_mv"], max_mv=config["max_mv"], battery_type=battery_type)
