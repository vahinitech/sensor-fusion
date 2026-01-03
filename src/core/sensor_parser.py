"""
Sensor Data Parser
Parses incoming CSV/Serial data into sensor readings
"""

from typing import Optional, Dict

class SensorParser:
    """Parses sensor data strings into structured data"""
    
    EXPECTED_FIELDS = 19  # timestamp + 18 sensor values
    
    @staticmethod
    def parse_line(line: str) -> Optional[Dict]:
        """
        Parse a data line from CSV/Serial
        
        Args:
            line: Comma-separated data string
            
        Returns:
            Dictionary with parsed sensor data, or None if parse failed
        """
        try:
            values = [float(x.strip()) for x in line.split(',')]
            
            if len(values) != SensorParser.EXPECTED_FIELDS:
                return None
            
            return {
                'timestamp': int(values[0]),
                # Top IMU (LSM6DSO)
                'top_accel_x': values[1],
                'top_accel_y': values[2],
                'top_accel_z': values[3],
                'top_gyro_x': values[4],
                'top_gyro_y': values[5],
                'top_gyro_z': values[6],
                # Magnetometer
                'mag_x': values[7],
                'mag_y': values[8],
                'mag_z': values[9],
                # Rear IMU (LSM6DSM)
                'rear_accel_x': values[10],
                'rear_accel_y': values[11],
                'rear_accel_z': values[12],
                'rear_gyro_x': values[13],
                'rear_gyro_y': values[14],
                'rear_gyro_z': values[15],
                # Force Sensor (HLP A04)
                'force_x': values[16],
                'force_y': values[17],
                'force_z': values[18],
            }
        except (ValueError, IndexError):
            return None
