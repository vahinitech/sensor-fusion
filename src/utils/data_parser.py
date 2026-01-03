"""Data parsing and validation for sensor streams"""


class SensorDataParser:
    """Parse and validate sensor data lines"""
    
    def __init__(self):
        self.parse_errors = 0
        self.valid_samples = 0
    
    def parse_line(self, line):
        """
        Parse CSV line into sensor values
        
        Returns:
            dict: Parsed sensor data or None if invalid
        """
        try:
            values = [float(x.strip()) for x in line.split(',')]
            
            if len(values) != 19:
                self.parse_errors += 1
                return None
            
            # Structure data
            data = {
                'timestamp': values[0],
                'top_imu': {
                    'accel': {'x': values[1], 'y': values[2], 'z': values[3]},
                    'gyro': {'x': values[4], 'y': values[5], 'z': values[6]}
                },
                'magnetometer': {
                    'x': values[7], 'y': values[8], 'z': values[9]
                },
                'rear_imu': {
                    'accel': {'x': values[10], 'y': values[11], 'z': values[12]},
                    'gyro': {'x': values[13], 'y': values[14], 'z': values[15]}
                },
                'force_sensor': {
                    'x': values[16], 'y': values[17], 'z': values[18]
                }
            }
            
            self.valid_samples += 1
            return data
            
        except (ValueError, IndexError):
            self.parse_errors += 1
            return None
    
    def get_stats(self):
        """Get parser statistics"""
        return {
            'valid': self.valid_samples,
            'errors': self.parse_errors,
            'total': self.valid_samples + self.parse_errors
        }
