"""Configuration management for sensor dashboard"""
import json


class Config:
    """Load and manage dashboard configuration"""
    
    def __init__(self, config_path='config.json'):
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path):
        """Load JSON configuration file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Config file not found: {config_path}, using defaults")
            return self._default_config()
        except json.JSONDecodeError:
            print(f"⚠️  Invalid JSON in {config_path}, using defaults")
            return self._default_config()
    
    def _default_config(self):
        """Return default configuration"""
        return {
            'dashboard': {
                'title': 'Multi-Sensor Real-Time Dashboard (104Hz)',
                'target_frequency': 104,
                'buffer_size': 256
            },
            'data_source': {
                'serial': {
                    'port': 'COM3',
                    'baudrate': 115200,
                    'timeout': 1
                },
                'csv': {
                    'path': 'sample_sensor_data.csv',
                    'has_header': False,
                    'delimiter': ','
                }
            },
            'ui': {
                'update_interval_ms': 100,
                'figure_size': [18, 14],
                'show_grid': True,
                'grid_alpha': 0.3,
                'line_width': 1.5
            }
        }
    
    def get(self, *keys, default=None):
        """Get nested config value"""
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value
