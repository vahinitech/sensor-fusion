"""
Serial Configuration Manager
Handles COM port detection and configuration
"""

import serial
import platform
from typing import List, Tuple

class SerialConfig:
    """Manages serial port configuration"""
    
    @staticmethod
    def get_available_ports() -> List[str]:
        """
        Get list of available COM ports
        
        Returns:
            List of available COM port names
        """
        import serial.tools.list_ports
        
        ports = []
        for port_info in serial.tools.list_ports.comports():
            ports.append(port_info.device)
        
        return sorted(ports) if ports else []
    
    @staticmethod
    def get_standard_baudrates() -> List[int]:
        """Get standard baud rates"""
        return [
            9600,
            14400,
            19200,
            38400,
            57600,
            115200,
            230400,
            460800,
            921600
        ]
    
    @staticmethod
    def validate_connection(port: str, baudrate: int, timeout: float = 1.0) -> Tuple[bool, str]:
        """
        Validate serial connection
        
        Args:
            port: COM port name
            baudrate: Baud rate
            timeout: Connection timeout in seconds
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            ser = serial.Serial(port, baudrate, timeout=timeout)
            ser.close()
            return True, f"✓ Connected to {port} @ {baudrate} baud"
        except Exception as e:
            return False, f"✗ Failed to connect: {str(e)}"
