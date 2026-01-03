"""Data reading from serial ports or CSV files"""
import serial
import csv
import os
import time
import threading


class DataReader:
    """Base class for data readers"""
    
    def __init__(self, data_callback):
        self.data_callback = data_callback
        self.running = False
        self.thread = None
    
    def start(self):
        """Start reading in background thread"""
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop reading"""
        self.running = False
    
    def _read_loop(self):
        """Override in subclass"""
        raise NotImplementedError


class SerialReader(DataReader):
    """Read sensor data from serial port"""
    
    def __init__(self, port, baudrate, data_callback):
        super().__init__(data_callback)
        self.port = port
        self.baudrate = baudrate
        self.serial_port = None
    
    def connect(self):
        """Establish serial connection"""
        try:
            self.serial_port = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            print(f"✓ Connected to {self.port} @ {self.baudrate} baud")
            return True
        except serial.SerialException as e:
            print(f"✗ Failed to connect to {self.port}: {e}")
            return False
    
    def _read_loop(self):
        """Read from serial port continuously"""
        while self.running and self.serial_port:
            try:
                if self.serial_port.in_waiting:
                    line = self.serial_port.readline().decode('utf-8').strip()
                    if line and not line.startswith('#'):
                        self.data_callback(line)
                else:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
            except Exception as e:
                if self.running:
                    print(f"Serial read error: {e}")
                    time.sleep(0.1)  # Sleep on error to avoid tight error loop
    
    def close(self):
        """Close serial port"""
        if self.serial_port:
            self.serial_port.close()
            print("✓ Serial port closed")


class CSVReader(DataReader):
    """Read sensor data from CSV file"""
    
    def __init__(self, csv_path, data_callback, has_header=False):
        super().__init__(data_callback)
        self.csv_path = csv_path
        self.has_header = has_header
        self.csv_file = None
        self.csv_reader = None
        self.line_count = 0
    
    def connect(self):
        """Open CSV file"""
        try:
            if not os.path.exists(self.csv_path):
                print(f"✗ CSV file not found: {self.csv_path}")
                return False
            
            self.csv_file = open(self.csv_path, 'r')
            self.csv_reader = csv.reader(self.csv_file)
            
            if self.has_header:
                next(self.csv_reader)
            
            print(f"✓ Opened CSV file: {self.csv_path}")
            return True
        except Exception as e:
            print(f"✗ Failed to open CSV file: {e}")
            return False
    
    def _read_loop(self):
        """Read from CSV file with 104Hz simulation"""
        try:
            for row in self.csv_reader:
                if not self.running:
                    break
                
                if row and len(row) >= 18:
                    line = ','.join(row)
                    self.data_callback(line)
                    self.line_count += 1
                    time.sleep(0.0096)  # ~104Hz
        except StopIteration:
            self.running = False
            print("\n✓ CSV file playback completed")
        except Exception as e:
            if self.running:
                print(f"CSV read error: {e}")
    
    def close(self):
        """Close CSV file"""
        if self.csv_file:
            self.csv_file.close()
            print("✓ CSV file closed")
