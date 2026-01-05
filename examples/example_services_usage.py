#!/usr/bin/env python3
"""
Example: Using the new service-based architecture
This demonstrates how the services can be used independently of any UI
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from services import SensorService, BatteryService, ActionService, AIService
import time


def example_headless_logging():
    """Example: Log sensor data to console without GUI"""
    print("=" * 60)
    print("EXAMPLE 1: Headless Data Logging")
    print("=" * 60)

    # Create services
    sensor_service = SensorService(buffer_size=256)
    battery_service = BatteryService()
    action_service = ActionService()

    # Register callbacks
    def on_data(data):
        """Print data when received"""
        # Update battery
        battery = battery_service.update(data.get("battery_mv", 0))

        # Update action
        action_service.update(data)
        action = action_service.get_action()

        # Print every 50 samples
        if sensor_service.sample_count % 50 == 0:
            print(
                f"Sample {sensor_service.sample_count}: "
                f"Battery: {battery['percentage']:.0f}% | "
                f"Action: {action} | "
                f"Force: {data.get('force_raw', 0):.0f}"
            )

    sensor_service.register_data_callback(on_data)

    # Connect (would connect to real serial port)
    print("\nTo use: sensor_service.connect_serial('COM3', 115200)")
    print("Data will be logged to console automatically")


def example_rest_api_ready():
    """Example: Structure for REST API"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: REST API Ready Structure")
    print("=" * 60)

    # All services are independent and can be used in Flask/FastAPI
    sensor_service = SensorService()
    battery_service = BatteryService()

    print(
        """
Flask API Example:

    @app.route('/api/sensor/data')
    def get_sensor_data():
        data = sensor_service.get_buffer_data()
        return jsonify(data)
    
    @app.route('/api/battery/status')
    def get_battery():
        status = battery_service.get_status()
        return jsonify(status)
    
    @app.route('/api/connect', methods=['POST'])
    def connect():
        port = request.json['port']
        baudrate = request.json['baudrate']
        success = sensor_service.connect_serial(port, baudrate)
        return jsonify({'connected': success})
    """
    )


def example_websocket_streaming():
    """Example: WebSocket data streaming"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: WebSocket Streaming Ready")
    print("=" * 60)

    sensor_service = SensorService()

    print(
        """
Socket.IO Example:

    def on_data(data):
        # Stream to all connected clients
        socketio.emit('sensor_data', {
            'timestamp': data['timestamp'],
            'force_raw': data['force_raw'],
            'battery_mv': data['battery_mv'],
            # ... all sensor data
        })
    
    sensor_service.register_data_callback(on_data)
    sensor_service.connect_serial('COM3', 115200)
    
    # Real-time updates to browser!
    """
    )


def example_data_export():
    """Example: Export data to file"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Data Export/Analysis")
    print("=" * 60)

    sensor_service = SensorService()

    print(
        """
Export to CSV:

    import csv
    
    csv_file = open('sensor_log.csv', 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['timestamp', 'force_raw', 'battery_mv', ...])
    
    def on_data(data):
        writer.writerow([
            data['timestamp'],
            data['force_raw'],
            data['battery_mv'],
            data['force_mv'],
            # ... all fields
        ])
    
    sensor_service.register_data_callback(on_data)
    sensor_service.connect_serial('COM3', 115200)
    """
    )


def example_multi_ui():
    """Example: Run multiple UIs simultaneously"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Multiple UIs Simultaneously")
    print("=" * 60)

    print(
        """
You can run Tkinter GUI + Flask Web + Data Logger all at once:

    # Start services once
    sensor_service = SensorService()
    battery_service = BatteryService()
    
    # Tkinter GUI callback
    def update_tkinter(data):
        gui.update_plots(data)
    
    # Flask WebSocket callback
    def update_web(data):
        socketio.emit('data', data)
    
    # File logger callback
    def log_to_file(data):
        logger.write(data)
    
    # Register all callbacks
    sensor_service.register_data_callback(update_tkinter)
    sensor_service.register_data_callback(update_web)
    sensor_service.register_data_callback(log_to_file)
    
    # One connection, multiple outputs!
    sensor_service.connect_serial('COM3', 115200)
    """
    )


if __name__ == "__main__":
    print("\n🚀 NEW MODULAR ARCHITECTURE - USAGE EXAMPLES\n")

    example_headless_logging()
    example_rest_api_ready()
    example_websocket_streaming()
    example_data_export()
    example_multi_ui()

    print("\n" + "=" * 60)
    print("✅ Services are framework-independent!")
    print("✅ Easy to port to Flask, Qt, or any framework")
    print("✅ Can run headless, API-only, or with GUI")
    print("✅ Multiple UIs can share the same services")
    print("=" * 60)
    print("\nSee REFACTORED_ARCHITECTURE.md for full documentation")
    print()
