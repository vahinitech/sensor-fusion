# Usage Guide

## Running the Application

### From Project Root

```bash
# Activate virtual environment
source .venv/bin/activate

# Run GUI app
python src/gui_app.py
```

### From src/ Directory

```bash
# Activate virtual environment (from src)
source ../.venv/bin/activate

# Run GUI app
python gui_app.py

# Run tests
python test_gui.py
```

## GUI Application

### Main Window

The application opens with a 1600x900 window containing:

1. **Control Panel** (Top)
   - Status indicator
   - Sample counter
   - Data rate display
   - Data source selection
   - Connection controls

2. **Visualization Area** (Main)
   - 4 rows of sensor plots
   - Real-time updating graphs
   - Statistics panel

### Control Panel

#### Data Source Selection

**Serial Mode:**
1. Select "serial" from dropdown
2. Choose COM port from list
3. Select baud rate (default: 115200)
4. Click "Connect"

**CSV Mode:**
1. Select "csv" from dropdown
2. Enter CSV file path (e.g., `../data/sample_sensor_data.csv`)
3. Click "Connect"

#### Buttons

- **Connect**: Establish data connection
- **Disconnect**: Close connection and clear buffers
- **↻ (Refresh)**: Refresh available COM ports

### Real-Time Display

#### Row 1: Top IMU (LSM6DSO)
- **Left**: Accelerometer X, Y, Z (m/s²)
- **Center**: Gyroscope X, Y, Z (°/s)
- **Right**: Acceleration magnitude (m/s²)

#### Row 2: Magnetometer (ST)
- Magnetic field X, Y, Z (μT)

#### Row 3: Rear IMU (LSM6DSM)
- **Left**: Accelerometer X, Y, Z (m/s²)
- **Center**: Gyroscope X, Y, Z (°/s)
- **Right**: Acceleration magnitude (m/s²)

#### Row 4: Force & Statistics
- **Left/Center**: Force Fx, Fy, Fz (ADC)
- **Right**: Statistics panel showing:
  - Sample count
  - Data rate (Hz)
  - Elapsed time (s)
  - Latest sensor readings
  - System status

## Example Workflows

### Testing with CSV Data

```bash
# Run with sample data
python src/gui_app.py

# In GUI:
1. Select "csv" source
2. Path will auto-fill: ../data/sample_sensor_data.csv
3. Click "Connect"
4. Watch real-time plots update
5. Monitor stats panel for data rate (~104 Hz)
```

### Live Serial Connection

```bash
# Run with live sensor
python src/gui_app.py

# In GUI:
1. Select "serial" source
2. Choose COM port (e.g., COM3)
3. Set baud rate (115200 recommended)
4. Click "Connect"
5. Monitor connection status
6. View real-time sensor data
```

### Running Tests

```bash
cd src
python test_gui.py

# Output:
# ✓ PASS - Module Imports
# ✓ PASS - SerialConfig
# ✓ PASS - SensorBuffers
# ✓ PASS - SensorParser
# ✓ PASS - CSV Reading
```

## Data Files

### Sample Data Location

```
data/
├── sample_sensor_data.csv    # 76 rows of test data
├── test_data.csv              # Additional test data
└── config.json                # Configuration
```

### CSV Format

Each row contains 19 comma-separated values:

```csv
timestamp,top_accel_x,top_accel_y,top_accel_z,top_gyro_x,top_gyro_y,top_gyro_z,mag_x,mag_y,mag_z,rear_accel_x,rear_accel_y,rear_accel_z,rear_gyro_x,rear_gyro_y,rear_gyro_z,force_x,force_y,force_z
0,100,200,300,10,20,30,1,2,3,400,500,600,40,50,60,100,200,300
...
```

## Status Indicators

### Connection Status

- **✓ Connected** (Green) - Active data connection
- **⊘ Disconnected** (Red) - No active connection

### Data Rate

- Should stabilize around **104 Hz** when connected
- CSV playback may vary based on file size

### Sample Counter

- Increments with each data point received
- Resets on disconnect/reconnect

## Troubleshooting

### "COM Port not found"
- Verify USB device is connected
- Try "↻" button to refresh ports
- Check Device Manager for correct port

### "Failed to open CSV file"
- Verify file path is correct
- Use absolute paths if relative paths don't work
- Check file exists: `ls -la ../data/sample_sensor_data.csv`

### "No data points showing"
- Verify data source is properly selected
- Check status shows "Connected"
- Wait for buffer to fill (depends on data rate)

### GUI freezes
- Disconnect current connection
- Kill the process and restart
- Check for corrupted data in input

## Performance

- **Sample Rate**: 104 Hz (all sensors)
- **Buffer Size**: 256 samples (~2.46 seconds)
- **GUI Refresh**: 100ms
- **Memory**: ~1-2 MB for full buffers

## Keyboard Shortcuts

- **Ctrl+C** - Stop application (in terminal)
- **Window Close (X)** - Gracefully shutdown

## Tips

1. **Start with CSV**: Test with sample data first
2. **Monitor Status**: Watch connection status and data rate
3. **Check Stats**: Review stats panel for latest values
4. **Adjust Window**: Resize window to see all plots clearly
5. **Reset on Error**: Disconnect and reconnect if data stops

## Getting Help

See documentation files:
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [SETUP.md](SETUP.md) - Installation guide
- [README.md](../README.md) - Project overview
