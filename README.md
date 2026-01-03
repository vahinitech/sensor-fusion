# Sensor Fusion Dashboard

Professional multi-sensor real-time visualization application for LSM6DSO, LSM6DSM, ST Magnetometer & HLP A04 Force Sensor at 104Hz.

## Quick Start

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run
python src/gui_app.py
```

## Features

- **GUI Application**: Interactive tkinter interface with COM port & baudrate selection
- **Multi-Sensor Support**: Top IMU, Magnetometer, Rear IMU, Force Sensor
- **Dual Data Sources**: Live serial COM port or CSV file playback
- **Professional Visualization**: 4-row stacked matplotlib plots (1600x900)
- **104Hz Optimization**: 256-sample rolling buffer (~2.46 seconds)
- **Real-time Stats**: Sample count, data rate, sensor readings

## Project Structure

```
sensor_fusion/
├── src/                    # Python source code
│   ├── gui_app.py         # Main GUI application
│   ├── test_gui.py        # Validation tests
│   └── [sensor modules]   # Core functionality
├── data/                  # Data files
│   ├── sample_sensor_data.csv
│   ├── test_data.csv
│   └── config.json
├── docs/                  # Documentation
│   ├── SETUP.md          # Installation guide
│   ├── USAGE.md          # How to use
│   ├── ARCHITECTURE.md   # System design
│   └── CLEANUP_SUMMARY.md
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Documentation

- **[SETUP.md](docs/SETUP.md)** - Installation & troubleshooting
- **[USAGE.md](docs/USAGE.md)** - How to run and use the application
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design & components

## Usage

### 1. With CSV Test Data

```bash
python src/gui_app.py
# Select "csv" source → Connect
# Watch real-time plots with sample data
```

### 2. With Live Serial Connection

```bash
python src/gui_app.py
# Select "serial" source → Choose COM port → Connect
# Monitor live sensor data
```

### 3. Run Tests

```bash
cd src
python test_gui.py
```

## System Requirements

| Component | Requirement |
|-----------|------------|
| Python | 3.8+ |
| OS | macOS, Linux, Windows |
| RAM | 512 MB minimum |
| Display | 1600x900+ recommended |

## Dependencies

- `pyserial>=3.5` - Serial communication
- `matplotlib>=3.8.0` - Real-time plotting
- `numpy>=1.26.0` - Numerical operations
- `tkinter` - GUI (built-in)

## Sensor Specifications

| Sensor | Model | Channels | Sample Rate |
|--------|-------|----------|------------|
| Top IMU | LSM6DSO | Accel (3) + Gyro (3) | 104 Hz |
| Magnetometer | ST | Magnetic field (3) | 104 Hz |
| Rear IMU | LSM6DSM | Accel (3) + Gyro (3) | 104 Hz |
| Force Sensor | HLP A04 | Force (3) | 104 Hz |

## Performance

- **Sample Rate**: 104 Hz across all sensors
- **Buffer**: 256 samples (~2.46 seconds)
- **Update Rate**: 100ms GUI refresh
- **Memory**: ~50KB per sensor stream

## Troubleshooting

**GUI Won't Start**
```bash
brew install python-tk@3.13  # macOS only
```

**Module Not Found**
```bash
pip install --upgrade -r requirements.txt
```

**Serial Port Issues**
- Verify device is connected
- Check port in Device Manager
- Try refresh button in GUI

See [SETUP.md](docs/SETUP.md) for more help.

## Testing

```bash
cd src
python test_gui.py

# Expected output:
# ✓ PASS - Module Imports
# ✓ PASS - SerialConfig
# ✓ PASS - SensorBuffers
# ✓ PASS - SensorParser
# ✓ PASS - CSV Reading
```

## File Organization

### Data Files (`data/`)
- Sample CSV files for testing
- Configuration settings

### Documentation (`docs/`)
- Setup & installation guides
- Usage instructions
- Architecture documentation

### Source Code (`src/`)
- GUI application
- Data processing modules
- Serial communication utilities
- Test suite

## Next Steps

1. Read [SETUP.md](docs/SETUP.md) for installation
2. Check [USAGE.md](docs/USAGE.md) for running the app
3. Review [ARCHITECTURE.md](docs/ARCHITECTURE.md) for system design
4. Run tests: `cd src && python test_gui.py`

## License

See LICENSE file for details.

## Support

For issues or questions, refer to documentation in `/docs` folder.
4. **View Data**: Watch real-time plots update

## Project Structure

```
sensor_fusion/
├── gui_app.py                 # Main GUI application
├── config.json                # Configuration
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── sample_sensor_data.csv     # Test data
│
├── src/                       # Source modules
│   ├── serial_config.py       # COM port detection
│   ├── sensor_buffers.py      # Data buffers
│   ├── sensor_parser.py       # Data parsing
│   └── data_reader.py         # Data reading
│
└── docs/                      # Documentation
```

## Data Format

CSV files: 19 comma-separated fields
```
timestamp, accel_x1, accel_y1, accel_z1, gyro_x1, gyro_y1, gyro_z1,
mag_x, mag_y, mag_z, accel_x2, accel_y2, accel_z2, gyro_x2, gyro_y2, gyro_z2,
force_x, force_y, force_z
```

## Visualization Layout

| Row | Sensor | Layout |
|-----|--------|--------|
| 1 | Top IMU (LSM6DSO) | Accel \| Gyro \| Magnitude |
| 2 | Magnetometer (ST) | Magnetic Field (full width) |
| 3 | Rear IMU (LSM6DSO) | Accel \| Gyro \| Magnitude |
| 4 | Force Sensor + Stats | Force (2 cols) \| Status (1 col) |

## Requirements

- Python 3.8+
- pyserial 3.5+
- matplotlib 3.8+
- numpy 1.26+

## Troubleshooting

**COM Port not detected**: Click ↻ button or reconnect cable  
**CSV not loading**: Verify file path and format (19 columns)  
**Plots not updating**: Check connection status, verify data arriving  

## Version

1.0.0 - January 2026
- **LSM6DSO** - Top IMU (Accelerometer + Gyroscope)
- **LSM6DSM** - Rear IMU (Accelerometer + Gyroscope)
- **ST Magnetometer** - Magnetic field sensing
- **HLP A04** - 3-Axis Force sensor

## 📊 Features

✅ **104Hz Data Acquisition** - Optimized for real-time performance  
✅ **256-Sample Buffer** - ~2.46 seconds rolling window for smooth visualization  
✅ **4-Row Stacked Layout** - Professional dashboard with 9 simultaneous plots  
✅ **Dual Input Modes** - Serial COM port or CSV file playback  
✅ **Color-Coded Axes** - Red=X, Green=Y, Blue=Z for quick identification  
✅ **Auto-Scaling** - Dynamic range adjustment based on sensor values  
✅ **Real-Time Statistics** - Live data rate monitoring and system status  
✅ **JSON Configuration** - Fully customizable sensor configuration  
✅ **Multi-Threaded** - Non-blocking serial/CSV reading  
✅ **Memory Efficient** - Circular deque buffers for optimal performance  

## 🎯 Dashboard Layout

### Row 1: Top IMU (LSM6DSO)
```
┌─────────────────────────────┬─────────────────────────────┬─────────────────────────────┐
│ Accelerometer (X/Y/Z)       │ Gyroscope (X/Y/Z)           │ Accel Magnitude             │
│ Range: ±8000 m/s²           │ Range: ±2000 °/s            │ Derived value               │
└─────────────────────────────┴─────────────────────────────┴─────────────────────────────┘
```

### Row 2: Magnetometer (Full Width)
```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ Magnetic Field X/Y/Z (ST Sensor) - Range: ±10000 μT                                      │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

### Row 3: Rear IMU (LSM6DSM)
```
┌─────────────────────────────┬─────────────────────────────┬─────────────────────────────┐
│ Accelerometer (X/Y/Z)       │ Gyroscope (X/Y/Z)           │ Accel Magnitude             │
│ Range: ±8000 m/s²           │ Range: ±2000 °/s            │ Derived value               │
└─────────────────────────────┴─────────────────────────────┴─────────────────────────────┘
```

### Row 4: Force Sensor & Status
```
┌─────────────────────────────────────────────┬─────────────────────────────┐
│ 3-Axis Force (HLP A04) - Range: 0-4096 ADC  │ System Status               │
│ • Real-time force visualization             │ • Samples count             │
│ • X/Y/Z axis tracking                       │ • Elapsed time              │
│                                              │ • Data rate (Hz)            │
│                                              │ • Latest sensor values      │
└─────────────────────────────────────────────┴─────────────────────────────┘
```

## 📋 Data Format (CSV)

19 comma-separated fields per line:

```
timestamp, accel_x1, accel_y1, accel_z1, gyro_x1, gyro_y1, gyro_z1, 
mag_x, mag_y, mag_z, accel_x2, accel_y2, accel_z2, gyro_x2, gyro_y2, gyro_z2, 
force_x, force_y, force_z
```

**Example:**
```
5430303,2651,50,-3099,-10,-13,-2,5033,226,-6074,7,4,-7,-160,89,399,3209,1660,2821
5430305,2652,45,-3098,-9,-12,-3,5034,228,-6072,8,5,-8,-158,91,401,3210,1662,2820
```

## 🚀 Quick Start

### Installation

```bash
# Clone or navigate to repository
cd sensor_fusion

# Install dependencies
pip install -r requirements.txt
```

### CSV Mode (Testing)

```bash
# Use sample data
python sensor_dashboard.py --source csv

# Or use generated test data
python sensor_dashboard.py --source csv --csv test_data.csv

# Or specify custom CSV
python sensor_dashboard.py --source csv --csv your_data.csv
```

### Serial Mode (Live Hardware)

```bash
# Connect to COM port
python sensor_dashboard.py --source serial --port COM3

# Linux/Mac format
python sensor_dashboard.py --source serial --port /dev/ttyUSB0
```

### Custom Configuration

```bash
python sensor_dashboard.py --config custom_config.json
```

## 📁 Project Files

```
sensor_fusion/
├── sensor_dashboard.py      # Main application
├── test_dashboard.py        # Test & verification script
├── config.json              # Default configuration
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore file
├── sample_sensor_data.csv  # Sample data (76 samples)
├── test_data.csv           # Generated test data (500 samples)
└── README.md               # This file
```

## ⚙️ Configuration

Edit `config.json` to customize:

```json
{
  "dashboard": {
    "title": "Multi-Sensor Real-Time Dashboard (104Hz)",
    "target_frequency": 104,
    "buffer_size": 256
  },
  "data_source": {
    "serial": {
      "port": "COM3",
      "baudrate": 115200
    },
    "csv": {
      "path": "sample_sensor_data.csv",
      "has_header": false
    }
  },
  "ui": {
    "update_interval_ms": 100,
    "animation_fps": 10,
    "figure_size": [18, 14],
    "show_grid": true
  }
}
```

## 🧪 Testing

Run the complete test suite:

```bash
python test_dashboard.py
```

This will:
- ✓ Verify all dependencies
- ✓ Generate 500 samples of synthetic data
- ✓ Validate CSV format
- ✓ Display configuration info
- ✓ List available data files

## 📊 Performance

- **Data Rate**: 104 Hz (9.6 ms per sample)
- **Buffer Size**: 256 samples = ~2.46 seconds
- **Update Rate**: 10 FPS (100 ms interval)
- **Memory Usage**: ~2.5 MB for 256-sample buffers
- **Real-Time**: Zero latency for smooth visualization

## 🔧 Usage Examples

### Example 1: Test with sample data
```bash
python sensor_dashboard.py --source csv --csv sample_sensor_data.csv
```

### Example 2: Live serial monitoring
```bash
python sensor_dashboard.py --source serial --port COM3
```

### Example 3: Custom CSV playback
```bash
python sensor_dashboard.py --source csv --csv /path/to/experiment_data.csv
```

### Example 4: Alternative config
```bash
python sensor_dashboard.py --source csv --config my_config.json
```

## 📝 Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--source` | str | `csv` | Data source: `serial` or `csv` |
| `--port` | str | `COM3` | Serial port name |
| `--csv` | str | `sample_sensor_data.csv` | CSV file path |
| `--config` | str | `config.json` | Configuration JSON file |

## 🎨 Visualization Features

### Color Scheme
- **Red Line**: X-axis data
- **Green Line**: Y-axis data
- **Blue Line**: Z-axis data
- **Purple Line**: Magnitude calculations

### Interactive Dashboard
- Real-time scrolling buffer
- Live statistics panel
- Adaptive axis scaling
- Grid overlay for reference
- Professional layout with proper spacing

### Status Indicators
- Connection status (✓ RUNNING / ⊘ IDLE)
- Sample counter
- Elapsed time
- Data rate monitoring
- Latest sensor values

## 🛠️ Architecture

### Threading Model
- **Main Thread**: Matplotlib GUI and animation
- **Reader Thread**: Serial/CSV reading (non-blocking)
- **Daemon Thread**: Ensures clean shutdown

### Memory Management
- **Circular Buffers**: Fixed-size deques for efficiency
- **No Memory Leaks**: Automatic buffer overflow handling
- **Scalable**: Easy to adjust buffer size

### Data Pipeline
```
Serial/CSV → Parser → Buffers → Animation → Display
```

## 📦 Dependencies

- **matplotlib** (3.8.2) - Plotting and visualization
- **numpy** (1.24.3) - Numerical computations
- **pyserial** (3.5) - Serial port communication

## 🐛 Troubleshooting

### "Port not found" error
```bash
# Check available ports (Windows)
python -m serial.tools.list_ports

# Or on Linux/Mac
ls /dev/tty*
```

### CSV file not found
- Ensure CSV is in the same directory as the script
- Use full path: `--csv /full/path/to/file.csv`

### Slow visualization
- Reduce `update_interval_ms` in config.json
- Decrease `figure_size` for smaller display
- On slow systems, use larger buffer intervals

### Matplotlib backend issues
```bash
# Try different backend
python -c "import matplotlib; matplotlib.use('Agg')"
```

## 📈 Performance Optimization

For optimal performance:

1. **CSV Mode**: Fastest for testing (simulated 104Hz)
2. **Larger Buffer**: Smoother visualization but more memory
3. **Update Rate**: 10 FPS is a good balance
4. **Grid Display**: Disable for faster rendering if needed

## 🔐 Data Format Validation

The dashboard validates:
- ✓ CSV line count (must have 19 fields)
- ✓ Numeric data types
- ✓ Buffer overflow handling
- ✓ Serial port connectivity
- ✓ File I/O errors

## 📞 Support

For issues or questions:
1. Check `config.json` settings
2. Run `test_dashboard.py` to verify setup
3. Review data format requirements
4. Check log output for error messages

## 📄 License

This project is provided as-is for educational and research purposes.

## 🎯 Next Steps

1. **Calibrate Sensors**: Adjust config.json ranges for your hardware
2. **Collect Data**: Use serial mode with your actual devices
3. **Analyze Results**: Export data for further analysis
4. **Customize Layout**: Modify matplotlib subplot configuration

---

**Happy Sensor Fusion! 🚀**
