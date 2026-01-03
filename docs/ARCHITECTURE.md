# Sensor Fusion Dashboard - Architecture

## Project Structure

```
sensor_fusion/
├── src/                          # Source code
│   ├── gui_app.py               # Main GUI application (tkinter)
│   ├── test_gui.py              # GUI app validation tests
│   ├── serial_config.py         # Serial port utilities
│   ├── sensor_buffers.py        # Circular buffer management
│   ├── sensor_parser.py         # CSV/Serial data parsing
│   ├── data_reader.py           # Serial & CSV data readers
│   ├── dashboard.py             # Dashboard logic
│   ├── plotter.py               # Plotting utilities
│   ├── config.py                # Configuration management
│   └── __init__.py
├── data/                        # Data files
│   ├── sample_sensor_data.csv   # Test data
│   ├── test_data.csv            # Additional test data
│   └── config.json              # Configuration
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md          # This file
│   ├── SETUP.md                 # Installation & setup guide
│   ├── USAGE.md                 # How to use the application
│   └── CLEANUP_SUMMARY.md       # Project cleanup notes
├── README.md                    # Project overview
├── requirements.txt             # Python dependencies
└── .gitignore                  # Git ignore rules
```

## Core Components

### 1. GUI Application (`src/gui_app.py`)
- **Framework**: tkinter
- **Resolution**: 1600x900
- **Features**:
  - Multi-sensor real-time visualization
  - COM port and CSV file data source support
  - 104Hz optimization with 256-sample rolling buffer
  - Live statistics panel
  - Professional matplotlib integration

### 2. Data Processing (`src/`)
- **sensor_buffers.py**: Circular deque buffers for 9 sensor streams
- **sensor_parser.py**: Parses 19-field CSV/serial format
- **data_reader.py**: Abstracts serial and CSV reading
- **serial_config.py**: Port discovery and connection validation

### 3. Sensor Layout

The dashboard displays data in a 4-row, 3-column grid:

**Row 1: Top IMU (LSM6DSO)**
- Accelerometer (X, Y, Z)
- Gyroscope (X, Y, Z)
- Acceleration magnitude

**Row 2: Magnetometer (ST)**
- Magnetic field (X, Y, Z)

**Row 3: Rear IMU (LSM6DSM)**
- Accelerometer (X, Y, Z)
- Gyroscope (X, Y, Z)
- Acceleration magnitude

**Row 4: Force Sensor & Stats**
- Force (Fx, Fy, Fz)
- System statistics

## Data Format

CSV format with 19 columns:

```
TIMESTAMP, TOP_ACCEL_X, TOP_ACCEL_Y, TOP_ACCEL_Z,
TOP_GYRO_X, TOP_GYRO_Y, TOP_GYRO_Z,
MAG_X, MAG_Y, MAG_Z,
REAR_ACCEL_X, REAR_ACCEL_Y, REAR_ACCEL_Z,
REAR_GYRO_X, REAR_GYRO_Y, REAR_GYRO_Z,
FORCE_X, FORCE_Y, FORCE_Z
```

## Sensor Specifications

| Sensor | Model | Range | Resolution | Sample Rate |
|--------|-------|-------|------------|------------|
| Top IMU | LSM6DSO | ±16g, ±2000°/s | 16-bit | 104 Hz |
| Magnetometer | ST | ±50mT | 16-bit | 104 Hz |
| Rear IMU | LSM6DSM | ±16g, ±2000°/s | 16-bit | 104 Hz |
| Force Sensor | HLP A04 | 0-4096 ADC | 12-bit | 104 Hz |

## Performance

- **Sample Rate**: 104 Hz across all sensors
- **Buffer Size**: 256 samples (~2.46 seconds)
- **Update Rate**: 100ms GUI refresh
- **Memory**: ~50KB per 256-sample buffer (all sensors combined)

## Dependencies

- `pyserial>=3.5` - Serial port communication
- `matplotlib>=3.8.0` - Real-time plotting
- `numpy>=1.26.0` - Numerical operations
- `tkinter` - GUI framework (included with Python)

## Testing

Run validation tests:
```bash
cd src
python test_gui.py
```

Tests cover:
- Module imports
- Serial configuration
- Sensor buffers
- Data parsing
- CSV file reading
