# Sensor Fusion Dashboard

Professional multi-sensor real-time visualization application for LSM6DSO, LSM6DSM, ST Magnetometer & HLP A04 Force Sensor at 104Hz.

## Quick Start

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run
python run.py
```

## Features

- **GUI Application**: Interactive tkinter interface with COM port & baudrate selection
- **Multi-Sensor Support**: Top IMU, Magnetometer, Rear IMU, Force Sensor
- **Dual Data Sources**: Live serial COM port or CSV file playback
- **Professional Visualization**: 4-row stacked matplotlib plots (1600x900)
- **104Hz Optimization**: 256-sample rolling buffer (~2.46 seconds)
- **Real-time Stats**: Sample count, data rate, sensor readings
- **Character Recognition**: AI-powered handwriting recognition using LSTM models
- **Action Detection**: Gesture and action recognition from sensor data

## Project Structure

```text
sensor_fusion/
├── run.py                     # Main application entry point
├── train_character_model.py   # Character recognition training
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── data/                      # Data files
│   ├── sample_sensor_data.csv
│   ├── test_data.csv
│   ├── config.json
│   └── onhw-chars_2021-06-30/ # Character recognition dataset
├── examples/                  # Example scripts
│   ├── QUICK_START.py
│   ├── train_with_onhw.py
│   └── evaluate_model.py
└── src/                       # Source code
    ├── gui/                   # GUI components
    ├── ai/                    # AI models (LSTM, character recognition)
    ├── core/                  # Core functionality
    ├── services/              # Service layer
    ├── actions/               # Action detection
    ├── interfaces/            # Interfaces
    └── utils/                 # Utility modules
```

## Usage

### 1. With CSV Test Data

```bash
python run.py
# Select "csv" source → Connect
# Watch real-time plots with sample data
```

### 2. With Live Serial Connection

```bash
python run.py
# Select "serial" source → Choose COM port → Connect
# Monitor live sensor data
```

### 3. Train Character Recognition Model

```bash
python train_character_model.py
```

### 4. Run Examples

```bash
# Quick start guide
python examples/QUICK_START.py

# Train with ONHW dataset
python examples/train_with_onhw.py

# Evaluate trained model
python examples/evaluate_model.py
```

## System Requirements

| Component | Requirement |
| --------- | ---------- |
| Python | 3.8+ |
| OS | macOS, Linux, Windows |
| RAM | 2 GB minimum (4 GB for AI training) |
| Display | 1600x900+ recommended |

## Dependencies

Core dependencies:

- `pyserial>=3.5` - Serial communication
- `matplotlib>=3.8.0` - Real-time plotting
- `numpy>=1.26.0` - Numerical operations
- `tensorflow>=2.10.0` - Machine learning
- `scikit-learn>=1.0.0` - ML utilities
- `tkinter` - GUI (built-in)

## Sensor Specifications

| Sensor | Model | Channels | Sample Rate |
| ------ | ----- | -------- | ----------- |
| Top IMU | LSM6DSO | Accel (3) + Gyro (3) | 104 Hz |
| Magnetometer | ST | Magnetic field (3) | 104 Hz |
| Rear IMU | LSM6DSM | Accel (3) + Gyro (3) | 104 Hz |
| Force Sensor | HLP A04 | Force (3) | 104 Hz |

## Data Format

CSV files: 19 comma-separated fields per line

```csv
timestamp, accel_x1, accel_y1, accel_z1, gyro_x1, gyro_y1, gyro_z1,
mag_x, mag_y, mag_z, accel_x2, accel_y2, accel_z2, gyro_x2, gyro_y2, gyro_z2,
force_x, force_y, force_z
```

**Example:**

```csv
5430303,2651,50,-3099,-10,-13,-2,5033,226,-6074,7,4,-7,-160,89,399,3209,1660,2821
5430305,2652,45,-3098,-9,-12,-3,5034,228,-6072,8,5,-8,-158,91,401,3210,1662,2820
```

## Dashboard Layout

### Row 1: Top IMU (LSM6DSO)

- Accelerometer (X/Y/Z) - Range: ±8000 m/s²
- Gyroscope (X/Y/Z) - Range: ±2000 °/s
- Acceleration Magnitude

### Row 2: Magnetometer (Full Width)

- Magnetic Field X/Y/Z - Range: ±10000 μT

### Row 3: Rear IMU (LSM6DSM)

- Accelerometer (X/Y/Z) - Range: ±8000 m/s²
- Gyroscope (X/Y/Z) - Range: ±2000 °/s
- Acceleration Magnitude

### Row 4: Force Sensor & Status

- 3-Axis Force (HLP A04) - Range: 0-4096 ADC
- System Status (samples, data rate, latest values)

## Performance

- **Sample Rate**: 104 Hz across all sensors
- **Buffer**: 256 samples (~2.46 seconds)
- **Update Rate**: 100ms GUI refresh
- **Memory**: ~50KB per sensor stream

## Troubleshooting

### GUI Won't Start

```bash
brew install python-tk@3.13  # macOS only
```

### Module Not Found

```bash
pip install --upgrade -r requirements.txt
```

### Serial Port Issues

- Verify device is connected
- Check port in Device Manager (Windows) or `ls /dev/tty*` (Linux/Mac)
- Try refresh button in GUI

### CSV File Not Found

- Ensure CSV is in the correct directory
- Check file path in config or command arguments


## License

See LICENSE file for details.

## Support

For issues or questions, check the examples in the `examples/` folder or review the source code documentation.

