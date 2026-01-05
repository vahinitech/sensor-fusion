# Sensor Fusion Dashboard

[![Code Quality](https://img.shields.io/badge/pylint-9.81%2F10-brightgreen)](https://github.com/vahinitech/sensor-fusion-dashboard)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-30%20passing-success)](https://github.com/vahinitech/sensor-fusion-dashboard)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

**Real-time multi-sensor data acquisition, visualization, and AI-powered handwriting recognition system**

A professional-grade sensor fusion dashboard that combines real-time data visualization with LSTM-based character recognition. Built for research, development, and production deployment of IMU-based gesture recognition systems.

## 🎯 Overview

The Sensor Fusion Dashboard is a comprehensive platform for collecting, processing, and analyzing multi-sensor data streams from IMU devices. It features a modern GUI interface, real-time plotting, sensor fusion algorithms (Kalman/EKF/Complementary filters), and deep learning-based character recognition capabilities.

**Key Capabilities:**
- **Real-time Visualization**: Professional matplotlib-based dashboard displaying 18+ sensor channels simultaneously at 208Hz
- **AI Character Recognition**: LSTM neural network trained on OnHW dataset for handwriting recognition from sensor movements
- **Action Detection**: Intelligent gesture and pen state detection (idle, hovering, writing, pen_down, pen_up)
- **Sensor Fusion**: Kalman Filter, Extended Kalman Filter, and Complementary Filter implementations for noise reduction
- **Battery Monitoring**: Real-time voltage-to-percentage conversion with health status indicators
- **Production Ready**: Complete CI/CD pipeline with automated testing, code quality gates (Pylint 9.5+), and Black formatting

**Hardware Support:**
- Top IMU (LSM6DSO): 6-axis accelerometer + gyroscope
- Rear IMU (LSM6DSM): 6-axis accelerometer + gyroscope  
- Magnetometer (ST): 3-axis magnetic field sensor
- Force Sensor (HLP A04): Analog force measurement
- Battery Monitor: 2S LiPo voltage tracking (6.0V - 8.4V)

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/vahinitech/sensor-fusion-dashboard.git
cd sensor-fusion-dashboard

# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch dashboard
python run.py
```

**First-time setup (~2 minutes):**
1. Select COM port from dropdown (or use CSV mode for testing)
2. Choose baudrate (default: 115200)
3. Click "Connect" to start real-time visualization
4. View 4-row dashboard with all sensor streams

## ✨ Features

### Core Capabilities
- **📊 Real-time Visualization**: 4-row professional dashboard with 12+ subplots updating at 100ms intervals
- **🤖 AI Character Recognition**: LSTM-based handwriting recognition with 80%+ confidence threshold
- **🎮 Action Detection**: 5-state pen action classification (idle, hovering, writing, pen_down, pen_up)
- **🔌 Dual Data Sources**: Live serial (COM/UART) or CSV file replay for testing
- **⚡ High Performance**: 208Hz sampling rate with 256-sample circular buffers (~1.23s window)
- **🔋 Battery Monitoring**: LiPo voltage tracking with percentage and health status
- **🎛️ Sensor Fusion**: Kalman Filter, Extended Kalman Filter, Complementary Filter support

### GUI Features
- Interactive COM port and baudrate selection
- Filter selection dropdown (None, KF, EKF, CF)
- Real-time statistics panel (sample count, rate, latest values)
- AI character recognition display with confidence scores
- Pen action insights panel with detailed metrics
- Professional color-coded plots with legends

### Development Features
- **Code Quality**: Pylint score 9.81/10 with Black formatting (line length 120)
- **Test Suite**: 30 comprehensive tests across core, AI, actions, and launch modules
- **CI/CD**: GitHub Actions workflows for automated testing and quality gates
- **Documentation**: Complete developer guides, API documentation, and examples
- **Modular Architecture**: Framework-independent service layer for easy UI swapping

## 📁 Project Structure

```text
sensor-fusion-dashboard/
├── run.py                          # 🚀 Main application entry point
├── train_character_model.py        # 🎓 LSTM character recognition training
├── requirements.txt                # 📦 Python dependencies
├── README.md                       # 📖 Project documentation
├── DEVELOPMENT.md                  # 👨‍💻 Developer guide
├── PRODUCTION_CHECKLIST.md         # ✅ Production readiness checklist
├── .pylintrc                       # 🔍 Pylint configuration (9.5+ score)
├── pyproject.toml                  # ⚙️ Black & pytest configuration
├── pytest.ini                      # 🧪 Test configuration
├── run_quality_checks.sh           # 🎯 Quality check automation script
├── check_production_ready.sh       # 🚦 Production verification script
├── .github/workflows/              # 🔄 CI/CD automation
│   ├── code-quality.yml            #    - Black + Pylint checks
│   ├── tests.yml                   #    - Multi-version Python testing
│   └── ci.yml                      #    - Combined CI with launch verification
├── data/                           # 📊 Data files and datasets
│   ├── config.json                 #    - Sensor configuration
│   ├── sample_sensor_data.csv      #    - Sample CSV data (not tracked)
│   ├── test_data.csv               #    - Test dataset (not tracked)
│   └── onhw-chars_2021-06-30/      #    - OnHW handwriting dataset (not tracked)
├── examples/                       # 💡 Example scripts and tutorials
│   ├── README.md                   #    - Examples documentation
│   ├── QUICK_START.py              #    - Quick start tutorial
│   ├── train_with_onhw.py          #    - OnHW training example
│   ├── train_model_only.py         #    - Model-only training
│   ├── train_fast.py               #    - Fast training demo
│   ├── evaluate_model.py           #    - Model evaluation
│   ├── verify_setup.py             #    - Setup verification
│   ├── example_services_usage.py   #    - Services layer demo
│   └── test_integration.py         #    - Integration testing
├── tests/                          # 🧪 Test suite (30 tests)
│   ├── README.md                   #    - Testing documentation
│   ├── test_core.py                #    - Core module tests (12 tests)
│   ├── test_ai.py                  #    - AI module tests (6 tests)
│   ├── test_actions.py             #    - Action detection tests (5 tests)
│   └── test_launch.py              #    - Launch verification tests (10 tests)
└── src/                            # 💻 Source code
    ├── __init__.py
    ├── gui/                        # 🖥️ GUI components (tkinter + matplotlib)
    │   ├── gui_app.py              #    - Main dashboard GUI (395 lines)
    │   ├── character_recognition_integration.py  # - AI recognition integration
    │   └── plotter.py              #    - Matplotlib plotting utilities
    ├── ai/                         # 🤖 AI & Machine Learning
    │   ├── lstm.py                 #    - LSTM model implementation
    │   ├── character_recognition/  #    - Character recognition module
    │   │   ├── model.py            #       - CNN+BiLSTM architecture
    │   │   ├── preprocessor.py     #       - Data preprocessing
    │   │   └── trainer.py          #       - Training loop
    │   └── utils/                  #    - AI utilities
    │       └── data_utils.py       #       - Dataset loading & processing
    ├── core/                       # ⚙️ Core functionality
    │   ├── config.py               #    - Configuration manager
    │   ├── sensor_parser.py        #    - CSV/Serial data parser (100% coverage)
    │   ├── serial_config.py        #    - COM port configuration
    │   └── data_buffers.py         #    - Legacy buffer implementation
    ├── services/                   # 🔧 Service layer (framework-independent)
    │   ├── sensor_service.py       #    - Sensor data management
    │   ├── battery_service.py      #    - Battery monitoring service
    │   ├── action_service.py       #    - Action detection service
    │   └── ai_service.py           #    - AI inference service
    ├── interfaces/                 # 🔌 Interface abstractions
    │   ├── dashboard_interface.py  #    - Abstract UI interface
    │   └── dashboard_controller.py #    - Business logic controller
    ├── actions/                    # 🎯 Action detection
    │   └── action_detector.py      #    - 5-state pen action classifier (75% coverage)
    └── utils/                      # 🛠️ Utility modules
        ├── sensor_buffers.py       #    - Thread-safe circular buffers (96% coverage)
        ├── battery_utils.py        #    - Battery voltage converter (72% coverage)
        └── sensor_fusion_filters.py #   - KF/EKF/CF implementations (22% coverage)
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

## 📊 Sensor Specifications

| Sensor | Model | Type | Channels | Sample Rate | Range | Notes |
|--------|-------|------|----------|-------------|-------|-------|
| **Top IMU** | LSM6DSO | 6-axis | Accel (3) + Gyro (3) | 208 Hz | ±16g / ±2000°/s | Primary motion sensor |
| **Rear IMU** | LSM6DSM | 6-axis | Accel (3) + Gyro (3) | 208 Hz | ±16g / ±2000°/s | Secondary motion sensor |
| **Magnetometer** | ST | 3-axis | Magnetic field (3) | 208 Hz | ±50 gauss | Heading/orientation |
| **Force Sensor** | HLP A04 | Analog | Force (1) | 208 Hz | 0-4096 ADC | Pen pressure |
| **Battery** | 2S LiPo | Voltage | Battery (1) | 208 Hz | 6.0-8.4V | Power monitoring |

**Total Data Channels**: 19 fields per sample (timestamp + 18 sensor values)

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

