# Installation & Setup Guide

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- macOS, Linux, or Windows

## Installation

### 1. Clone or Download Project

```bash
cd /path/to/sensor_fusion
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
```

### 3. Activate Virtual Environment

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install GUI Framework (macOS only)

If tkinter is not available:

```bash
brew install python-tk@3.13
```

### 6. Verify Installation

```bash
cd src
python test_gui.py
```

You should see all tests pass:
```
✓ PASS - Module Imports
✓ PASS - SerialConfig
✓ PASS - SensorBuffers
✓ PASS - SensorParser
✓ PASS - CSV Reading

Results: 5/5 tests passed
✓ ALL VALIDATIONS PASSED - GUI APP IS READY
```

## Project Structure

```
sensor_fusion/
├── src/                    # Python source code
├── data/                   # Data files (CSV, config)
├── docs/                   # Documentation
├── requirements.txt        # Dependencies
└── README.md              # Overview
```

## File Organization

### Source Code (`src/`)
- `gui_app.py` - Main GUI application
- `test_gui.py` - Validation tests
- `serial_config.py` - Serial utilities
- `sensor_buffers.py` - Data buffers
- `sensor_parser.py` - Data parser
- `data_reader.py` - Data readers

### Data Files (`data/`)
- `sample_sensor_data.csv` - Test data (76 samples)
- `test_data.csv` - Additional test data
- `config.json` - Configuration settings

### Documentation (`docs/`)
- `SETUP.md` - This file
- `ARCHITECTURE.md` - System architecture
- `USAGE.md` - How to use
- `CLEANUP_SUMMARY.md` - Cleanup notes

## Troubleshooting

### tkinter Not Found (macOS)
```bash
brew install python-tk@3.13
```

### Serial Port Not Found
- Verify USB connection
- Check device manager for port name
- Try refreshing port list in GUI

### Module Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Virtual Environment Issues
```bash
# Recreate venv
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Next Steps

1. Review [USAGE.md](USAGE.md) for running the application
2. Check [ARCHITECTURE.md](ARCHITECTURE.md) for system design
3. Run `cd src && python test_gui.py` to verify setup
