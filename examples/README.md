# Examples Directory

This directory contains example scripts demonstrating various features of the Sensor Fusion Dashboard.

## Production Entry Points

**Run the main application:**
```bash
python run.py
```

**Train character recognition model:**
```bash
python train_character_model.py
```

## Example Scripts

### Data & Model Training

- **`train_with_onhw.py`** - Train on full OnHW handwriting dataset (both upper/lowercase)
  - Best for production models with comprehensive training
  - Requires OnHW dataset in `data/onhw-chars_2021-06-30/`
  - ~2-3 hours training time

- **`train_model_only.py`** - Fast training on OnHW uppercase subset
  - Faster training (uppercase A-Z only, 26 classes)
  - Good for quick iteration and testing
  - ~30-45 minutes training time

- **`train_fast.py`** - Quick training for development
  - Minimal model for rapid testing
  - Uses small OnHW dataset subset
  - ~5-10 minutes training time

- **`evaluate_model.py`** - Evaluate trained model performance
  - Tests model accuracy on OnHW test set
  - Generates classification reports
  - Requires trained model at `src/ai/models/character_model.h5`

### Integration & Testing

- **`example_services_usage.py`** - Demonstrates service layer usage
  - Shows how to use SensorService, BatteryService, ActionService independently
  - Example REST API structure with services

- **`test_integration.py`** - Integration test for all components
  - Tests data flow from serial/CSV → processing → prediction
  - Verifies sensor parsing, buffering, and action detection

- **`verify_setup.py`** - Verify installation and dependencies
  - Checks all required packages are installed
  - Validates configuration files
  - Tests serial port detection (if hardware connected)

- **`QUICK_START.py`** - Quick start guide
  - Simple example to get started
  - Demonstrates basic sensor data processing

## Training Workflow

For production:

```bash
# 1. Train model (choose one)
python train_character_model.py              # Default recommended training
python examples/train_model_only.py          # Uppercase only (faster)
python examples/train_with_onhw.py           # Full OnHW dataset

# 2. Evaluate results
python examples/evaluate_model.py

# 3. Run application
python run.py
```

For development/testing:

```bash
python examples/train_fast.py                # Quick training
python examples/evaluate_model.py            # Check accuracy
python run.py                                # Launch GUI
```

## Testing Services

```bash
# Test services independently
python examples/example_services_usage.py

# Full integration test
python examples/test_integration.py

# Verify setup
python examples/verify_setup.py
```

## Configuration

See root-level configuration files:
- `data/config.json` - Sensor and dashboard configuration
- `.pylintrc` - Code quality settings
- `pytest.ini` - Testing configuration
- `pyproject.toml` - Black formatting settings

## Hardware Testing

To test with actual hardware:

1. Connect sensor device via USB/serial
2. Run verification: `python examples/verify_setup.py`
3. Launch dashboard: `python run.py`
4. Select appropriate COM port and connect
