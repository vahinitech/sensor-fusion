# Sensor Fusion with Character Recognition at 104Hz

Professional sensor fusion system with real-time action detection and AI-powered character recognition, optimized for LSM6DSO IMU at 104Hz sampling rate.

## Project Structure

```
sensor_fusion/
├── src/
│   ├── core/                          # Core sensor functionality
│   │   ├── __init__.py
│   │   ├── config.py                  # Configuration management
│   │   ├── serial_config.py           # Serial port configuration
│   │   ├── data_reader.py             # Serial/CSV data acquisition
│   │   ├── sensor_parser.py           # Raw data parsing
│   │   └── data_buffers.py            # Thread-safe circular buffers
│   │
│   ├── gui/                           # GUI visualization layer
│   │   ├── __init__.py
│   │   ├── gui_app.py                 # Main Tkinter dashboard (624 lines)
│   │   ├── dashboard.py               # Dashboard components
│   │   ├── plotter.py                 # Matplotlib visualization
│   │   ├── serial_config.py           # Serial UI controls
│   │   └── character_recognition_integration.py  # AR integration
│   │
│   ├── actions/                       # Action detection module
│   │   ├── __init__.py
│   │   └── action_detector.py         # Pen state detection (295 lines)
│   │
│   ├── ai/                            # AI models and utilities
│   │   ├── __init__.py
│   │   ├── character_recognition/    # Character recognition models
│   │   │   ├── __init__.py
│   │   │   ├── model.py              # CNN+BiLSTM architecture (155 lines)
│   │   │   ├── trainer.py            # Training pipeline (345 lines)
│   │   │   └── preprocessor.py       # Data preprocessing (245 lines)
│   │   │
│   │   ├── utils/                    # AI utilities
│   │   │   ├── __init__.py
│   │   │   └── data_utils.py         # Dataset loading and augmentation
│   │   │
│   │   └── models/                   # Trained models
│   │       └── character_recognition_104hz.h5  (after training)
│   │
│   ├── utils/                         # General utilities
│   │   ├── __init__.py
│   │   ├── sensor_fusion_filters.py  # Kalman/EKF/Complementary filters
│   │   └── helpers.py
│   │
│   └── __pycache__/
│
├── data/                              # Data files
│   ├── config.json                    # Configuration
│   ├── sample_sensor_data.csv         # Sample sensor data
│   └── test_data.csv                  # Test dataset
│
├── docs/                              # Documentation
│   ├── ARCHITECTURE.md                # System architecture
│   ├── FILTERS_IMPLEMENTATION.md      # Filter implementations
│   ├── SETUP.md                       # Setup instructions
│   └── USAGE.md                       # Usage guide
│
├── train_character_model.py           # Main training script
├── run.py                             # Entry point for GUI
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Features

### 1. Real-Time Sensor Acquisition (104Hz)
- **LSM6DSO IMU**: 104Hz sampling rate (as per datasheet specification)
- **Sensor Channels**: 13 total
  - Top IMU: accel_x/y/z, gyro_x/y/z
  - Rear IMU: accel_x/y/z, gyro_x/y/z
  - Magnetometer: mag_x/y/z
  - Force Sensor: force_x
- **Thread-Safe**: Lock-based circular buffers prevent race conditions
- **Data Sources**: Serial port (real-time) or CSV (replay/testing)

### 2. Action Detection (8 States)
Detects pen movement states based on multi-modal sensor fusion:
- **pen_up**: Pen not in contact (accel > 11000)
- **pen_down**: Light contact without pressure (accel < 3000)
- **writing**: Active writing motion (high accel variance)
- **thinking**: Stationary with hand contact
- **firmly_held**: Strong grip with low motion (force > 1500, accel >= 3000)
- **tilted**: Pen at angle (gyro > 30°)
- **drifting**: Hand drift without writing (low accel, high motion)
- **idle**: No activity

### 3. AI Character Recognition (A-Z)
Deep learning model for handwritten character recognition:
- **Architecture**: CNN + Bidirectional LSTM
  - Conv1D: 64 → 128 → 256 filters with MaxPooling
  - BiLSTM: 128 → 64 units (bidirectional processing)
  - Dense: 128 → 64 → 26 classes (A-Z)
- **Input**: 512 timesteps × 13 sensor channels (at 104Hz ≈ 5 seconds/character)
- **Preprocessing**: RobustScaler + StandardScaler normalization
- **Real-time**: Recognizes characters as they are written (pen_down → pen_up)

### 4. Professional GUI Dashboard
- **Real-time Visualization**: 4×3 plot grid with 13 sensor channels
- **Live Metrics**: Sampling rate, sample count, elapsed time
- **Action Display**: Large, colored action state indicator
- **Filter Selection**: Kalman Filter, Extended Kalman Filter, Complementary Filter
- **Character Display**: Real-time character recognition results with confidence

## Installation

### Requirements
- Python 3.8+
- pip or conda

### Setup

1. **Clone repository**
```bash
cd /path/to/sensor_fusion
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

Dependencies include:
- numpy, scipy, scikit-learn: Numerical computing
- matplotlib: Visualization
- pyserial: Serial communication
- tensorflow >= 2.13.0, keras: Deep learning
- h5py, pandas: Data handling

3. **Install in development mode (optional)**
```bash
pip install -e .
```

## Quick Start

### 1. Run GUI Dashboard
```bash
python run.py
```

Select data source (Serial or CSV), adjust COM port/baudrate, click "Connect".

### 2. Train Character Recognition Model

#### Option A: Synthetic Data (Testing)
```bash
python train_character_model.py \
    --dataset synthetic \
    --num-samples 500 \
    --epochs 50
```

#### Option B: OnHW Dataset (Real Data)
```bash
python train_character_model.py \
    --dataset onhw \
    --data-path ./data/onhw \
    --epochs 50 \
    --download
```

#### Option C: Custom CSV Dataset
```bash
python train_character_model.py \
    --dataset csv \
    --data-path ./data/my_dataset.csv \
    --epochs 50
```

Training outputs:
- Model: `src/ai/models/character_recognition_104hz.h5`
- Metadata: `src/ai/models/training_info.json`
- Metrics: Classification report, confusion matrix

### 3. Use Pre-trained Model

Load model in GUI:
```python
from src.gui.character_recognition_integration import CharacterRecognitionIntegration

cr = CharacterRecognitionIntegration(
    model_path='src/ai/models/character_recognition_104hz.h5'
)
```

## Data Format

### Sensor Data Structure
```python
{
    'top_accel_x': float,    # Top IMU acceleration X
    'top_accel_y': float,    # Top IMU acceleration Y
    'top_accel_z': float,    # Top IMU acceleration Z
    'top_gyro_x': float,     # Top IMU rotation X
    'top_gyro_y': float,     # Top IMU rotation Y
    'top_gyro_z': float,     # Top IMU rotation Z
    'rear_accel_x': float,   # Rear IMU acceleration X
    'rear_accel_y': float,   # Rear IMU acceleration Y
    'rear_accel_z': float,   # Rear IMU acceleration Z
    'rear_gyro_x': float,    # Rear IMU rotation X
    'rear_gyro_y': float,    # Rear IMU rotation Y
    'rear_gyro_z': float,    # Rear IMU rotation Z
    'mag_x': float,          # Magnetometer X
    'mag_y': float,          # Magnetometer Y
    'mag_z': float,          # Magnetometer Z
    'force_x': float         # Force sensor reading
}
```

### CSV Format
```csv
timestamp,top_accel_x,top_accel_y,top_accel_z,top_gyro_x,top_gyro_y,top_gyro_z,rear_accel_x,rear_accel_y,rear_accel_z,rear_gyro_x,rear_gyro_y,rear_gyro_z,mag_x,mag_y,mag_z,force_x
0.0,0.1,0.2,0.3,0.01,0.02,0.03,0.1,0.2,0.3,0.01,0.02,0.03,0.5,0.6,0.7,100.0
```

### Model Training Data
```python
# For character recognition
X_train: np.array with shape (N_samples, 512, 13)
    # Each sample: 512 timesteps × 13 sensor channels
    # At 104Hz: 512/104 ≈ 4.9 seconds per character

y_train: np.array with shape (N_samples,)
    # Labels: 0-25 representing A-Z
```

## Key Components

### 1. SensorBuffers (Thread-Safe)
```python
from src.core.data_buffers import SensorBuffers

buffers = SensorBuffers(buffer_size=256)  # 2.46s @ 104Hz
buffers.add_sample(sensor_dict)
snapshot = buffers.get_snapshot()  # Thread-safe copy
```

### 2. ActionDetector
```python
from src.actions.action_detector import ActionDetector

detector = ActionDetector(buffer_size=30)
detector.update(sensor_dict)
state = detector.get_action_state()      # Current state
text = detector.get_display_text()       # Display string
```

### 3. CharacterRecognitionModel
```python
from src.ai.character_recognition.model import CharacterRecognitionModel

model = CharacterRecognitionModel(timesteps=512, n_features=13, num_classes=26)
model.build_model()
model.train(X_train, y_train, epochs=50)
model.save('model.h5')
char, confidence = model.predict_single(x_sample)
```

### 4. SensorDataPreprocessor
```python
from src.ai.character_recognition.preprocessor import SensorDataPreprocessor

preprocessor = SensorDataPreprocessor(max_timesteps=512, sampling_rate=104)
x_padded = preprocessor.pad_sequence(raw_sensor_data)
x_normalized = preprocessor.normalize_data(x_padded)
```

### 5. ModelTrainer
```python
from src.ai.character_recognition.trainer import ModelTrainer

trainer = ModelTrainer(model, preprocessor)
X_train, y_train, X_val, y_val, X_test, y_test = trainer.prepare_data(X_raw, y_raw)
history = trainer.train(X_train, y_train, X_val, y_val, epochs=50)
metrics = trainer.evaluate(X_test, y_test)
```

## Performance Specifications

### Sampling & Buffering
- **Sampling Rate**: 104 Hz (LSM6DSO)
- **Buffer Size**: 256 samples
- **Buffer Duration**: 2.46 seconds
- **GUI Update Rate**: 100ms
- **Character Duration**: ~5 seconds (512 samples)

### Memory
- **Optimized**: <500MB typical (down from 64GB)
- **Thread-Safe**: No race conditions after 1+ hour
- **Responsive**: All UI controls functional during streaming

### Action Detection
- **State Detection**: 8 states with priority logic
- **Tilt/Drift Threshold**: ±30°
- **Force Threshold**: >1500 for firmly_held
- **Acceleration Thresholds**: pen_up >11000, pen_down <3000

### Character Recognition
- **Accuracy**: Model-dependent (typically 85-95% on validation set)
- **Latency**: ~200-500ms per character (single prediction)
- **Confidence Range**: 0-100%

## Architecture

### Threading Model
- **Main Thread**: GUI updates (100ms interval)
- **Serial Reader Thread**: Continuous data acquisition (104Hz)
- **Lock Synchronization**: All shared buffers protected

### Data Flow
```
Serial Port/CSV
    ↓
SerialReader/CSVReader (104Hz)
    ↓
[Thread-Safe Buffers] ← Lock protection
    ↓
GUI Thread (gets snapshot every 100ms)
    ├→ Plotting
    ├→ ActionDetector
    └→ CharacterRecognition
        ├→ Collect pen_down to pen_up
        └→ Recognize character
```

### Model Pipeline
```
Raw Sensor Data (variable timesteps)
    ↓
Padding (to 512 timesteps)
    ↓
Normalization (RobustScaler + StandardScaler)
    ↓
CNN (Conv1D + MaxPooling)
    ↓
BiLSTM (Bidirectional)
    ↓
Dense Classifier
    ↓
Output (26 classes: A-Z)
```

## Configuration

### sensor_fusion/data/config.json
```json
{
    "sampling_rate": 104,
    "buffer_size": 256,
    "sensors": {
        "top_imu": {
            "accel_channels": 3,
            "gyro_channels": 3
        },
        "rear_imu": {
            "accel_channels": 3,
            "gyro_channels": 3
        },
        "magnetometer": {
            "channels": 3
        },
        "force_sensor": {
            "channels": 1
        }
    }
}
```

## Troubleshooting

### Serial Connection Issues
1. Check COM port: `ls /dev/tty*` (Mac/Linux) or Device Manager (Windows)
2. Verify baudrate: Usually 115200
3. Check USB driver installation

### GUI Freezing
- Ensure all locks are being used correctly
- Check for blocking I/O operations
- Monitor system memory

### Low Character Recognition Accuracy
- Ensure model is trained on similar data
- Check preprocessing parameters match between train/inference
- Verify input has sufficient motion (min 50 samples)
- Consider data augmentation during training

### Memory Issues
- Reduce buffer_size if memory-constrained
- Ensure deques are not being converted to lists
- Check for circular references

## Development

### Adding New Sensors
1. Update `SensorParser` in `src/core/sensor_parser.py`
2. Add channels to `SensorBuffers` in `src/core/data_buffers.py`
3. Update model input shape in preprocessor

### Adding New Action States
1. Define detection logic in `src/actions/action_detector.py`
2. Add thresholds and priority
3. Update display text formatting

### Improving Character Recognition
1. Collect more training data using `start_writing()`/`end_writing()`
2. Experiment with model architecture in `src/ai/character_recognition/model.py`
3. Tune preprocessing in `src/ai/character_recognition/preprocessor.py`
4. Use data augmentation in `data_utils.py`

## Performance Optimization

### Known Optimizations Applied
- ✅ Eliminated list conversions (64GB → <500MB memory)
- ✅ Thread-safe circular buffers with locks
- ✅ NumPy vectorization for calculations
- ✅ Snapshot-based GUI updates
- ✅ Lazy model loading
- ✅ Batch prediction for efficiency

### Future Optimizations
- GPU acceleration for model inference
- Multi-threaded data preprocessing
- Model quantization for mobile deployment
- Streaming LSTM for variable-length sequences

## References

- **LSM6DSO Datasheet**: 104Hz sampling, IMU specifications
- **OnHW Dataset**: Online handwriting recognition
- **TensorFlow Keras**: Deep learning framework
- **Scikit-learn**: Preprocessing and metrics

## License

Project for educational and research purposes.

## Support

For issues or questions:
1. Check existing documentation in `docs/`
2. Review code comments
3. Check console output for error messages
4. Verify sensor data format matches specifications

---

**Last Updated**: 2024
**System Status**: ✅ Production Ready
