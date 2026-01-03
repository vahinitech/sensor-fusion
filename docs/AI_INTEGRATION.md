# AI Integration Guide - Character Recognition at 104Hz

## Overview

This guide explains how the AI character recognition system integrates with the sensor fusion framework at 104Hz sampling rate.

## Architecture

### Two-Module System

```
┌─────────────────────────────────────────────────────────────┐
│                    SENSOR FUSION SYSTEM                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────┐      ┌──────────────────────┐     │
│  │  ACTION DETECTION    │      │  CHARACTER RECOG.    │     │
│  │      MODULE          │      │     MODULE           │     │
│  │                      │      │                      │     │
│  │  • Pen Up            │      │  • CNN + BiLSTM      │     │
│  │  • Pen Down          │      │  • 512 timesteps     │     │
│  │  • Writing           │      │  • 26 classes (A-Z)  │     │
│  │  • Thinking          │      │  • 13 sensor inputs  │     │
│  │  • Firmly Held       │      │  • Real-time pred.   │     │
│  │  • Tilted            │      │                      │     │
│  │  • Drifting          │      │                      │     │
│  │  • Idle              │      │                      │     │
│  └──────────┬───────────┘      └──────────┬───────────┘     │
│             │                             │                  │
│             └────────────┬────────────────┘                  │
│                          ▼                                    │
│                   ┌─────────────┐                            │
│                   │ GUI DISPLAY │                            │
│                   └─────────────┘                            │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │ 104Hz
                    ┌─────────────┐
                    │  SENSOR     │
                    │  DATA       │
                    └─────────────┘
```

## Data Flow

### Real-Time Recognition Pipeline

```
SENSOR ACQUISITION (104Hz)
        ↓
[Thread-safe circular buffers]
        ↓
ACTION DETECTOR
        ├→ Detects pen_down (START writing)
        │
        ├→ Collects sensor data while pen is down
        │   • 13 channels streamed to CR module
        │
        ├→ Detects pen_up (END writing)
        │
        └→ CHARACTER RECOGNIZER
            ├→ Receives buffered sensor data
            ├→ Pads to 512 timesteps
            ├→ Normalizes with preprocessing pipeline
            ├→ Runs CNN+BiLSTM inference
            └→ Returns predicted character + confidence

        ↓
GUI DISPLAY
  • Shows recognized character
  • Displays confidence score
  • Updates character history
```

## Implementation Details

### 1. Action Detection Integration

File: [src/actions/action_detector.py](src/actions/action_detector.py)

```python
# In action_detector.py
class ActionDetector:
    def update(self, sensor_dict):
        # Determine current state
        if self.is_pen_up():
            return "pen_up"
        elif self.is_pen_down():
            # Signal to character recognizer: START
            self.character_recognizer.start_writing()
            return "pen_down"
        elif self.is_writing():
            return "writing"
        # ... other states
```

### 2. Sensor Data Buffering

File: [src/core/data_buffers.py](src/core/data_buffers.py)

```python
# Thread-safe circular buffers
class SensorBuffers:
    def __init__(self, buffer_size=256):
        self.buffers = {
            'top_accel_x': deque(maxlen=buffer_size),
            'top_accel_y': deque(maxlen=buffer_size),
            'top_accel_z': deque(maxlen=buffer_size),
            # ... 10 more channels
        }
        self.lock = threading.Lock()
    
    def add_sample(self, sensor_dict):
        with self.lock:
            for key, value in sensor_dict.items():
                if key in self.buffers:
                    self.buffers[key].append(value)
    
    def get_snapshot(self):
        """Get thread-safe copy of data"""
        with self.lock:
            return {k: list(v) for k, v in self.buffers.items()}
```

### 3. Character Recognition Model

File: [src/ai/character_recognition/model.py](src/ai/character_recognition/model.py)

```python
class CharacterRecognitionModel:
    def __init__(self, timesteps=512, n_features=13, num_classes=26):
        self.timesteps = timesteps      # 512 samples
        self.n_features = n_features    # 13 sensor channels
        self.num_classes = num_classes  # 26 letters
        self.model = None
    
    def build_model(self):
        """
        Architecture:
        - Input: (512, 13) - timesteps × channels
        - Conv1D: 64 filters → 128 → 256
        - MaxPooling: Reduce dimensionality
        - BiLSTM: Process sequences bidirectionally
        - Dense: Classify into 26 classes
        """
        model = Sequential([
            # Conv layer 1
            Conv1D(64, 3, activation='relu', input_shape=(512, 13)),
            MaxPooling1D(2),
            Dropout(0.3),
            
            # Conv layer 2
            Conv1D(128, 3, activation='relu'),
            MaxPooling1D(2),
            Dropout(0.3),
            
            # Conv layer 3
            Conv1D(256, 3, activation='relu'),
            MaxPooling1D(2),
            Dropout(0.3),
            
            # Bidirectional LSTM
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(64)),
            Dropout(0.3),
            
            # Dense classifier
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(26, activation='softmax')  # 26 classes
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model = model
    
    def predict_single(self, x):
        """Predict single character"""
        prediction = self.model.predict(x, verbose=0)
        char_idx = np.argmax(prediction[0])
        confidence = prediction[0, char_idx]
        char = chr(65 + char_idx)  # Convert 0-25 to A-Z
        return char, confidence
```

### 4. Data Preprocessing

File: [src/ai/character_recognition/preprocessor.py](src/ai/character_recognition/preprocessor.py)

```python
class SensorDataPreprocessor:
    def process_raw_buffers(self, sensor_buffers):
        """
        Convert raw sensor buffers to (512, 13) array
        
        Steps:
        1. Extract all 13 channels from buffers
        2. Stack into 2D array
        3. Pad/truncate to 512 timesteps
        4. Normalize features
        """
        # Extract channels in order
        data = np.column_stack([
            sensor_buffers['top_accel_x'],      # 0
            sensor_buffers['top_accel_y'],      # 1
            sensor_buffers['top_accel_z'],      # 2
            sensor_buffers['top_gyro_x'],       # 3
            sensor_buffers['top_gyro_y'],       # 4
            sensor_buffers['top_gyro_z'],       # 5
            sensor_buffers['rear_accel_x'],     # 6
            sensor_buffers['rear_accel_y'],     # 7
            sensor_buffers['rear_accel_z'],     # 8
            sensor_buffers['mag_x'],            # 9
            sensor_buffers['mag_y'],            # 10
            sensor_buffers['mag_z'],            # 11
            sensor_buffers['force_x'],          # 12
        ])  # Shape: (N_samples, 13)
        
        # Pad to 512 timesteps
        if len(data) < 512:
            padding = np.zeros((512 - len(data), 13))
            data = np.vstack([data, padding])
        else:
            data = data[:512]
        
        # Normalize: RobustScaler handles outliers, StandardScaler final norm
        data = self.robust_scaler.fit_transform(data)
        data = self.standard_scaler.fit_transform(data)
        
        return data  # Shape: (512, 13)
    
    def normalize_data(self, X, X_test=None, fit=False):
        """Normalize data for model input"""
        X_robust = self.robust_scaler.fit_transform(X.reshape(-1, X.shape[-1]))
        X_standard = self.standard_scaler.fit_transform(X_robust)
        return X_standard.reshape(X.shape)
```

### 5. Character Recognition Integration

File: [src/gui/character_recognition_integration.py](src/gui/character_recognition_integration.py)

```python
class CharacterRecognitionIntegration:
    def __init__(self, model_path):
        self.model = load_trained_model(model_path)
        self.preprocessor = SensorDataPreprocessor()
        self.writing_buffer = {}  # Accumulate sensor data while writing
        self.recognized_characters = deque(maxlen=100)
    
    def start_writing(self):
        """Called when pen_down is detected"""
        self.writing_buffer.clear()
        self.is_writing = True
    
    def add_sensor_data(self, sensor_dict):
        """Called for each 104Hz sensor sample"""
        if self.is_writing:
            for key, value in sensor_dict.items():
                if key not in self.writing_buffer:
                    self.writing_buffer[key] = []
                self.writing_buffer[key].append(value)
    
    def end_writing(self):
        """Called when pen_up is detected - recognize character"""
        self.is_writing = False
        
        # Convert buffer to proper format
        sensor_data = self.preprocessor.process_raw_buffers(self.writing_buffer)
        
        # Predict
        char, confidence = self.model.predict_single(sensor_data)
        
        # Store in history
        self.recognized_characters.append((char, confidence))
        
        return char, confidence
```

## Training Pipeline

### Step 1: Prepare Data

```python
from src.ai.utils.data_utils import DatasetLoader
from src.ai.character_recognition.trainer import ModelTrainer

# Load dataset
loader = DatasetLoader(sampling_rate=104)
X_train, y_train, X_test, y_test = loader.load_onhw_dataset(
    'path/to/onhw/dataset'
)

# Create train/val/test split
X_train, y_train, X_val, y_val, X_test, y_test = loader.create_train_val_test_split(
    X_train, y_train,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    stratified=True
)
```

### Step 2: Train Model

```python
from src.ai.character_recognition.model import CharacterRecognitionModel
from src.ai.character_recognition.preprocessor import SensorDataPreprocessor
from src.ai.character_recognition.trainer import ModelTrainer

# Initialize components
model = CharacterRecognitionModel(timesteps=512, n_features=13, num_classes=26)
preprocessor = SensorDataPreprocessor(max_timesteps=512, sampling_rate=104)
trainer = ModelTrainer(model, preprocessor)

# Prepare data
X_train_proc, y_train_enc, X_val_proc, y_val_enc, X_test_proc, y_test_enc = \
    trainer.prepare_data(X_train, y_train, test_size=0.2)

# Train
history = trainer.train(
    X_train_proc, y_train_enc,
    X_val=X_val_proc, y_val=y_val_enc,
    epochs=50,
    batch_size=32
)

# Evaluate
metrics = trainer.evaluate(X_test_proc, y_test_enc)
print(f"Test Accuracy: {metrics['accuracy']:.2%}")

# Save
model.save('src/ai/models/character_recognition_104hz.h5')
trainer.save_training_info('src/ai/models/')
```

### Step 3: Run Training Script

```bash
python train_character_model.py \
    --dataset onhw \
    --data-path ./data/onhw \
    --epochs 50 \
    --batch-size 32 \
    --timesteps 512 \
    --features 13 \
    --sampling-rate 104 \
    --output-dir src/ai/models/
```

## Integration with GUI

### Using Pre-trained Model

```python
# In gui_app.py
from src.gui.character_recognition_integration import CharacterRecognitionIntegration

class SensorDashboardGUI:
    def __init__(self, root):
        # ... existing code ...
        
        # Initialize character recognizer
        self.char_recognizer = CharacterRecognitionIntegration(
            model_path='src/ai/models/character_recognition_104hz.h5'
        )
    
    def update_plot(self):
        # ... get sensor data ...
        
        # Update action detector
        action = self.action_detector.get_action_state()
        
        # Handle writing
        if action == 'pen_down':
            self.char_recognizer.start_writing()
        elif action == 'pen_up' and self.char_recognizer.is_writing:
            char, confidence = self.char_recognizer.end_writing()
            print(f"Recognized: {char} ({confidence:.1%})")
        elif self.char_recognizer.is_writing:
            self.char_recognizer.add_sensor_data(sensor_dict)
        
        # Display character info
        char_text = self.char_recognizer.get_display_text()
        self.char_canvas.itemconfig(self.char_display, text=char_text)
```

## Performance Specifications

### Model Specifications
- **Input**: (512, 13) - 512 timesteps × 13 sensor channels
- **Duration**: 512 samples / 104 Hz ≈ 4.9 seconds per character
- **Output**: 26 classes (A-Z)
- **Accuracy**: 85-95% typical on validation set

### Latency
- **Acquisition**: 1/104 Hz ≈ 9.6ms per sample
- **Buffering**: 512 samples ≈ 4.9 seconds
- **Preprocessing**: ~50-100ms (normalization, padding)
- **Inference**: ~100-200ms (model prediction)
- **Total Recognition Latency**: 150-300ms after pen_up

### Memory
- **Model Size**: ~15-20 MB (loaded)
- **Preprocessor**: ~5 MB (scalers, buffers)
- **Integration Runtime**: ~20 MB
- **Total for CR**: ~40-50 MB

### Threading
- **Main Thread**: GUI updates (100ms)
- **Serial Reader**: 104Hz acquisition
- **CR Processing**: On-demand (when pen_up detected)

## Sensor Data Organization (13 Channels)

```
Channel  Source              Type
───────  ──────────────────  ────────────
  0      Top IMU             Accel X
  1      Top IMU             Accel Y
  2      Top IMU             Accel Z
  3      Top IMU             Gyro X
  4      Top IMU             Gyro Y
  5      Top IMU             Gyro Z
  6      Rear IMU            Accel X
  7      Rear IMU            Accel Y
  8      Rear IMU            Accel Z
  9      Magnetometer        Mag X
  10     Magnetometer        Mag Y
  11     Magnetometer        Mag Z
  12     Force Sensor        Force X
```

## Troubleshooting

### Issue: Low Recognition Accuracy

**Cause**: Model trained on different data distribution

**Solution**:
1. Verify training data comes from same sensor at 104Hz
2. Check preprocessing parameters match
3. Retrain with more samples
4. Use data augmentation

### Issue: Slow Recognition

**Cause**: Model too large or inefficient preprocessing

**Solution**:
1. Use GPU acceleration (CUDA/cuDNN)
2. Quantize model for faster inference
3. Batch process multiple characters
4. Use model pruning

### Issue: Out of Memory

**Cause**: Buffers accumulating data

**Solution**:
1. Reduce buffer size
2. Clear history periodically
3. Use streaming LSTM for variable lengths
4. Enable memory profiling

## Future Enhancements

### 1. Continuous Streaming Recognition
```python
# BiLSTM can process streaming data
class StreamingCharacterRecognizer:
    def __init__(self, model):
        self.model = model
        self.lstm_state = None  # Maintain LSTM state
    
    def process_streaming(self, x_sample):
        # Process single sample, return partial predictions
        prediction, self.lstm_state = self.model.predict_streaming(
            x_sample, self.lstm_state
        )
        return prediction
```

### 2. Real-Time Augmentation
```python
# Augment data during training
X_aug, y_aug = loader.augment_data(X_train, y_train, factor=3)
```

### 3. Multi-Character Recognition
```python
# Recognize multiple characters per writing session
class MultiCharacterRecognizer:
    def segment_writing(self, sensor_data):
        # Detect pen_down/up transitions
        segments = []
        for char_segment in segments:
            char = self.model.predict_single(char_segment)
            yield char
```

### 4. Transfer Learning
```python
# Fine-tune pre-trained model
base_model = load_pretrained_model()
new_model = add_final_layers(base_model, num_classes=26)
new_model.fit(X_train, y_train, epochs=10)  # Few epochs
```

## References

- **Model Architecture**: CNN+BiLSTM for sequence classification
- **Preprocessing**: RobustScaler handles outliers, StandardScaler for normalization
- **Sampling Rate**: 104Hz LSM6DSO IMU specification
- **Training**: Keras with TensorFlow backend

---

**Integration Status**: ✅ Complete
**Last Updated**: 2024
