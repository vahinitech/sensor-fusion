# Character Recognition Training & Evaluation Guide

## 🎯 Overview

We have **separated the training pipeline into 3 phases**:

1. **Training Phase** - Train the CNN+BiLSTM model on OnHW dataset
2. **Evaluation Phase** - Test the model and report metrics  
3. **GUI Usage** - Use the trained model in real-time handwriting recognition

---

## 📊 What We're Building

### The Model: CNN + BiLSTM Architecture

**CNN (Convolutional Neural Network) - Feature Extraction:**
- Extracts spatial patterns from sensor data
- 3 convolutional layers: 64 → 128 → 256 filters
- Max pooling to reduce dimensions
- Learns local motion patterns

**BiLSTM (Bidirectional LSTM) - Sequence Understanding:**
- Processes temporal sequences in both directions
- 2 BiLSTM layers: 256 → 128 units
- Captures character stroke dependencies
- Understands handwriting dynamics

**Output:**
- 26 neurons (one for each letter A-Z)
- Softmax activation for probability distribution
- Recognizes uppercase handwritten characters

### Dataset: OnHW (Online Handwriting)

**Source:** Fraunhofer Institute professional dataset
- **Training samples:** 11,542 handwritten characters
- **Test samples:** 4,108 held-out samples
- **Classes:** 26 uppercase letters (A-Z)
- **Sensor data:** 13 channels @ 104Hz sampling
  - 3-axis accelerometer (front)
  - 3-axis gyroscope
  - 3-axis accelerometer (rear)
  - 3-axis magnetometer
  - 1-axis force sensor

**Data format:** Variable-length sequences (17-100+ timesteps)
- Padded to 512 timesteps (~5 seconds at 104Hz)
- Normalized using RobustScaler + StandardScaler

---

## 🚀 Step-by-Step Usage

### Phase 1: Training (Train the Model)

**Run:**
```bash
source .venv/bin/activate
python train_model_only.py
```

**What happens:**
1. Loads 11,542 training samples from OnHW dataset
2. Preprocesses and normalizes sensor data
3. Trains CNN+BiLSTM for 30 epochs (~15-20 minutes)
4. Uses 90% for training, 10% for validation
5. Saves trained model to `src/ai/models/character_model.h5`

**Expected output:**
```
[Epoch 1/30] - accuracy: 0.45, val_accuracy: 0.48
[Epoch 2/30] - accuracy: 0.62, val_accuracy: 0.63
...
[Epoch 30/30] - accuracy: 0.88, val_accuracy: 0.89
✓ Model saved to src/ai/models/character_model.h5
```

**Performance indicators:**
- Training accuracy should increase epoch by epoch
- Validation accuracy should reach 85-90%
- Loss should decrease over time
- No warnings about data or shapes = good!

### Phase 2: Evaluation (Test the Model)

**Run:**
```bash
python evaluate_model.py
```

**What happens:**
1. Loads the trained model from disk
2. Loads held-out test set (4,108 samples)
3. Predicts character for each test sample
4. Calculates overall accuracy
5. Shows per-character performance

**Expected output:**
```
Test Accuracy: 87.5% (3596 / 4108 samples correct)

Per-Character Performance:
           precision  recall  f1-score  support
A            0.85     0.82     0.84      158
B            0.88     0.91     0.89      142
...
Z            0.84     0.86     0.85      165
```

### Phase 3: GUI Usage (Real-time Recognition)

**Run:**
```bash
./run.py
```

**What you get:**
- Real-time sensor plots (accelerometer, gyroscope, magnetometer, force)
- **Pen Action Insights** panel: Shows pen state (writing, thinking, pen up/down)
- **AI Character Recognition** panel: 
  - Shows "Ready - Model: Loaded"
  - Displays recognized character and confidence when you write
  - Automatically starts/stops collection on pen down/up

---

## 📈 Training Details

### Architecture Statistics

```
Total Parameters: 710,874
Trainable Params: 710,874 (2.71 MB)

Layer breakdown:
- Input: 512 timesteps × 13 features
- Conv1D: 64 filters → 128 → 256
- BiLSTM: 256 units → 128 units
- Dense: 128 → 64 → 26 (output)
```

### Training Configuration

- **Optimizer:** Adam (learning rate 0.001)
- **Loss:** Categorical crossentropy
- **Batch size:** 64 samples/batch
- **Epochs:** 30
- **Callbacks:**
  - Early stopping (patience=10)
  - Learning rate reduction on plateau
- **Data split:**
  - Training: 90% (10,387 samples)
  - Validation: 10% (1,155 samples)
  - Test: 4,108 samples (held-out for final evaluation)

### System Optimization

- **Multiprocessing:** Uses 75% of available CPU cores (automatically detected)
- **TensorFlow threading:** Configured for parallel training
- **GPU support:** If available, automatically enabled with memory growth

---

## 🔧 Troubleshooting

### Training issues:

**"Model not found at X_train.npy"**
- The OnHW dataset must be in `data/onhw-chars_2021-06-30/`
- Download from: https://www.iis.fraunhofer.de/de/ff/lv/dataanalytics/anwproj/schreibtrainer/onhw-dataset.html

**"CUDA out of memory"**
- GPU memory exhausted
- Solution: Reduce batch size (32 instead of 64) in the script

**"Training stalls at epoch X"**
- Ctrl+C to stop, then check system resources
- Try closing other applications

### Evaluation issues:

**"Model not found at src/ai/models/character_model.h5"**
- Run training first: `python train_model_only.py`

**Accuracy is very low (<70%)**
- The model may not have trained properly
- Check training output for increasing accuracy values

### GUI issues:

**"No Model" shown in AI Recognition panel**
- Model file doesn't exist or path is wrong
- Run: `python train_model_only.py` first
- Model should save to: `src/ai/models/character_model.h5`

---

## 📊 What to Expect

### Training Phase (15-20 minutes):
- ✅ System config shows available cores
- ✅ Dataset loads successfully
- ✅ Model builds with 710K parameters
- ✅ Training starts epoch-by-epoch
- ✅ Accuracy increases from ~40% → 88%+
- ✅ Model saves successfully

### Evaluation Phase (2-3 minutes):
- ✅ Model loads from disk
- ✅ Test set preprocessed
- ✅ Per-sample predictions generated
- ✅ Accuracy ~85-90% on held-out test set

### GUI Phase:
- ✅ Model automatically loaded
- ✅ "Model: Loaded" shown in AI panel
- ✅ Write characters to recognize them
- ✅ Confidence displayed for each character

---

## 🎓 Learning Outcomes

After successful training, you'll have:

1. **A trained neural network** that:
   - Extracts features from 13-channel IMU data
   - Recognizes handwritten uppercase letters
   - Achieves ~87% accuracy on unseen data
   
2. **Understanding of:**
   - CNN+BiLSTM architectures
   - Sequence processing (timesteps)
   - Normalization and preprocessing
   - Training vs validation vs test splits
   - Real-time sensor-based recognition

3. **A deployable model** that:
   - Runs in real-time on your sensor hardware
   - Integrates with the GUI dashboard
   - Can be retrained with your own data

---

## 🚀 Next Steps

After successful training & evaluation:

```bash
# Step 1: Train the model
python train_model_only.py

# Step 2: Evaluate performance  
python evaluate_model.py

# Step 3: Use in GUI
./run.py
```

---

## 📝 Files Generated

After running all phases:

```
src/ai/models/
├── character_model.h5          ← Trained model (2.71 MB)
├── model_metadata.json         ← Training info
└── training_info.json          ← Additional metadata

data/onhw-chars_2021-06-30/
└── onhw2_upper_dep_0/
    ├── X_train.npy             ← Training sensor data
    ├── y_train.npy             ← Training labels
    ├── X_test.npy              ← Test sensor data
    └── y_test.npy              ← Test labels
```

---

## 💡 Tips for Better Results

1. **Increase epochs** (in `train_model_only.py`): Change `epochs=30` to `epochs=50` for potentially better accuracy
2. **Use full dataset**: Change from `onhw2_upper_dep_0` to `onhw2_both_dep_0` to train on all 52 characters (uppercase + lowercase)
3. **Monitor GPU**: On systems with GPU, training will be faster than CPU
4. **Larger batch size**: If you have enough memory, use `batch_size=128` for faster training

---

**Questions?** Check the terminal output during training - it provides detailed progress information!
