import os
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Bidirectional, LSTM,
    Dense, Dropout, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# ============================================================
# 1. PATHS & DATA LOADING
# ============================================================
# download and extract the dataset https://www2.iis.fraunhofer.de/LV-OnHW/onhw-chars_2021-06-30.zip
# if is downloaded, set the correct path here:
path = r"data/onhw-chars_2021-06-30.zip"

X_train = np.load(os.path.join(path, "X_train.npy"), allow_pickle=True)
X_test  = np.load(os.path.join(path, "X_test.npy"),  allow_pickle=True)
y_train = np.load(os.path.join(path, "y_train.npy"), allow_pickle=True)
y_test  = np.load(os.path.join(path, "y_test.npy"),  allow_pickle=True)

print("Raw shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train unique:", np.unique(y_train))
print("y_test  unique:", np.unique(y_test))

# ============================================================
# 2. FIX LABELS: Convert strings to integers
# ============================================================
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded  = label_encoder.transform(y_test)

num_classes = len(label_encoder.classes_)
print(f"Number of classes: {num_classes}")
print(f"Classes: {label_encoder.classes_}")

y_train_cat = to_categorical(y_train_encoded, num_classes)
y_test_cat  = to_categorical(y_test_encoded, num_classes)

# ============================================================
# 3. FIND GLOBAL MAX LENGTH
# ============================================================
def find_global_max_length(X_train, X_test):
    all_lengths = []
    for sample in X_train:
        sample_array = np.array(sample)
        all_lengths.append(len(sample_array))
    for sample in X_test:
        sample_array = np.array(sample)
        all_lengths.append(len(sample_array))
    return max(all_lengths)

max_seq_len = find_global_max_length(X_train, X_test)
print(f"Global max length: {max_seq_len}")

# ============================================================
# 4. SAFE NUMERIC CONVERSION → FIXED LENGTH
# ============================================================
def safe_numeric_convert_flat(data, max_seq_len):
    cleaned = []
    for sample in data:
        if isinstance(sample, (list, tuple, np.ndarray)):
            numeric_sample = np.array(sample, dtype=np.float32).ravel()
        else:
            numeric_sample = np.array([float(sample)], dtype=np.float32)
        cleaned.append(numeric_sample)

    max_len = min(max(len(s) for s in cleaned), max_seq_len)
    padded = np.zeros((len(cleaned), max_len), dtype=np.float32)
    for i, s in enumerate(cleaned):
        s = s.ravel()
        L = min(len(s), max_len)
        padded[i, :L] = s[:L]
    return padded

X_train_flat = safe_numeric_convert_flat(X_train, max_seq_len=max_seq_len)
X_test_flat  = safe_numeric_convert_flat(X_test,  max_seq_len=max_seq_len)

print("\nAfter padding (flat):")
print("X_train_flat:", X_train_flat.shape)
print("X_test_flat :", X_test_flat.shape)

# Clean NaN/inf
X_train_flat = np.nan_to_num(X_train_flat, nan=0.0, posinf=0.0, neginf=0.0)
X_test_flat  = np.nan_to_num(X_test_flat,  nan=0.0, posinf=0.0, neginf=0.0)

# ============================================================
# 5. RESHAPE TO (T, C) WITH 16 CHANNELS (UPDATED: force now has X,Y,Z)
# ============================================================
n_channels = 16  # UPDATED: Was 13, now 16 because force has 3 channels (X, Y, Z)
def reshape_to_seq(X_flat):
    total_len = X_flat.shape[1]
    timesteps = total_len // n_channels
    used_len = timesteps * n_channels
    X_trim = X_flat[:, :used_len]
    X_seq = X_trim.reshape(X_flat.shape[0], timesteps, n_channels)
    return X_seq

X_train_seq = reshape_to_seq(X_train_flat)
X_test_seq  = reshape_to_seq(X_test_flat)

print("\nSequence shapes:")
print("X_train_seq:", X_train_seq.shape)
print("X_test_seq :", X_test_seq.shape)

# ============================================================
# 6. SENSOR SLICING & COMBINATIONS (UPDATED: force now 3 channels)
# ============================================================
def split_sensors_3d(X):
    front_acc = X[:, :, 0:3]
    gyro      = X[:, :, 3:6]
    rear_acc  = X[:, :, 6:9]
    magnet    = X[:, :, 9:12]
    force     = X[:, :, 12:16]  # UPDATED: Was 12:13, now 12:16 (3 channels: X, Y, Z)
    return front_acc, gyro, rear_acc, magnet, force

def combine_seq(*xs):
    return np.concatenate(xs, axis=-1)

f_acc_tr, gyro_tr, r_acc_tr, mag_tr, force_tr = split_sensors_3d(X_train_seq)
f_acc_te, gyro_te, r_acc_te, mag_te, force_te = split_sensors_3d(X_test_seq)

sensor_combos = {
    "front_acc":      ((f_acc_tr,), (f_acc_te,)),
    "front_acc_gyro": ((f_acc_tr, gyro_tr), (f_acc_te, gyro_te)),
    "front_acc_rear": ((f_acc_tr, r_acc_tr), (f_acc_te, r_acc_te)),
    "acc_gyro":       ((f_acc_tr, r_acc_tr, gyro_tr), (f_acc_te, r_acc_te, gyro_te)),
    "all_sensors":    ((X_train_seq,), (X_test_seq,)),
}

# ============================================================
# 7. CNN + Bi-LSTM CLASSIFIER
# ============================================================
def build_cnn_bilstm_classifier(timesteps, n_features, num_classes):
    inp = Input(shape=(timesteps, n_features))
    
    # CNN feature extractor
    x = Conv1D(64, 3, activation='relu', padding='same')(inp)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    
    # Bi-LSTM temporal modeling
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Bidirectional(LSTM(32, return_sequences=False))(x)
    
    # Dense head
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inp, out, name="cnn_bilstm_classifier")
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ============================================================
# 8. TRAIN & EVALUATE FOR EACH SENSOR COMBINATION
# ============================================================
results = {}

for name, (train_parts, test_parts) in sensor_combos.items():
    print("\n" + "="*60)
    print(f"Processing combination: {name}")
    print("="*60)

    # Build combination tensors
    X_train_c = combine_seq(*train_parts)
    X_test_c  = combine_seq(*test_parts)

    print("Combined shapes:", X_train_c.shape, X_test_c.shape)

    # Scale per combination
    N_tr, T, C = X_train_c.shape
    N_te = X_test_c.shape[0]

    X_train_flat_c = X_train_c.reshape(N_tr, T * C)
    X_test_flat_c  = X_test_c.reshape(N_te, T * C)

    robust_scaler = RobustScaler()
    X_train_robust = robust_scaler.fit_transform(X_train_flat_c)
    X_test_robust  = robust_scaler.transform(X_test_flat_c)

    std_scaler = StandardScaler()
    X_train_scaled_flat = std_scaler.fit_transform(X_train_robust)
    X_test_scaled_flat  = std_scaler.transform(X_test_robust)

    X_train_scaled = X_train_scaled_flat.reshape(N_tr, T, C)
    X_test_scaled  = X_test_scaled_flat.reshape(N_te, T, C)

    # Build and train CNN+BiLSTM
    model = build_cnn_bilstm_classifier(T, C, num_classes)
    
    model.fit(
        X_train_scaled, y_train_cat,
        validation_data=(X_test_scaled, y_test_cat),
        epochs=20,
        batch_size=64,
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
    y_pred_prob = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    print(f"[{name}] Test Accuracy: {test_acc:.4f}")
    print(f"[{name}] Classification report:")
    print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))
    
    results[name] = test_acc

print("\n" + "="*60)
print("FINAL RESULTS (CNN+BiLSTM):")
print("="*60)
for k, v in results.items():
    print(f"{k:15s}: {v:.4f}")
