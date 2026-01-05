"""
Character Recognition Model - CNN + BiLSTM
Recognizes handwritten characters from sensor data (104Hz sampling)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Reshape,
)
from tensorflow.keras.models import Model
import os


class CharacterRecognitionModel:
    """CNN + BiLSTM model for character recognition from sensor data"""

    def __init__(self, timesteps=512, n_features=13, num_classes=26, model_path=None):
        """
        Initialize character recognition model

        Args:
            timesteps: Maximum sequence length (samples at 104Hz)
            n_features: Number of sensor channels (13: acc, gyro, mag, force)
            num_classes: Number of characters to recognize (26 for A-Z)
            model_path: Path to load pre-trained model
        """
        self.timesteps = timesteps
        self.n_features = n_features
        self.num_classes = num_classes
        self.model = None

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()

    def build_model(self):
        """Build CNN + BiLSTM architecture"""
        inp = Input(shape=(self.timesteps, self.n_features))

        # CNN feature extractor - learns local patterns
        x = Conv1D(64, kernel_size=3, activation="relu", padding="same")(inp)
        x = MaxPooling1D(pool_size=2, padding="same")(x)
        x = Dropout(0.2)(x)

        x = Conv1D(128, kernel_size=3, activation="relu", padding="same")(x)
        x = MaxPooling1D(pool_size=2, padding="same")(x)
        x = Dropout(0.2)(x)

        x = Conv1D(256, kernel_size=3, activation="relu", padding="same")(x)
        x = MaxPooling1D(pool_size=2, padding="same")(x)
        x = Dropout(0.2)(x)

        # BiLSTM temporal modeling - learns temporal patterns
        x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(x)
        x = Bidirectional(LSTM(64, return_sequences=False, dropout=0.2))(x)

        # Dense classification head
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.3)(x)

        # Output layer
        out = Dense(self.num_classes, activation="softmax")(x)

        self.model = Model(inputs=inp, outputs=out, name="character_recognition")

        # Compile
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        print("Model architecture:")
        self.model.summary()

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, callbacks=None):
        """
        Train the model

        Args:
            X_train: Training data (N, timesteps, n_features)
            y_train: Training labels (one-hot encoded)
            X_val: Validation data
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            callbacks: List of keras callbacks
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        return history

    def predict(self, X, return_confidence=False):
        """
        Predict character from sensor data

        Args:
            X: Input data (N, timesteps, n_features)
            return_confidence: Return confidence scores

        Returns:
            Predicted class indices (or tuple with confidences)
        """
        if self.model is None:
            raise ValueError("Model not loaded or built.")

        predictions = self.model.predict(X, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)

        if return_confidence:
            confidences = np.max(predictions, axis=1)
            return pred_classes, confidences

        return pred_classes

    def predict_single(self, x, return_confidence=False):
        """
        Predict single sample

        Args:
            x: Single input (timesteps, n_features)
            return_confidence: Return confidence

        Returns:
            Predicted class index
        """
        # Add batch dimension
        x_batch = np.expand_dims(x, axis=0)

        if return_confidence:
            pred_class, conf = self.predict(x_batch, return_confidence=True)
            return pred_class[0], conf[0]
        else:
            return self.predict(x_batch)[0]

    def save_model(self, path):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("No model to save.")

        self.model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load model from disk"""
        self.model = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")

    def get_summary(self):
        """Get model summary as string"""
        if self.model is None:
            return "No model built."

        from io import StringIO
        import sys

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        self.model.summary()

        summary_str = sys.stdout.getvalue()
        sys.stdout = old_stdout

        return summary_str
