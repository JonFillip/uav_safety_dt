#!/usr/bin/env python3
"""
Temporal Fusion Transformer (TFT) for UAV Safety Analysis

This module implements a Temporal Fusion Transformer model for UAV orientation prediction
and anomaly detection. The TFT architecture combines variable selection networks,
LSTM layers, and self-attention mechanisms to provide accurate multi-horizon forecasts
with uncertainty estimates.

Key features:
- Variable selection networks to identify important features
- LSTM layers for local processing of temporal patterns
- Self-attention mechanisms for capturing long-range dependencies
- Quantile forecasting for uncertainty estimation
- Interpretable feature importance
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Dense, LSTM, Dropout, Concatenate, Reshape, 
    LayerNormalization, MultiHeadAttention, Add
)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from datetime import datetime

# --- Utility Functions ---

def prepare_tft_data(df, past_history_len, future_target_len, features, target):
    """
    Prepares windowed data for the TFT model.
    """
    X, y = [], []
    # Assuming a single target feature for simplicity, which matches your use case
    target_series = df[target] 
    features_df = df[features]

    for i in range(len(df) - past_history_len - future_target_len + 1):
        past_end = i + past_history_len
        future_end = past_end + future_target_len
        
        # --- FIX: Use np.stack to convert lists into a numerical array ---
        # For the input features (X)
        x_slice = features_df.iloc[i:past_end]
        # We need to handle one or more features
        stacked_x = np.stack(x_slice[features[0]].values)
        if len(features) > 1:
            for i, feature in enumerate(features[1:]):
                stacked_feature = np.stack(x_slice[feature].values)
                stacked_x = np.dstack([stacked_x, stacked_feature])

        # For the target variable (y)
        stacked_y = np.stack(target_series.iloc[past_end:future_end].values)
        
        X.append(stacked_x)
        # The TFT model's output layer expects a shape of (prediction_horizon, num_quantiles)
        # The y data needs to be shaped as (prediction_horizon, 1) to match the median prediction
        y.append(stacked_y.reshape(-1, 1))
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# --- Core TFT Layers (Simplified for this use case) ---

class GatedResidualNetwork(tf.keras.layers.Layer):
    def __init__(self, units, dropout_rate=0.1, **kwargs):
        super(GatedResidualNetwork, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.elu_dense = Dense(units, activation='elu')
        self.linear_dense = Dense(units)
        self.dropout = Dropout(dropout_rate)
        self.gate = Dense(units, activation='sigmoid')
        self.layer_norm = LayerNormalization()
        self.project_skip = None

    def build(self, input_shape):
        if input_shape[-1] != self.units:
            self.project_skip = Dense(self.units)

    def call(self, inputs, training=None):
        x = self.elu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x, training=training)
        
        if self.project_skip is not None:
            skip = self.project_skip(inputs)
        else:
            skip = inputs
            
        gate_val = self.gate(x)
        gated_output = skip * (1.0 - gate_val) + x * gate_val
        return self.layer_norm(gated_output)

# --- Main TFT Model Class ---

class TemporalFusionTransformer:
    def __init__(self, window_size=25, prediction_horizon=5, quantiles=[0.1, 0.5, 0.9]):
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.model = None
        self.history = None

    def build_model(self, num_features=1, hidden_layer_size=64, num_attention_heads=4, dropout_rate=0.1):
        # Input layer
        past_inputs = Input(shape=(self.window_size, num_features), name='past_inputs')

        # Feature projection GRN
        projected_features = GatedResidualNetwork(hidden_layer_size, dropout_rate)(past_inputs)

        # LSTM encoder
        lstm_out = LSTM(
            hidden_layer_size,
            return_sequences=True,
            dropout=dropout_rate,
            name='lstm_encoder'
        )(projected_features)
        
        lstm_with_skip = GatedResidualNetwork(hidden_layer_size, dropout_rate)(lstm_out)

        # Self-attention layer
        attention_out = MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=hidden_layer_size,
            name='self_attention'
        )(lstm_with_skip, lstm_with_skip)
        
        attention_with_skip = GatedResidualNetwork(hidden_layer_size, dropout_rate)(attention_out)
        
        # Another GRN for post-attention processing
        final_processing = GatedResidualNetwork(hidden_layer_size, dropout_rate)(attention_with_skip)

        # Flatten the sequence to make a single prediction vector
        flat_output = tf.keras.layers.Flatten()(final_processing)

        # Output layer to predict all quantiles for all horizons at once
        output_size = self.prediction_horizon * self.num_quantiles
        outputs = Dense(output_size, name='quantile_forecasts')(flat_output)
        outputs = Reshape((self.prediction_horizon, self.num_quantiles))(outputs)
        
        self.model = Model(inputs=past_inputs, outputs=outputs)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss=self.quantile_loss)
        self.model.summary()
        return self.model

    def quantile_loss(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        error = y_true - y_pred
        qs = tf.constant(self.quantiles, dtype=tf.float32)
        
        # Expand dims to broadcast correctly
        qs = tf.reshape(qs, [1, 1, self.num_quantiles])
        
        loss = tf.maximum(qs * error, (qs - 1) * error)
        return tf.reduce_mean(loss)

    def fit(self, X_train, y_train, validation_split=0.2, epochs=50, batch_size=64, callbacks=None):
        if self.model is None:
            print("Model not built. Call build_model() first.")
            return

        if callbacks is None:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ModelCheckpoint(
                    filepath=f'models/tft_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.keras',
                    monitor='val_loss', save_best_only=True, verbose=1
                )
            ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs, batch_size=batch_size,
            callbacks=callbacks, verbose=1
        )
        return self.history

    def predict_anomaly_score(self, data):
        """
        Generates predictions and calculates an anomaly score based on prediction uncertainty.
        """
        if self.model is None:
            print("Model not trained or loaded.")
            return None
            
        predictions = self.model.predict(data)
        
        # Anomaly score is the width of the prediction interval (a direct measure of uncertainty)
        # We use the difference between the highest and lowest quantiles
        lower_quantile = predictions[:, :, 0]
        upper_quantile = predictions[:, :, -1]
        
        # We average the interval width over the prediction horizon
        interval_width = np.mean(upper_quantile - lower_quantile, axis=1)
        
        return interval_width

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # We need to save weights only as custom layers can be complex to save/load fully
        self.model.save_weights(filepath)

    def load(self, filepath, num_features=1, hidden_layer_size=64, num_attention_heads=4):
        # Build the model first to have the correct architecture
        self.build_model(
            num_features=num_features, 
            hidden_layer_size=hidden_layer_size, 
            num_attention_heads=num_attention_heads
        )
        self.model.load_weights(filepath)