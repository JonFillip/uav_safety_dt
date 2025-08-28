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
    Prepares windowed data for the TFT model by splitting the list in each row
    into a history and a future target.
    """
    X, y = [], []
    
    # The 'r_zero' column contains lists which are our sequences
    sequences = df[features[0]].to_list()
    
    # Process each sequence (from each row)
    for seq in sequences:
        # The total length must accommodate both history and future
        if len(seq) >= past_history_len + future_target_len:
            past_end = past_history_len
            future_end = past_end + future_target_len
            
            # Append the history part for X
            X.append(seq[0:past_end])
            # Append the future part for y
            y.append(seq[past_end:future_end])

    # Convert to numpy arrays
    X_array = np.array(X, dtype=np.float32)
    y_array = np.array(y, dtype=np.float32)
    
    # Reshape arrays to add the "features" dimension, which is 1
    # X shape becomes: (num_samples, 20, 1)
    # y shape becomes: (num_samples, 5, 1) for the quantile loss
    X_array = X_array.reshape(X_array.shape[0], X_array.shape[1], 1)
    y_array = y_array.reshape(y_array.shape[0], y_array.shape[1], 1)
    
    return X_array, y_array

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

    def get_tunable_model(self, hp):
        """Builds a tunable TFT model for KerasTuner."""
        # Define Hyperparameters
        hidden_layer_size = hp.Int('hidden_layer_size', min_value=16, max_value=64, step=16)
        num_attention_heads = hp.Int('num_attention_heads', min_value=2, max_value=4, step=2)
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.4, step=0.1)
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        # Reuse the existing build_model logic with the hyperparameters
        # Assuming num_features=1 for this problem
        self.build_model(
            num_features=1,
            hidden_layer_size=hidden_layer_size,
            num_attention_heads=num_attention_heads,
            dropout_rate=dropout_rate
        )
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss=self.quantile_loss)
        return self.model

    def get_best_model(self, num_features=1):
        """Builds the final TFT model using the best-found hyperparameters."""
        self.build_model(
            num_features=num_features,
            hidden_layer_size=64,
            num_attention_heads=2,
            dropout_rate=0.1
        )
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss=self.quantile_loss)
        return self.model

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

    def load(self, filepath, num_features=1, hidden_layer_size=64, num_attention_heads=2, dropout_rate=0.1):
        # Build the model first to have the correct architecture
        self.build_model(
            num_features=num_features, 
            hidden_layer_size=hidden_layer_size, 
            num_attention_heads=num_attention_heads,
            dropout_rate=dropout_rate
        )
        self.model.load_weights(filepath)