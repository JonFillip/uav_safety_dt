#!/usr/bin/env python3
"""
Predictive Autoencoder for UAV Safety Digital Twin

This module implements an enhanced version of the SUPERIALIST autoencoder that not only
reconstructs current states but also predicts future states. The architecture includes:
- Encoder: Convolutional layers that extract features from orientation data
- Temporal Predictor: LSTM/GRU layers that capture temporal dependencies and predict future states
- Decoder: Transposed convolutional layers that reconstruct both current and predicted future states
"""

import os
import sys
import numpy as np
import tensorflow as tf
from keras.models import Model, load_model, save_model
from keras.layers import (
    Input, Dense, LSTM, GRU, Conv1D, Conv1DTranspose, 
    BatchNormalization, Dropout, Flatten, Reshape, 
    MaxPooling1D, Concatenate, TimeDistributed, RepeatVector
)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
from datetime import datetime

# Add the project root to the path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the original SUPERIALIST model
try:
    from model import CNNModel
except ImportError as e:
    print(f"Error importing Superialist modules: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)


class PredictiveAutoencoder:
    """
    Enhanced autoencoder model that predicts future states for UAV safety analysis.
    
    This class extends the SUPERIALIST autoencoder with temporal prediction capabilities,
    allowing it to forecast future UAV states and safety metrics.
    """
    
    def __init__(self, window_size=25, prediction_horizon=5, model_type='lstm'):
        """
        Initialize the predictive autoencoder.
        
        Args:
            window_size: Size of the input window (default: 25)
            prediction_horizon: Number of future steps to predict (default: 5)
            model_type: Type of temporal model to use ('lstm', 'gru', or 'transformer')
        """
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.model_type = model_type
        self.model = None
        self.encoder = None
        self.temporal_predictor = None
        self.decoder = None
        self.history = None
        
    def build_model(self, input_shape=(25, 1), latent_dim=32, dropout_rate=0.2):
        """
        Build the predictive autoencoder model.
        
        Args:
            input_shape: Shape of the input data (window_size, features)
            latent_dim: Dimension of the latent space
            dropout_rate: Dropout rate for regularization
            
        Returns:
            The compiled model
        """
        # Input layer
        inputs = Input(shape=input_shape, name='encoder_input')
        
        # Encoder
        x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(dropout_rate)(x)
        
        x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(dropout_rate)(x)
        
        # Flatten and encode to latent space
        x = Flatten()(x)
        encoded = Dense(latent_dim, activation='relu', name='latent_encoding')(x)
        
        # Temporal predictor
        if self.model_type == 'lstm':
            # Create a sequence of latent vectors by repeating the encoded vector
            temporal_input = RepeatVector(self.prediction_horizon)(encoded)
            # Process the sequence with LSTM layers
            temporal_output = LSTM(latent_dim, return_sequences=True)(temporal_input)
            # Add more LSTM layers for future prediction
            temporal_output = LSTM(latent_dim * 2, return_sequences=True)(temporal_output)
            # Use return_sequences=True for the final LSTM layer to preserve temporal information
            temporal_output = LSTM(latent_dim * 2, return_sequences=True)(temporal_output)
            # Use TimeDistributed Dense to process each time step independently
            future_states = TimeDistributed(Dense(latent_dim))(temporal_output)
            # No need to reshape, already has the correct shape [batch, prediction_horizon, latent_dim]
            future_states_reshaped = future_states
        elif self.model_type == 'gru':
            # Create a sequence of latent vectors by repeating the encoded vector
            temporal_input = RepeatVector(self.prediction_horizon)(encoded)
            # Process the sequence with GRU layers
            temporal_output = GRU(latent_dim, return_sequences=True)(temporal_input)
            # Add more GRU layers for future prediction
            temporal_output = GRU(latent_dim * 2, return_sequences=True)(temporal_output)
            # Use return_sequences=True for the final GRU layer
            temporal_output = GRU(latent_dim * 2, return_sequences=True)(temporal_output)
            # Use TimeDistributed Dense to process each time step independently
            future_states = TimeDistributed(Dense(latent_dim))(temporal_output)
            # No need to reshape, already has the correct shape [batch, prediction_horizon, latent_dim]
            future_states_reshaped = future_states
        else:  # transformer or other future implementations
            # Placeholder for transformer implementation
            future_states = Dense(latent_dim * self.prediction_horizon)(encoded)
            # Reshape for decoder
            future_states_reshaped = Reshape((self.prediction_horizon, latent_dim))(future_states)
        
        # Decoder for current state reconstruction
        # Use a simpler approach with a dense layer and reshape
        current_decoder = Dense(input_shape[0] * input_shape[1], activation='linear')(encoded)
        current_decoded = Reshape(input_shape, name='current_reconstruction')(current_decoder)
        
        # Decoder for future state prediction
        future_decoder = TimeDistributed(Dense(input_shape[0] * input_shape[1]))(future_states_reshaped)
        future_decoded = Reshape((self.prediction_horizon, input_shape[0], input_shape[1]), name='future_decoded')(future_decoder)
        
        # Create the full model
        self.model = Model(inputs=inputs, outputs=[current_decoded, future_decoded])
        
        # Create encoder and decoder submodels for inference
        self.encoder = Model(inputs=inputs, outputs=encoded)
        
        # Temporal predictor model
        temporal_input_layer = Input(shape=(latent_dim,))
        if self.model_type == 'lstm':
            temp_repeat = RepeatVector(self.prediction_horizon)(temporal_input_layer)
            temp_lstm1 = LSTM(latent_dim, return_sequences=True)(temp_repeat)
            temp_lstm2 = LSTM(latent_dim * 2, return_sequences=True)(temp_lstm1)
            # Use return_sequences=True for the final LSTM layer
            temp_lstm3 = LSTM(latent_dim * 2, return_sequences=True)(temp_lstm2)
            # Use TimeDistributed Dense to process each time step independently
            temp_output = TimeDistributed(Dense(latent_dim))(temp_lstm3)
            # Flatten the output for the temporal predictor model
            temp_output = Flatten()(temp_output)
        elif self.model_type == 'gru':
            temp_repeat = RepeatVector(self.prediction_horizon)(temporal_input_layer)
            temp_gru1 = GRU(latent_dim, return_sequences=True)(temp_repeat)
            temp_gru2 = GRU(latent_dim * 2, return_sequences=True)(temp_gru1)
            # Use return_sequences=True for the final GRU layer
            temp_gru3 = GRU(latent_dim * 2, return_sequences=True)(temp_gru2)
            # Use TimeDistributed Dense to process each time step independently
            temp_output = TimeDistributed(Dense(latent_dim))(temp_gru3)
            # Flatten the output for the temporal predictor model
            temp_output = Flatten()(temp_output)
        else:
            temp_output = Dense(latent_dim * self.prediction_horizon)(temporal_input_layer)
        
        self.temporal_predictor = Model(inputs=temporal_input_layer, outputs=temp_output)
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'current_reconstruction': 'mse',
                'future_decoded': 'mse'
            },
            loss_weights={
                'current_reconstruction': 0.2,  # Give less weight to reconstruction
                'future_decoded': 0.8  # Give more weight to future prediction
            }
        )
        
        # Print model summary to debug output names
        self.model.summary()
        
        return self.model
    
    def fit(self, train_data, targets=None, validation_split=0.2, epochs=100, batch_size=32, callbacks=None):
        """
        Train the predictive autoencoder.
        
        Args:
            train_data: Training data for current state
            targets: Dictionary with targets for each output or future data for prediction targets
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: List of Keras callbacks
            
        Returns:
            Training history
        """
        if callbacks is None:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ModelCheckpoint(
                    filepath=f'models/predictive_autoencoder_{datetime.now().strftime("%Y%m%d_%H%M%S")}.keras',
                    monitor='val_loss',
                    save_best_only=True
                )
            ]
        
        # If targets is a dictionary, use it directly
        if isinstance(targets, dict):
            target_dict = targets
        # If targets is future data, create a dictionary with output names as keys
        elif targets is not None:
            target_dict = {
                'current_reconstruction': train_data,
                'future_decoded': targets
            }
        # If targets is None, use train_data for both outputs
        else:
            target_dict = {
                'current_reconstruction': train_data,
                'future_decoded': train_data  # This is a placeholder, not used in training
            }
        
        self.history = self.model.fit(
            train_data,
            target_dict,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, data, mc_dropout=False, n_samples=10):
        """
        Generate predictions for current reconstruction and future states.
        
        Args:
            data: Input data
            mc_dropout: Whether to use Monte Carlo dropout for uncertainty estimation
            n_samples: Number of Monte Carlo samples if mc_dropout is True
            
        Returns:
            Tuple of (current_reconstruction, future_predictions)
        """
        if not mc_dropout:
            return self.model.predict(data)
        
        # For Monte Carlo dropout, we need to create a custom function
        # that keeps dropout active during inference
        predictions = []
        
        # Create a tf.function that runs the model with dropout active
        @tf.function
        def predict_with_dropout(x):
            return self.model(x, training=True)
        
        # Run multiple forward passes with dropout active
        for _ in range(n_samples):
            pred = predict_with_dropout(data)
            predictions.append(pred)
        
        # Calculate mean and variance of predictions
        current_mean = np.mean([p[0] for p in predictions], axis=0)
        current_var = np.var([p[0] for p in predictions], axis=0)
        
        future_mean = np.mean([p[1] for p in predictions], axis=0)
        future_var = np.var([p[1] for p in predictions], axis=0)
        
        return (current_mean, future_mean), (current_var, future_var)
    
    def calculate_reconstruction_error(self, data, future_data=None):
        """
        Calculate reconstruction error for anomaly detection.
        
        Args:
            data: Input data
            future_data: Future data for prediction targets (optional)
            
        Returns:
            DataFrame with reconstruction errors
        """
        # Get predictions
        current_pred, future_pred = self.model.predict(data)
        
        # Calculate reconstruction error for current state
        current_error = np.mean(np.square(data - current_pred), axis=(1, 2))
        
        # Calculate prediction error for future states if future_data is provided
        if future_data is not None:
            future_error = np.mean(np.square(future_data - future_pred), axis=(1, 2, 3))
        else:
            future_error = np.zeros_like(current_error)
        
        # Create DataFrame with errors
        error_df = pd.DataFrame({
            'current_reconstruction_error': current_error,
            'future_prediction_error': future_error,
            'total_error': current_error + future_error
        })
        
        return error_df
    
    def save(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the full model
        self.model.save(filepath)
        
        # Save encoder and temporal predictor separately
        encoder_path = filepath.replace('.keras', '_encoder.keras')
        temporal_path = filepath.replace('.keras', '_temporal.keras')
        
        self.encoder.save(encoder_path)
        self.temporal_predictor.save(temporal_path)
    
    def load(self, filepath):
        """
        Load the model from a file.
        
        Args:
            filepath: Path to the saved model
        """
        # Load the full model
        self.model = load_model(filepath)
        
        # Load encoder and temporal predictor
        encoder_path = filepath.replace('.keras', '_encoder.keras')
        temporal_path = filepath.replace('.keras', '_temporal.keras')
        
        if os.path.exists(encoder_path):
            self.encoder = load_model(encoder_path)
        
        if os.path.exists(temporal_path):
            self.temporal_predictor = load_model(temporal_path)
    
    @classmethod
    def from_superialist(cls, superialist_model_path, prediction_horizon=5, model_type='lstm'):
        """
        Create a predictive autoencoder from an existing SUPERIALIST model.
        
        Args:
            superialist_model_path: Path to the SUPERIALIST model
            prediction_horizon: Number of future steps to predict
            model_type: Type of temporal model to use
            
        Returns:
            Initialized PredictiveAutoencoder with weights from SUPERIALIST
        """
        # Load the SUPERIALIST model
        superialist_model = CNNModel()
        superialist_model.load(superialist_model_path)
        
        # Create a new predictive autoencoder
        predictive_model = cls(
            window_size=CNNModel.WINSIZE,
            prediction_horizon=prediction_horizon,
            model_type=model_type
        )
        
        # Build the model with the same input shape
        input_shape = (CNNModel.WINSIZE, 1)  # Assuming 1 feature (r_zero)
        predictive_model.build_model(input_shape=input_shape)
        
        # TODO: Transfer weights from SUPERIALIST encoder to predictive encoder
        # This requires careful mapping of layer weights and may need manual adjustment
        
        return predictive_model


def prepare_sequence_data(df, input_col='r_zero', window_size=25, prediction_horizon=5):
    """
    Prepare sequence data for training the predictive autoencoder.
    
    Args:
        df: DataFrame with input data
        input_col: Column name for input data
        window_size: Size of the input window
        prediction_horizon: Number of future steps to predict
        
    Returns:
        Tuple of (current_data, future_data)
    """
    # Extract input sequences
    sequences = df[input_col].tolist()
    
    # Convert to numpy arrays
    X = np.array(sequences)
    
    # Reshape to [samples, timesteps, features]
    if len(X.shape) == 2:
        X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Create future sequences by shifting the data
    future_X = np.zeros((X.shape[0], prediction_horizon, X.shape[1], X.shape[2]))
    
    # Group by log to ensure we don't create sequences across different logs
    for log_folder, log_group in df.groupby(['log_folder', 'log_name']):
        indices = log_group.index.tolist()
        
        for i, idx in enumerate(indices[:-prediction_horizon]):
            for j in range(prediction_horizon):
                if i + j + 1 < len(indices):
                    future_idx = indices[i + j + 1]
                    future_X[idx, j] = X[future_idx]
    
    return X, future_X


if __name__ == '__main__':
    # Example usage
    print("Predictive Autoencoder for UAV Safety Digital Twin")
    print("This module is not meant to be run directly.")
    print("Use train_predictive_autoencoder.py to train the model.")
