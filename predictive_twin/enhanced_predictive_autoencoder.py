#!/usr/bin/env python3
"""
Enhanced Predictive Autoencoder for UAV Safety Digital Twin

This module implements an improved version of the predictive autoencoder with:
- Deeper LSTM/GRU architecture with more units
- Bidirectional LSTM/GRU layers for better context
- Residual connections between layers
- Attention mechanisms for better temporal modeling
- Advanced regularization techniques
- Learning rate scheduling and gradient clipping
"""

import os
import sys
import numpy as np
import tensorflow as tf
from keras.models import Model, load_model, save_model
from keras.layers import (
    Input, Dense, LSTM, GRU, Conv1D, Conv1DTranspose, 
    BatchNormalization, Dropout, Flatten, Reshape, 
    MaxPooling1D, Concatenate, TimeDistributed, RepeatVector,
    Bidirectional, Add, LayerNormalization
)
from keras.optimizers import Adam
from keras.callbacks import (
    EarlyStopping, ModelCheckpoint, TensorBoard, 
    ReduceLROnPlateau, LearningRateScheduler
)
from keras.regularizers import l2
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


# Custom Attention layer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        # Alignment scores. Shape: (batch_size, seq_len, 1)
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        
        # Remove last dim: (batch_size, seq_len)
        e = tf.keras.backend.squeeze(e, axis=-1)
        
        # Compute the weights
        alpha = tf.keras.backend.softmax(e)
        
        # Reshape to (batch_size, seq_len, 1)
        alpha = tf.keras.backend.expand_dims(alpha, axis=-1)
        
        # Compute the context vector
        context = x * alpha
        context = tf.keras.backend.sum(context, axis=1)
        
        return context


class EnhancedPredictiveAutoencoder:
    """
    Enhanced autoencoder model that predicts future states for UAV safety analysis.
    
    This class extends the original PredictiveAutoencoder with advanced features:
    - Deeper network architecture
    - Bidirectional LSTM/GRU layers
    - Residual connections
    - Attention mechanisms
    - Advanced regularization
    """
    
    def __init__(self, window_size=25, prediction_horizon=5, model_type='lstm', 
                 use_bidirectional=True, use_attention=True, use_residual=True):
        """
        Initialize the enhanced predictive autoencoder.
        
        Args:
            window_size: Size of the input window (default: 25)
            prediction_horizon: Number of future steps to predict (default: 5)
            model_type: Type of temporal model to use ('lstm', 'gru', or 'transformer')
            use_bidirectional: Whether to use bidirectional LSTM/GRU layers
            use_attention: Whether to use attention mechanisms
            use_residual: Whether to use residual connections
        """
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.model_type = model_type
        self.use_bidirectional = use_bidirectional
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.model = None
        self.encoder = None
        self.temporal_predictor = None
        self.decoder = None
        self.history = None
        
    def build_model(self, input_shape=(25, 1), latent_dim=64, dropout_rate=0.3, 
                    recurrent_dropout_rate=0.2, l2_reg=0.001):
        """
        Build the enhanced predictive autoencoder model.
        
        Args:
            input_shape: Shape of the input data (window_size, features)
            latent_dim: Dimension of the latent space
            dropout_rate: Dropout rate for regularization
            recurrent_dropout_rate: Dropout rate for recurrent connections
            l2_reg: L2 regularization factor
            
        Returns:
            The compiled model
        """
        # Input layer
        inputs = Input(shape=input_shape, name='encoder_input')
        
        # Enhanced Encoder with deeper CNN
        x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', 
                  kernel_regularizer=l2(l2_reg))(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(dropout_rate)(x)
        
        x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu',
                  kernel_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(dropout_rate)(x)
        
        # Add another convolutional layer for deeper feature extraction
        x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu',
                  kernel_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Flatten and encode to latent space
        x = Flatten()(x)
        x = Dense(latent_dim * 2, activation='relu', kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(dropout_rate)(x)
        encoded = Dense(latent_dim, activation='relu', name='latent_encoding', 
                       kernel_regularizer=l2(l2_reg))(x)
        
        # Enhanced Temporal predictor with deeper architecture
        if self.model_type == 'lstm':
            # Create a sequence of latent vectors by repeating the encoded vector
            temporal_input = RepeatVector(self.prediction_horizon)(encoded)
            
            # First LSTM layer
            if self.use_bidirectional:
                lstm1 = Bidirectional(LSTM(latent_dim, return_sequences=True, 
                                          recurrent_dropout=recurrent_dropout_rate,
                                          kernel_regularizer=l2(l2_reg)))(temporal_input)
            else:
                lstm1 = LSTM(latent_dim, return_sequences=True, 
                            recurrent_dropout=recurrent_dropout_rate,
                            kernel_regularizer=l2(l2_reg))(temporal_input)
            
            lstm1 = LayerNormalization()(lstm1)
            lstm1_dropout = Dropout(dropout_rate)(lstm1)
            
            # Second LSTM layer with residual connection
            if self.use_bidirectional:
                lstm2 = Bidirectional(LSTM(latent_dim * 2, return_sequences=True,
                                          recurrent_dropout=recurrent_dropout_rate,
                                          kernel_regularizer=l2(l2_reg)))(lstm1_dropout)
            else:
                lstm2 = LSTM(latent_dim * 2, return_sequences=True,
                            recurrent_dropout=recurrent_dropout_rate,
                            kernel_regularizer=l2(l2_reg))(lstm1_dropout)
            
            lstm2 = LayerNormalization()(lstm2)
            lstm2_dropout = Dropout(dropout_rate)(lstm2)
            
            # Third LSTM layer with residual connection
            if self.use_bidirectional:
                lstm3 = Bidirectional(LSTM(latent_dim * 2, return_sequences=True,
                                          recurrent_dropout=recurrent_dropout_rate,
                                          kernel_regularizer=l2(l2_reg)))(lstm2_dropout)
            else:
                lstm3 = LSTM(latent_dim * 2, return_sequences=True,
                            recurrent_dropout=recurrent_dropout_rate,
                            kernel_regularizer=l2(l2_reg))(lstm2_dropout)
            
            lstm3 = LayerNormalization()(lstm3)
            lstm3_dropout = Dropout(dropout_rate)(lstm3)
            
            # Apply attention mechanism if enabled
            if self.use_attention:
                # Self-attention on the sequence
                attention_output = AttentionLayer()(lstm3_dropout)
                # Reshape attention output to match sequence shape for concatenation
                attention_repeated = RepeatVector(self.prediction_horizon)(attention_output)
                # Combine attention with original sequence
                temporal_output = Concatenate()([lstm3_dropout, attention_repeated])
                temporal_output = TimeDistributed(Dense(latent_dim * 2, activation='relu',
                                                      kernel_regularizer=l2(l2_reg)))(temporal_output)
            else:
                temporal_output = lstm3_dropout
            
            # Add residual connection if enabled
            if self.use_residual and not self.use_bidirectional:
                # Only add residual if dimensions match
                temporal_output = Add()([temporal_output, lstm1])
            
            # Use TimeDistributed Dense to process each time step independently
            future_states = TimeDistributed(Dense(latent_dim, kernel_regularizer=l2(l2_reg)))(temporal_output)
            # No need to reshape, already has the correct shape [batch, prediction_horizon, latent_dim]
            future_states_reshaped = future_states
            
        elif self.model_type == 'gru':
            # Create a sequence of latent vectors by repeating the encoded vector
            temporal_input = RepeatVector(self.prediction_horizon)(encoded)
            
            # First GRU layer
            if self.use_bidirectional:
                gru1 = Bidirectional(GRU(latent_dim, return_sequences=True, 
                                        recurrent_dropout=recurrent_dropout_rate,
                                        kernel_regularizer=l2(l2_reg)))(temporal_input)
            else:
                gru1 = GRU(latent_dim, return_sequences=True, 
                          recurrent_dropout=recurrent_dropout_rate,
                          kernel_regularizer=l2(l2_reg))(temporal_input)
            
            gru1 = LayerNormalization()(gru1)
            gru1_dropout = Dropout(dropout_rate)(gru1)
            
            # Second GRU layer with residual connection
            if self.use_bidirectional:
                gru2 = Bidirectional(GRU(latent_dim * 2, return_sequences=True,
                                        recurrent_dropout=recurrent_dropout_rate,
                                        kernel_regularizer=l2(l2_reg)))(gru1_dropout)
            else:
                gru2 = GRU(latent_dim * 2, return_sequences=True,
                          recurrent_dropout=recurrent_dropout_rate,
                          kernel_regularizer=l2(l2_reg))(gru1_dropout)
            
            gru2 = LayerNormalization()(gru2)
            gru2_dropout = Dropout(dropout_rate)(gru2)
            
            # Third GRU layer with residual connection
            if self.use_bidirectional:
                gru3 = Bidirectional(GRU(latent_dim * 2, return_sequences=True,
                                        recurrent_dropout=recurrent_dropout_rate,
                                        kernel_regularizer=l2(l2_reg)))(gru2_dropout)
            else:
                gru3 = GRU(latent_dim * 2, return_sequences=True,
                          recurrent_dropout=recurrent_dropout_rate,
                          kernel_regularizer=l2(l2_reg))(gru2_dropout)
            
            gru3 = LayerNormalization()(gru3)
            gru3_dropout = Dropout(dropout_rate)(gru3)
            
            # Apply attention mechanism if enabled
            if self.use_attention:
                # Self-attention on the sequence
                attention_output = AttentionLayer()(gru3_dropout)
                # Reshape attention output to match sequence shape for concatenation
                attention_repeated = RepeatVector(self.prediction_horizon)(attention_output)
                # Combine attention with original sequence
                temporal_output = Concatenate()([gru3_dropout, attention_repeated])
                temporal_output = TimeDistributed(Dense(latent_dim * 2, activation='relu',
                                                      kernel_regularizer=l2(l2_reg)))(temporal_output)
            else:
                temporal_output = gru3_dropout
            
            # Add residual connection if enabled
            if self.use_residual and not self.use_bidirectional:
                # Only add residual if dimensions match
                temporal_output = Add()([temporal_output, gru1])
            
            # Use TimeDistributed Dense to process each time step independently
            future_states = TimeDistributed(Dense(latent_dim, kernel_regularizer=l2(l2_reg)))(temporal_output)
            # No need to reshape, already has the correct shape [batch, prediction_horizon, latent_dim]
            future_states_reshaped = future_states
            
        else:  # transformer or other future implementations
            # Placeholder for transformer implementation
            future_states = Dense(latent_dim * self.prediction_horizon, 
                                 kernel_regularizer=l2(l2_reg))(encoded)
            # Reshape for decoder
            future_states_reshaped = Reshape((self.prediction_horizon, latent_dim))(future_states)
        
        # Enhanced Decoder for current state reconstruction
        current_decoder = Dense(input_shape[0] * input_shape[1] // 2, activation='relu',
                               kernel_regularizer=l2(l2_reg))(encoded)
        current_decoder = Dense(input_shape[0] * input_shape[1], activation='linear',
                               kernel_regularizer=l2(l2_reg))(current_decoder)
        current_decoded = Reshape(input_shape, name='current_reconstruction')(current_decoder)
        
        # Enhanced Decoder for future state prediction
        future_decoder = TimeDistributed(Dense(input_shape[0] * input_shape[1] // 2, 
                                             activation='relu',
                                             kernel_regularizer=l2(l2_reg)))(future_states_reshaped)
        future_decoder = TimeDistributed(Dense(input_shape[0] * input_shape[1], 
                                             activation='linear',
                                             kernel_regularizer=l2(l2_reg)))(future_decoder)
        future_decoded = Reshape((self.prediction_horizon, input_shape[0], input_shape[1]), 
                                name='future_decoded')(future_decoder)
        
        # Create the full model
        self.model = Model(inputs=inputs, outputs=[current_decoded, future_decoded])
        
        # Create encoder and decoder submodels for inference
        self.encoder = Model(inputs=inputs, outputs=encoded)
        
        # Temporal predictor model (simplified for inference)
        temporal_input_layer = Input(shape=(latent_dim,))
        temp_repeat = RepeatVector(self.prediction_horizon)(temporal_input_layer)
        
        if self.model_type == 'lstm':
            if self.use_bidirectional:
                temp_lstm1 = Bidirectional(LSTM(latent_dim, return_sequences=True))(temp_repeat)
                temp_lstm2 = Bidirectional(LSTM(latent_dim * 2, return_sequences=True))(temp_lstm1)
                temp_lstm3 = Bidirectional(LSTM(latent_dim * 2, return_sequences=True))(temp_lstm2)
            else:
                temp_lstm1 = LSTM(latent_dim, return_sequences=True)(temp_repeat)
                temp_lstm2 = LSTM(latent_dim * 2, return_sequences=True)(temp_lstm1)
                temp_lstm3 = LSTM(latent_dim * 2, return_sequences=True)(temp_lstm2)
            
            if self.use_attention:
                temp_attention = AttentionLayer()(temp_lstm3)
                temp_attention_repeated = RepeatVector(self.prediction_horizon)(temp_attention)
                temp_combined = Concatenate()([temp_lstm3, temp_attention_repeated])
                temp_output = TimeDistributed(Dense(latent_dim * 2, activation='relu'))(temp_combined)
            else:
                temp_output = temp_lstm3
                
            temp_output = TimeDistributed(Dense(latent_dim))(temp_output)
            temp_output = Flatten()(temp_output)
            
        elif self.model_type == 'gru':
            if self.use_bidirectional:
                temp_gru1 = Bidirectional(GRU(latent_dim, return_sequences=True))(temp_repeat)
                temp_gru2 = Bidirectional(GRU(latent_dim * 2, return_sequences=True))(temp_gru1)
                temp_gru3 = Bidirectional(GRU(latent_dim * 2, return_sequences=True))(temp_gru2)
            else:
                temp_gru1 = GRU(latent_dim, return_sequences=True)(temp_repeat)
                temp_gru2 = GRU(latent_dim * 2, return_sequences=True)(temp_gru1)
                temp_gru3 = GRU(latent_dim * 2, return_sequences=True)(temp_gru2)
            
            if self.use_attention:
                temp_attention = AttentionLayer()(temp_gru3)
                temp_attention_repeated = RepeatVector(self.prediction_horizon)(temp_attention)
                temp_combined = Concatenate()([temp_gru3, temp_attention_repeated])
                temp_output = TimeDistributed(Dense(latent_dim * 2, activation='relu'))(temp_combined)
            else:
                temp_output = temp_gru3
                
            temp_output = TimeDistributed(Dense(latent_dim))(temp_output)
            temp_output = Flatten()(temp_output)
            
        else:
            temp_output = Dense(latent_dim * self.prediction_horizon)(temporal_input_layer)
        
        self.temporal_predictor = Model(inputs=temporal_input_layer, outputs=temp_output)
        
        # Compile the model with gradient clipping
        optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
        
        self.model.compile(
            optimizer=optimizer,
            loss={
                'current_reconstruction': 'mse',
                'future_decoded': 'mse'
            },
            loss_weights={
                'current_reconstruction': 0.2,  # Give less weight to reconstruction
                'future_decoded': 0.8  # Give more weight to future prediction
            }
        )
        
        # Print model summary
        self.model.summary()
        
        return self.model
    
    def fit(self, train_data, targets=None, validation_split=0.2, epochs=100, 
            batch_size=32, callbacks=None, use_lr_scheduler=True):
        """
        Train the enhanced predictive autoencoder.
        
        Args:
            train_data: Training data for current state
            targets: Dictionary with targets for each output or future data for prediction targets
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: List of Keras callbacks
            use_lr_scheduler: Whether to use learning rate scheduling
            
        Returns:
            Training history
        """
        if callbacks is None:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ]
        
        # Add learning rate scheduler if enabled
        if use_lr_scheduler:
            callbacks.append(
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.00001,
                    verbose=1
                )
            )
        
        # Add model checkpoint
        model_path = f'models/enhanced_predictive_autoencoder_{self.model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.keras'
        callbacks.append(
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        )
        
        # Add TensorBoard callback
        log_dir = os.path.join(
            'logs',
            f'enhanced_predictive_autoencoder_{self.model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        callbacks.append(
            TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        )
        
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
            Tuple of (current_reconstruction, future_predictions) or
            Tuple of ((current_mean, future_mean), (current_var, future_var)) if mc_dropout is True
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
        self.model = load_model(filepath, custom_objects={'AttentionLayer': AttentionLayer})
        
        # Load encoder and temporal predictor
        encoder_path = filepath.replace('.keras', '_encoder.keras')
        temporal_path = filepath.replace('.keras', '_temporal.keras')
        
        if os.path.exists(encoder_path):
            self.encoder = load_model(encoder_path)
        
        if os.path.exists(temporal_path):
            self.temporal_predictor = load_model(temporal_path, custom_objects={'AttentionLayer': AttentionLayer})
    
    @classmethod
    def from_superialist(cls, superialist_model_path, prediction_horizon=5, model_type='lstm',
                        use_bidirectional=True, use_attention=True, use_residual=True):
        """
        Create an enhanced predictive autoencoder from an existing SUPERIALIST model.
        
        Args:
            superialist_model_path: Path to the SUPERIALIST model
            prediction_horizon: Number of future steps to predict
            model_type: Type of temporal model to use
            use_bidirectional: Whether to use bidirectional LSTM/GRU layers
            use_attention: Whether to use attention mechanisms
            use_residual: Whether to use residual connections
            
        Returns:
            Initialized EnhancedPredictiveAutoencoder with weights from SUPERIALIST
        """
        # Load the SUPERIALIST model
        superialist_model = CNNModel()
        superialist_model.load(superialist_model_path)
        
        # Create a new enhanced predictive autoencoder
        predictive_model = cls(
            window_size=CNNModel.WINSIZE,
            prediction_horizon=prediction_horizon,
            model_type=model_type,
            use_bidirectional=use_bidirectional,
            use_attention=use_attention,
            use_residual=use_residual
        )
        
        # Build the model with the same input shape
        input_shape = (CNNModel.WINSIZE, 1)  # Assuming 1 feature (r_zero)
        predictive_model.build_model(input_shape=input_shape)
        
        # TODO: Transfer weights from SUPERIALIST encoder to predictive encoder
        # This requires careful mapping of layer weights and may need manual adjustment
        
        return predictive_model


if __name__ == '__main__':
    # Example usage
    print("Enhanced Predictive Autoencoder for UAV Safety Digital Twin")
    print("This module is not meant to be run directly.")
    print("Use train_enhanced_predictive_autoencoder.py to train the model.")
