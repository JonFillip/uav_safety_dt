#!/usr/bin/env python3
"""
Train Enhanced Predictive Autoencoder for UAV Safety Digital Twin

This script trains the enhanced predictive autoencoder model for the UAV Safety Digital Twin.
It includes improvements such as:
- Deeper network architecture
- Bidirectional LSTM/GRU layers
- Residual connections
- Attention mechanisms
- Advanced regularization
- Learning rate scheduling
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

# Add the project root to the path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the enhanced predictive autoencoder
from predictive_twin.enhanced_predictive_autoencoder import EnhancedPredictiveAutoencoder

# Import the fixed prepare_sequence_data function
from prepare_sequence_data_fixed import prepare_sequence_data_fixed

# Import the original SUPERIALIST model
try:
    from model import CNNModel
except ImportError as e:
    print(f"Error importing Superialist modules: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Enhanced Predictive Autoencoder for UAV Safety Digital Twin",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset",
        default="datasets/train_dataset_features.csv",
        help="Path to the dataset CSV file"
    )
    
    parser.add_argument(
        "--model-type",
        choices=["lstm", "gru", "transformer"],
        default="lstm",
        help="Type of temporal model to use"
    )
    
    parser.add_argument(
        "--future-steps",
        type=int,
        default=5,
        help="Number of future steps to predict"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation"
    )
    
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=1,
        help="Number of models in the ensemble (1 for single model)"
    )
    
    parser.add_argument(
        "--from-superialist",
        help="Path to existing SUPERIALIST model to initialize from"
    )
    
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the trained model"
    )
    
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to save the trained model"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize training progress and results"
    )
    
    parser.add_argument(
        "--use-bidirectional",
        action="store_true",
        default=True,
        help="Use bidirectional LSTM/GRU layers"
    )
    
    parser.add_argument(
        "--use-attention",
        action="store_true",
        default=True,
        help="Use attention mechanisms"
    )
    
    parser.add_argument(
        "--use-residual",
        action="store_true",
        default=True,
        help="Use residual connections"
    )
    
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=64,
        help="Dimension of the latent space"
    )
    
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.3,
        help="Dropout rate for regularization"
    )
    
    parser.add_argument(
        "--recurrent-dropout-rate",
        type=float,
        default=0.2,
        help="Dropout rate for recurrent connections"
    )
    
    parser.add_argument(
        "--l2-reg",
        type=float,
        default=0.001,
        help="L2 regularization factor"
    )
    
    return parser.parse_args()


def load_dataset(filepath):
    """
    Load a dataset with error handling.
    
    Args:
        filepath: Path to the dataset CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print(f"Loading dataset from {filepath}...")
    
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)


def preprocess_dataset(df):
    """
    Preprocess the dataset for training.
    
    Args:
        df: DataFrame with input data
        
    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    print("Preprocessing dataset...")
    
    # Check if 'r_zero' column exists
    if 'r_zero' not in df.columns and 'r' in df.columns:
        print("Creating 'r_zero' column from 'r' data...")
        
        # Handle the specific case we're seeing: concatenated float values without separators
        def parse_r_column(x):
            if pd.isna(x):
                return [0.0] * CNNModel.WINSIZE  # Return zeros instead of empty list
            
            if isinstance(x, str):
                # First try to extract using regex
                import re
                values = re.findall(r'-?\d+\.\d+', x)
                if values and len(values) == CNNModel.WINSIZE:
                    return [float(val) for val in values]
                
                # If that doesn't work or doesn't have the right length, 
                # try to extract a single value and repeat it
                try:
                    # Try to extract the first value (assuming it's repeated)
                    if len(x) >= 16:
                        first_val = float(x[:16])  # Assuming each value is ~16 chars
                    else:
                        first_val = float(x)
                    return [first_val] * CNNModel.WINSIZE
                except:
                    print(f"Warning: Could not parse r value: {x[:30]}...")
                    return [0.0] * CNNModel.WINSIZE  # Return zeros instead of empty list
            
            elif isinstance(x, float):
                # If it's a float, repeat it
                return [float(x)] * CNNModel.WINSIZE
            
            elif isinstance(x, list):
                # If it's already a list, ensure it has the right length
                if len(x) == CNNModel.WINSIZE:
                    return x
                elif len(x) > CNNModel.WINSIZE:
                    return x[:CNNModel.WINSIZE]  # Truncate if too long
                else:
                    # Pad with the last value if too short
                    last_val = x[-1] if x else 0.0
                    return x + [last_val] * (CNNModel.WINSIZE - len(x))
            
            else:
                # For any other type, try to convert to float and repeat
                try:
                    return [float(x)] * CNNModel.WINSIZE
                except:
                    print(f"Warning: Could not parse r value of type {type(x)}")
                    return [0.0] * CNNModel.WINSIZE  # Return zeros instead of empty list
        
        # Apply the parsing function to the r column
        df['r'] = df['r'].apply(parse_r_column)
        
        # Verify that all rows have the correct length
        invalid_rows = df['r'].apply(lambda x: len(x) != CNNModel.WINSIZE)
        if invalid_rows.any():
            print(f"Warning: {invalid_rows.sum()} rows have invalid length for 'r' column")
            # Filter out invalid rows
            df = df[~invalid_rows]
        
        # Create zero-centered version for non-empty arrays
        df['r_zero'] = df['r'].apply(
            lambda x: [val - sum(x) / len(x) for val in x] if len(x) > 0 else []
        )
        
        print("Successfully created 'r_zero' column")
    
    # Filter rows with exactly WINSIZE columns in the 'r_zero' value
    if 'r_zero' in df.columns:
        df = df[df['r_zero'].apply(lambda x: len(x) == CNNModel.WINSIZE if isinstance(x, list) else True)]
    
    # Drop rows with missing values
    df = df.dropna(subset=['r_zero'])
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df


def visualize_training(history, output_dir):
    """
    Visualize training progress.
    
    Args:
        history: Training history
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training & validation loss
    plt.figure(figsize=(15, 10))
    
    # Overall loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Component losses
    plt.subplot(2, 2, 2)
    plt.plot(history.history['current_reconstruction_loss'])
    plt.plot(history.history['future_decoded_loss'])
    plt.title('Component Losses')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Current Reconstruction', 'Future Prediction'], loc='upper right')
    
    # Learning rate if available
    if 'lr' in history.history:
        plt.subplot(2, 2, 3)
        plt.plot(history.history['lr'])
        plt.title('Learning Rate')
        plt.ylabel('Learning Rate')
        plt.xlabel('Epoch')
    
    # Validation component losses
    plt.subplot(2, 2, 4)
    plt.plot(history.history['val_current_reconstruction_loss'])
    plt.plot(history.history['val_future_decoded_loss'])
    plt.title('Validation Component Losses')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Current Reconstruction', 'Future Prediction'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()


def visualize_predictions(model, test_data, future_data, output_dir, n_samples=5):
    """
    Visualize model predictions.
    
    Args:
        model: Trained model
        test_data: Test data
        future_data: Future data
        output_dir: Directory to save visualizations
        n_samples: Number of samples to visualize
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictions
    current_pred, future_pred = model.predict(test_data[:n_samples])
    
    # Get uncertainty estimates using Monte Carlo dropout
    if hasattr(model, 'predict') and 'mc_dropout' in model.predict.__code__.co_varnames:
        try:
            (current_mean, future_mean), (current_var, future_var) = model.predict(
                test_data[:n_samples], mc_dropout=True, n_samples=20
            )
            has_uncertainty = True
        except:
            has_uncertainty = False
    else:
        has_uncertainty = False
    
    # Visualize current reconstruction and future predictions
    for i in range(n_samples):
        plt.figure(figsize=(15, 10))
        
        # Plot current reconstruction
        plt.subplot(2, 1, 1)
        plt.plot(test_data[i, :, 0], label='Original')
        plt.plot(current_pred[i, :, 0], label='Reconstructed')
        
        if has_uncertainty:
            # Add uncertainty bounds (2 standard deviations)
            std_dev = np.sqrt(current_var[i, :, 0])
            plt.fill_between(
                range(len(current_mean[i, :, 0])),
                current_mean[i, :, 0] - 2 * std_dev,
                current_mean[i, :, 0] + 2 * std_dev,
                alpha=0.2,
                color='orange',
                label='Uncertainty (±2σ)'
            )
        
        plt.title(f'Sample {i+1}: Current Reconstruction')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot future prediction
        plt.subplot(2, 1, 2)
        
        # Create a colormap for the future steps
        colors = plt.cm.viridis(np.linspace(0, 1, future_data.shape[1]))
        
        for j in range(future_data.shape[1]):
            plt.plot(future_data[i, j, :, 0], 
                     label=f'Future {j+1} (True)', 
                     color=colors[j])
            plt.plot(future_pred[i, j, :, 0], 
                     label=f'Future {j+1} (Pred)', 
                     linestyle='--',
                     color=colors[j])
            
            if has_uncertainty and j == 0:  # Only show uncertainty for first future step to avoid clutter
                # Add uncertainty bounds (2 standard deviations)
                std_dev = np.sqrt(future_var[i, j, :, 0])
                plt.fill_between(
                    range(len(future_mean[i, j, :, 0])),
                    future_mean[i, j, :, 0] - 2 * std_dev,
                    future_mean[i, j, :, 0] + 2 * std_dev,
                    alpha=0.2,
                    color=colors[j],
                    label=f'Uncertainty Future {j+1} (±2σ)'
                )
        
        plt.title(f'Sample {i+1}: Future Prediction')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'prediction_sample_{i+1}.png'))
        plt.close()
    
    # Create a summary visualization showing prediction error over time
    plt.figure(figsize=(15, 8))
    
    # Calculate mean squared error for each future step
    mse_by_step = []
    for j in range(future_data.shape[1]):
        mse = np.mean(np.square(future_data[:n_samples, j, :, 0] - future_pred[:n_samples, j, :, 0]))
        mse_by_step.append(mse)
    
    plt.bar(range(1, len(mse_by_step) + 1), mse_by_step)
    plt.xlabel('Future Step')
    plt.ylabel('Mean Squared Error')
    plt.title('Prediction Error by Future Step')
    plt.xticks(range(1, len(mse_by_step) + 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_error_by_step.png'))
    plt.close()


def train_ensemble(args, train_data, future_data):
    """
    Train an ensemble of enhanced predictive autoencoder models.
    
    Args:
        args: Command-line arguments
        train_data: Training data
        future_data: Future data
        
    Returns:
        List of trained models
    """
    models = []
    
    for i in range(args.ensemble_size):
        print(f"\nTraining model {i+1}/{args.ensemble_size}...")
        
        # Create model
        if args.from_superialist:
            model = EnhancedPredictiveAutoencoder.from_superialist(
                args.from_superialist,
                prediction_horizon=args.future_steps,
                model_type=args.model_type,
                use_bidirectional=args.use_bidirectional,
                use_attention=args.use_attention,
                use_residual=args.use_residual
            )
        else:
            model = EnhancedPredictiveAutoencoder(
                window_size=CNNModel.WINSIZE,
                prediction_horizon=args.future_steps,
                model_type=args.model_type,
                use_bidirectional=args.use_bidirectional,
                use_attention=args.use_attention,
                use_residual=args.use_residual
            )
            model.build_model(
                input_shape=(CNNModel.WINSIZE, 1),
                latent_dim=args.latent_dim,
                dropout_rate=args.dropout_rate,
                recurrent_dropout_rate=args.recurrent_dropout_rate,
                l2_reg=args.l2_reg
            )
        
        # Create callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ]
        
        # Add learning rate scheduler
        callbacks.append(
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        )
        
        if args.save_model:
            os.makedirs(args.output_dir, exist_ok=True)
            model_path = os.path.join(
                args.output_dir,
                f'enhanced_predictive_autoencoder_{args.model_type}_{i+1}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.keras'
            )
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
            f'enhanced_predictive_autoencoder_{args.model_type}_{i+1}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        callbacks.append(
            TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        )
        
        # Train model
        history = model.fit(
            train_data,
            future_data,
            validation_split=args.validation_split,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            use_lr_scheduler=True
        )
        
        # Visualize training
        if args.visualize:
            visualize_training(
                history,
                os.path.join(args.output_dir, f'enhanced_model_{i+1}_visualizations')
            )
        
        models.append(model)
    
    return models


def evaluate_ensemble(models, test_data, future_data, args):
    """
    Evaluate an ensemble of models.
    
    Args:
        models: List of trained models
        test_data: Test data
        future_data: Future data
        args: Command-line arguments
        
    Returns:
        DataFrame with evaluation results
    """
    print("\nEvaluating ensemble...")
    
    # Get predictions from each model
    all_current_preds = []
    all_future_preds = []
    
    for i, model in enumerate(models):
        print(f"Evaluating model {i+1}/{len(models)}...")
        current_pred, future_pred = model.predict(test_data)
        all_current_preds.append(current_pred)
        all_future_preds.append(future_pred)
    
    # Calculate mean and variance of predictions
    current_mean = np.mean(all_current_preds, axis=0)
    current_var = np.var(all_current_preds, axis=0)
    
    future_mean = np.mean(all_future_preds, axis=0)
    future_var = np.var(all_future_preds, axis=0)
    
    # Calculate reconstruction error
    current_error = np.mean(np.square(test_data - current_mean), axis=(1, 2))
    future_error = np.mean(np.square(future_data - future_mean), axis=(1, 2, 3))
    
    # Calculate uncertainty (variance of predictions)
    current_uncertainty = np.mean(current_var, axis=(1, 2))
    future_uncertainty = np.mean(future_var, axis=(1, 2, 3))
    
    # Create DataFrame with results
    results = pd.DataFrame({
        'current_reconstruction_error': current_error,
        'future_prediction_error': future_error,
        'total_error': current_error + future_error,
        'current_uncertainty': current_uncertainty,
        'future_uncertainty': future_uncertainty,
        'total_uncertainty': current_uncertainty + future_uncertainty
    })
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"  Current Reconstruction Error: {current_error.mean():.6f} ± {current_error.std():.6f}")
    print(f"  Future Prediction Error: {future_error.mean():.6f} ± {future_error.std():.6f}")
    print(f"  Total Error: {(current_error + future_error).mean():.6f} ± {(current_error + future_error).std():.6f}")
    print(f"  Current Uncertainty: {current_uncertainty.mean():.6f} ± {current_uncertainty.std():.6f}")
    print(f"  Future Uncertainty: {future_uncertainty.mean():.6f} ± {future_uncertainty.std():.6f}")
    print(f"  Total Uncertainty: {(current_uncertainty + future_uncertainty).mean():.6f} ± {(current_uncertainty + future_uncertainty).std():.6f}")
    
    # Calculate error by future step
    future_error_by_step = []
    for j in range(future_data.shape[1]):
        step_error = np.mean(np.square(future_data[:, j, :, :] - future_mean[:, j, :, :]))
        future_error_by_step.append(step_error)
    
    print("\nFuture Prediction Error by Step:")
    for j, error in enumerate(future_error_by_step):
        print(f"  Step {j+1}: {error:.6f}")
    
    # Visualize predictions
    if args.visualize:
        visualize_predictions(
            models[0],  # Use first model for visualization
            test_data,
            future_data,
            os.path.join(args.output_dir, 'enhanced_prediction_visualizations')
        )
    
    return results


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load dataset
    df = load_dataset(args.dataset)
    
    # Preprocess dataset
    df = preprocess_dataset(df)
    
    # Prepare sequence data
    print(f"Preparing sequence data with prediction horizon {args.future_steps}...")
    X, future_X = prepare_sequence_data_fixed(
        df,
        input_col='r_zero',
        window_size=CNNModel.WINSIZE,
        prediction_horizon=args.future_steps
    )
    
    # Split data into training and testing sets
    train_size = int(len(X) * (1 - args.validation_split))
    X_train, X_test = X[:train_size], X[train_size:]
    future_X_train, future_X_test = future_X[:train_size], future_X[train_size:]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Future training data shape: {future_X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Future testing data shape: {future_X_test.shape}")
    
    # Train ensemble
    models = train_ensemble(args, X_train, future_X_train)
    
    # Evaluate ensemble
    results = evaluate_ensemble(models, X_test, future_X_test, args)
    
    # Save results
    if args.save_model:
        results_path = os.path.join(args.output_dir, 'enhanced_evaluation_results.csv')
        results.to_csv(results_path, index=False)
        print(f"Saved evaluation results to {results_path}")
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
