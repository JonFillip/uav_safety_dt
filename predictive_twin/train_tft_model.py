#!/usr/bin/env python3
"""
Train Temporal Fusion Transformer (TFT) for UAV Safety Analysis

This script trains the Temporal Fusion Transformer model for UAV safety analysis
and compares its performance with the original SUPERIALIST model.

Key features:
- Trains a TFT model on UAV orientation data
- Provides quantile forecasts with uncertainty estimates
- Compares performance with the SUPERIALIST model
- Visualizes predictions and feature importance
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from keras.models import Model

# Add the project root to the path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the TFT model
from temporal_fusion_transformer import TemporalFusionTransformer

# Import the original SUPERIALIST model for comparison
try:
    from model import CNNModel
    from superialist.data_analysis import stats
except ImportError as e:
    print(f"Error importing Superialist modules: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)

# Import the fixed prepare_sequence_data function
from prepare_sequence_data_fixed import prepare_sequence_data_fixed


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Temporal Fusion Transformer for UAV Safety Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset",
        default="datasets/train_dataset_features.csv",
        help="Path to the dataset CSV file"
    )
    
    parser.add_argument(
        "--test-dataset",
        default="datasets/test1_dataset_features.csv",
        help="Path to the test dataset CSV file"
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
        "--hidden-size",
        type=int,
        default=64,
        help="Size of hidden layers"
    )
    
    parser.add_argument(
        "--num-lstm-layers",
        type=int,
        default=2,
        help="Number of LSTM layers"
    )
    
    parser.add_argument(
        "--num-attention-heads",
        type=int,
        default=4,
        help="Number of attention heads"
    )
    
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.1,
        help="Dropout rate for regularization"
    )
    
    parser.add_argument(
        "--quantiles",
        type=str,
        default="0.1,0.5,0.9",
        help="Comma-separated list of quantiles for uncertainty estimation"
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
        "--compare-superialist",
        action="store_true",
        help="Compare with the SUPERIALIST model"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=-0.3,
        help="Threshold for anomaly detection (for SUPERIALIST comparison)"
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
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('TFT Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tft_training_history.png'))
    plt.close()


def visualize_predictions(model, test_data, future_data, output_dir, n_samples=5):
    """
    Visualize model predictions with uncertainty intervals.
    
    Args:
        model: Trained TFT model
        test_data: Test data
        future_data: Future data
        output_dir: Directory to save visualizations
        n_samples: Number of samples to visualize
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictions (quantiles)
    predictions = model.predict(test_data[:n_samples])
    
    # Extract quantile predictions
    median_idx = model.quantiles.index(0.5) if 0.5 in model.quantiles else len(model.quantiles) // 2
    lower_idx = 0  # Lowest quantile (e.g., 0.1)
    upper_idx = -1  # Highest quantile (e.g., 0.9)
    
    # Visualize predictions for each sample
    for i in range(n_samples):
        plt.figure(figsize=(15, 10))
        
        # Plot past data
        plt.subplot(2, 1, 1)
        plt.plot(range(-test_data.shape[1], 0), test_data[i, :, 0], 
                label='Past', color='blue', marker='o', markersize=4)
        
        # Plot future data and predictions
        plt.subplot(2, 1, 2)
        
        # Extract actual future values
        actual_future = future_data[i, :, 0, 0]
        
        # Plot actual future values
        plt.plot(range(future_data.shape[1]), actual_future, 
                label='Actual Future', color='blue', marker='o', markersize=4)
        
        # Plot median predictions
        median_pred = predictions[i, :, median_idx]
        plt.plot(range(future_data.shape[1]), median_pred, 
                label='Median Prediction', color='red', linestyle='--', marker='x', markersize=4)
        
        # Plot prediction intervals
        lower_pred = predictions[i, :, lower_idx]
        upper_pred = predictions[i, :, upper_idx]
        plt.fill_between(range(future_data.shape[1]), lower_pred, upper_pred, 
                        color='red', alpha=0.2, label=f'Prediction Interval ({model.quantiles[lower_idx]}-{model.quantiles[upper_idx]})')
        
        plt.title(f'Sample {i+1}: Future Prediction with Uncertainty')
        plt.xlabel('Future Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'tft_prediction_sample_{i+1}.png'))
        plt.close()
    
    # Create a summary visualization showing prediction error over time
    plt.figure(figsize=(12, 6))
    
    # Calculate mean squared error for each future step
    mse_by_step = []
    for j in range(future_data.shape[1]):
        actual = future_data[:n_samples, j, 0, 0]
        pred = predictions[:n_samples, j, median_idx]
        mse = np.mean(np.square(actual - pred))
        mse_by_step.append(mse)
    
    plt.bar(range(1, len(mse_by_step) + 1), mse_by_step)
    plt.xlabel('Future Step')
    plt.ylabel('Mean Squared Error')
    plt.title('TFT Prediction Error by Future Step')
    plt.xticks(range(1, len(mse_by_step) + 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tft_prediction_error_by_step.png'))
    plt.close()


def compute_feature_importance(model, data, batch_size=32):
    """
    Compute feature importance by running a small batch of data through the model.
    
    Args:
        model: Trained TFT model
        data: Input data
        batch_size: Batch size for prediction
        
    Returns:
        Feature importance as a numpy array
    """
    # Create a custom model that outputs the feature importance
    if hasattr(model, 'feature_importance_tensor') and model.feature_importance_tensor is not None:
        try:
            # Create a small batch of data
            small_batch = data[:batch_size]
            
            # Create a model that outputs the feature importance
            inputs = model.model.inputs
            outputs = model.feature_importance_tensor
            importance_model = Model(inputs=inputs, outputs=outputs)
            
            # Run the model to get feature importance
            importance = importance_model.predict(small_batch)
            
            # Average across batch dimension
            importance = np.mean(importance, axis=0)
            
            # Store the importance
            model.feature_importance = importance
            
            return importance
        except Exception as e:
            print(f"Could not compute feature importance: {e}")
            return None
    else:
        print("Feature importance tensor not available")
        return None


def visualize_feature_importance(model, output_dir):
    """
    Visualize feature importance from the TFT model.
    
    Args:
        model: Trained TFT model
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if feature importance is available
    if model.feature_importance is None:
        print("Feature importance not available")
        return
    
    try:
        # Get feature importance
        importance = model.feature_importance
        
        # Create feature names (time steps)
        feature_names = [f'T-{i}' for i in range(importance.shape[0], 0, -1)]
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        plt.bar(feature_names, importance)
        plt.xlabel('Time Step')
        plt.ylabel('Importance')
        plt.title('TFT Feature Importance by Time Step')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tft_feature_importance.png'))
        plt.close()
    except Exception as e:
        print(f"Could not visualize feature importance: {e}")
        print("Skipping feature importance visualization")


def train_tft_model(args, train_data, future_data):
    """
    Train the Temporal Fusion Transformer model.
    
    Args:
        args: Command-line arguments
        train_data: Training data
        future_data: Future data
        
    Returns:
        Trained TFT model
    """
    print("\nTraining Temporal Fusion Transformer model...")
    
    # Parse quantiles
    quantiles = [float(q) for q in args.quantiles.split(',')]
    
    # Create model
    model = TemporalFusionTransformer(
        window_size=CNNModel.WINSIZE,
        prediction_horizon=args.future_steps,
        quantiles=quantiles
    )
    
    # Build model
    model.build_model(
        input_shape=(CNNModel.WINSIZE, 1),
        hidden_layer_size=args.hidden_size,
        num_lstm_layers=args.num_lstm_layers,
        num_attention_heads=args.num_attention_heads,
        dropout_rate=args.dropout_rate
    )
    
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    if args.save_model:
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(
            args.output_dir,
            f'tft_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.keras'
        )
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        )
    
    # Add TensorBoard callback
    log_dir = os.path.join(
        'logs',
        f'tft_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    callbacks.append(
        tf.keras.callbacks.TensorBoard(
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
        callbacks=callbacks
    )
    
    # Visualize training
    if args.visualize:
        visualize_training(
            history,
            os.path.join(args.output_dir, 'tft_visualizations')
        )
    
    return model


def evaluate_tft_model(model, test_data, future_data, args):
    """
    Evaluate the Temporal Fusion Transformer model.
    
    Args:
        model: Trained TFT model
        test_data: Test data
        future_data: Future data
        args: Command-line arguments
        
    Returns:
        DataFrame with evaluation results
    """
    print("\nEvaluating TFT model...")
    
    # Calculate reconstruction error
    results = model.calculate_reconstruction_error(test_data, future_data)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"  Prediction Error: {results['prediction_error'].mean():.6f} ± {results['prediction_error'].std():.6f}")
    print(f"  Interval Coverage: {results['interval_coverage'].mean():.6f} ± {results['interval_coverage'].std():.6f}")
    print(f"  Interval Width: {results['interval_width'].mean():.6f} ± {results['interval_width'].std():.6f}")
    print(f"  Anomaly Score: {results['anomaly_score'].mean():.6f} ± {results['anomaly_score'].std():.6f}")
    
    # Visualize predictions
    if args.visualize:
        visualize_predictions(
            model,
            test_data,
            future_data,
            os.path.join(args.output_dir, 'tft_visualizations')
        )
        
        # Visualize feature importance
        visualize_feature_importance(
            model,
            os.path.join(args.output_dir, 'tft_visualizations')
        )
    
    return results


def load_superialist_model():
    """
    Load the pre-trained SUPERIALIST model.
    
    Returns:
        Tuple of (model, success)
    """
    print("Loading pre-trained SUPERIALIST model...")
    model = CNNModel()
    
    try:
        model_path = os.path.join("superialist", "models", "autoencoder_1.keras")
        model.load(model_path)
        print(f"Successfully loaded pre-trained SUPERIALIST model from {model_path}")
        return model, True
    except Exception as e:
        print(f"Error loading pre-trained SUPERIALIST model: {e}")
        return model, False


def evaluate_superialist_model(model, dataset, threshold):
    """
    Evaluate the SUPERIALIST model on the specified dataset.
    
    Args:
        model: SUPERIALIST model
        dataset: Dataset to evaluate
        threshold: Threshold for anomaly detection
        
    Returns:
        DataFrame with evaluation results
    """
    print(f"Evaluating SUPERIALIST model with threshold {threshold}...")
    
    try:
        # Make predictions
        predictions = model.predict_encoder(dataset, inputs=["r_zero"])
        
        # Analyze results
        print("\n=== SUPERIALIST Safety Analysis Results ===")
        
        # Check if 'unsafe' column exists for safety analysis
        if 'unsafe' in predictions.columns:
            safety_results = stats(
                predictions,
                flag="mean_loss_4",
                flag_threshold=threshold,
                label="unsafe",
                label_threshold=None,
                levels=["log"]
            )
            
            # Calculate metrics
            tp = len(safety_results["TP"])
            fp = len(safety_results["FP"])
            tn = len(safety_results["TN"])
            fn = len(safety_results["FN"])
            
            # Print summary
            print(f"\nSafety Detection Summary:")
            print(f"  True Positives: {tp} (correctly identified unsafe flights)")
            print(f"  False Positives: {fp} (incorrectly flagged as unsafe)")
            print(f"  True Negatives: {tn} (correctly identified safe flights)")
            print(f"  False Negatives: {fn} (missed unsafe flights)")
            
            # Calculate precision, recall, and F1 score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
        
        # Basic statistics on reconstruction loss
        print("\nReconstruction Loss Statistics:")
        loss_stats = predictions["mean_loss_4"].describe()
        print(f"  Min: {loss_stats['min']:.4f}")
        print(f"  Max: {loss_stats['max']:.4f}")
        print(f"  Mean: {loss_stats['mean']:.4f}")
        print(f"  Std Dev: {loss_stats['std']:.4f}")
        
        # Count anomalies
        anomaly_count = (predictions["mean_loss_4"] <= threshold).sum()
        total_count = len(predictions["mean_loss_4"].unique())
        print(f"\nAnomaly Detection:")
        print(f"  {anomaly_count} out of {total_count} unique windows flagged as anomalous ({anomaly_count/total_count*100:.2f}%)")
        
        return predictions
    
    except Exception as e:
        print(f"Error evaluating SUPERIALIST model: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_models(tft_results, superialist_results, args):
    """
    Compare TFT and SUPERIALIST models.
    
    Args:
        tft_results: TFT evaluation results
        superialist_results: SUPERIALIST evaluation results
        args: Command-line arguments
    """
    print("\n=== Model Comparison ===")
    
    # Check if both results are available
    if tft_results is None or superialist_results is None:
        print("Cannot compare models: results not available")
        return
    
    # Create output directory if it doesn't exist
    comparison_dir = os.path.join(args.output_dir, 'model_comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Compare anomaly detection
    try:
        # Get TFT anomaly scores
        tft_anomaly_scores = tft_results['anomaly_score']
        
        # Get SUPERIALIST anomaly scores (mean_loss_4)
        superialist_anomaly_scores = superialist_results['mean_loss_4']
        
        # Normalize scores for fair comparison
        tft_normalized = (tft_anomaly_scores - tft_anomaly_scores.mean()) / tft_anomaly_scores.std()
        superialist_normalized = (superialist_anomaly_scores - superialist_anomaly_scores.mean()) / superialist_anomaly_scores.std()
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        
        # Create a scatter plot of normalized scores
        plt.scatter(superialist_normalized, tft_normalized, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)
        
        plt.xlabel('SUPERIALIST Normalized Anomaly Score')
        plt.ylabel('TFT Normalized Anomaly Score')
        plt.title('Comparison of Anomaly Scores')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add correlation coefficient
        correlation = np.corrcoef(superialist_normalized, tft_normalized)[0, 1]
        plt.annotate(f'Correlation: {correlation:.4f}', xy=(0.05, 0.95), xycoords='axes fraction')
        
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, 'anomaly_score_comparison.png'))
        plt.close()
        
        print(f"\nAnomaly Score Correlation: {correlation:.4f}")
        
        # Compare detection rates
        if 'unsafe' in superialist_results.columns:
            # Calculate TFT detection rate at different thresholds
            thresholds = np.linspace(tft_normalized.min(), tft_normalized.max(), 100)
            tft_tpr = []
            tft_fpr = []
            
            for threshold in thresholds:
                # TFT predictions (lower anomaly score = more anomalous)
                tft_pred = tft_normalized < threshold
                
                # True values
                true_values = superialist_results['unsafe'].astype(bool)
                
                # Calculate true positive rate and false positive rate
                tp = np.sum(tft_pred & true_values)
                fp = np.sum(tft_pred & ~true_values)
                tn = np.sum(~tft_pred & ~true_values)
                fn = np.sum(~tft_pred & true_values)
                
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                tft_tpr.append(tpr)
                tft_fpr.append(fpr)
            
            # Calculate SUPERIALIST detection rate at different thresholds
            thresholds = np.linspace(superialist_normalized.min(), superialist_normalized.max(), 100)
            superialist_tpr = []
            superialist_fpr = []
            
            for threshold in thresholds:
                # SUPERIALIST predictions (lower mean_loss_4 = more anomalous)
                superialist_pred = superialist_normalized < threshold
                
                # True values
                true_values = superialist_results['unsafe'].astype(bool)
                
                # Calculate true positive rate and false positive rate
                tp = np.sum(superialist_pred & true_values)
                fp = np.sum(superialist_pred & ~true_values)
                tn = np.sum(~superialist_pred & ~true_values)
                fn = np.sum(~superialist_pred & true_values)
                
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                superialist_tpr.append(tpr)
                superialist_fpr.append(fpr)
            
            # Plot ROC curves
            plt.figure(figsize=(10, 8))
            plt.plot(superialist_fpr, superialist_tpr, label='SUPERIALIST')
            plt.plot(tft_fpr, tft_tpr, label='TFT')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve Comparison')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, 'roc_curve_comparison.png'))
            plt.close()
            
            # Calculate AUC
            from sklearn.metrics import auc
            superialist_auc = auc(superialist_fpr, superialist_tpr)
            tft_auc = auc(tft_fpr, tft_tpr)
            
            print(f"\nROC AUC Comparison:")
            print(f"  SUPERIALIST AUC: {superialist_auc:.4f}")
            print(f"  TFT AUC: {tft_auc:.4f}")
            
            # Compare precision-recall curves
            from sklearn.metrics import precision_recall_curve
            
            # TFT precision-recall curve
            tft_precision, tft_recall, _ = precision_recall_curve(
                superialist_results['unsafe'].astype(bool),
                -tft_normalized  # Negate because lower anomaly score = more anomalous
            )
            
            # SUPERIALIST precision-recall curve
            superialist_precision, superialist_recall, _ = precision_recall_curve(
                superialist_results['unsafe'].astype(bool),
                -superialist_normalized  # Negate because lower mean_loss_4 = more anomalous
            )
            
            # Plot precision-recall curves
            plt.figure(figsize=(10, 8))
            plt.plot(superialist_recall, superialist_precision, label='SUPERIALIST')
            plt.plot(tft_recall, tft_precision, label='TFT')
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve Comparison')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, 'precision_recall_comparison.png'))
            plt.close()
            
            # Calculate average precision
            from sklearn.metrics import average_precision_score
            superialist_ap = average_precision_score(
                superialist_results['unsafe'].astype(bool),
                -superialist_normalized
            )
            tft_ap = average_precision_score(
                superialist_results['unsafe'].astype(bool),
                -tft_normalized
            )
            
            print(f"\nAverage Precision Comparison:")
            print(f"  SUPERIALIST AP: {superialist_ap:.4f}")
            print(f"  TFT AP: {tft_ap:.4f}")
    
    except Exception as e:
        print(f"Error comparing models: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load and preprocess training dataset
    train_df = load_dataset(args.dataset)
    train_df = preprocess_dataset(train_df)
    
    # Prepare sequence data for training
    print(f"Preparing sequence data with prediction horizon {args.future_steps}...")
    X_train, future_X_train = prepare_sequence_data_fixed(
        train_df,
        input_col='r_zero',
        window_size=CNNModel.WINSIZE,
        prediction_horizon=args.future_steps
    )
    
    # Train TFT model
    tft_model = train_tft_model(args, X_train, future_X_train)
    
    # Load and preprocess test dataset
    test_df = load_dataset(args.test_dataset)
    test_df = preprocess_dataset(test_df)
    
    # Prepare sequence data for testing
    print(f"Preparing test sequence data...")
    X_test, future_X_test = prepare_sequence_data_fixed(
        test_df,
        input_col='r_zero',
        window_size=CNNModel.WINSIZE,
        prediction_horizon=args.future_steps
    )
    
    # Compute feature importance
    print("Computing feature importance...")
    compute_feature_importance(tft_model, X_test)
    
    # Evaluate TFT model
    tft_results = evaluate_tft_model(tft_model, X_test, future_X_test, args)
    
    # Compare with SUPERIALIST model if requested
    if args.compare_superialist:
        # Load SUPERIALIST model
        superialist_model, success = load_superialist_model()
        
        if success:
            # Evaluate SUPERIALIST model
            superialist_results = evaluate_superialist_model(
                superialist_model,
                test_df,
                args.threshold
            )
            
            # Compare models
            compare_models(tft_results, superialist_results, args)
        else:
            print("Skipping SUPERIALIST comparison due to model loading failure")
    
    print("\nTraining and evaluation complete!")


if __name__ == '__main__':
    main()
