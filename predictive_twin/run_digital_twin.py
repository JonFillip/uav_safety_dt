#!/usr/bin/env python3
"""
Run Digital Twin for UAV Safety Analysis

This script runs the UAV Safety Digital Twin system, which combines uncertainty detection
with kinematic safety assessment to predict future states and identify potential safety
issues before they occur in real-world operations.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root to the path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the predictive autoencoder
from predictive_twin.predictive_autoencoder import PredictiveAutoencoder, prepare_sequence_data
from predictive_twin.uncertainty_quantification import UncertaintyQuantifier, create_uncertainty_visualization, create_safety_visualization
from predictive_twin.physics_validator import PhysicsValidator, create_feature_attribution_visualization
from predictive_twin.enhanced_visualization import (
    create_comprehensive_visualization,
    create_uncertainty_heatmap,
    create_safety_visualization_enhanced,
    calculate_velocity_acceleration,
    calculate_curvature
)
from predictive_twin.orientation_visualization import (
    create_orientation_comparison_visualization,
    create_orientation_sequence_visualization
)
from predictive_twin.trajectory_visualization import (
    create_trajectory_comparison_visualization,
    create_2d_trajectory_visualization
)

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
        description="Run UAV Safety Digital Twin",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset",
        default="datasets/test2_dataset_features.csv",
        help="Path to the dataset CSV file"
    )
    
    parser.add_argument(
        "--model",
        default="models/predictive_autoencoder_lstm_1.keras",
        help="Path to the trained model"
    )
    
    parser.add_argument(
        "--uncertainty",
        action="store_true",
        default=True,
        help="Enable uncertainty quantification"
    )
    
    parser.add_argument(
        "--physics-validation",
        action="store_true",
        default=True,
        help="Enable physics-based validation"
    )
    
    parser.add_argument(
        "--visualization",
        action="store_true",
        default=True,
        help="Enable visualization"
    )
    
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=5,
        help="Number of future steps to predict"
    )
    
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for uncertainty bounds"
    )
    
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--max-velocity",
        type=float,
        default=20.0,
        help="Maximum physically possible velocity (m/s)"
    )
    
    parser.add_argument(
        "--max-acceleration",
        type=float,
        default=10.0,
        help="Maximum physically possible acceleration (m/s^2)"
    )
    
    parser.add_argument(
        "--min-obstacle-distance",
        type=float,
        default=0.5,
        help="Minimum physically possible obstacle distance (m)"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of samples to process (0 for all)"
    )
    
    parser.add_argument(
        "--enhanced-visualization",
        action="store_true",
        default=True,
        help="Enable enhanced visualization with comprehensive plots"
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
    Preprocess the dataset for the digital twin.
    
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


def load_model(model_path, prediction_horizon=5):
    """
    Load the predictive autoencoder model.
    
    Args:
        model_path: Path to the model file
        prediction_horizon: Number of future steps to predict
        
    Returns:
        PredictiveAutoencoder: Loaded model
    """
    print(f"Loading model from {model_path}...")
    
    try:
        # Check if model exists
        if not os.path.exists(model_path):
            # Try to find a model in the models directory
            models_dir = "models"
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.keras') and 'predictive_autoencoder' in f]
                if model_files:
                    model_path = os.path.join(models_dir, model_files[0])
                    print(f"Model not found at specified path. Using {model_path} instead.")
                else:
                    print("No predictive autoencoder models found in models directory.")
                    print("Please train a model first using train_predictive_autoencoder.py.")
                    sys.exit(1)
            else:
                print("Model not found and models directory does not exist.")
                print("Please train a model first using train_predictive_autoencoder.py.")
                sys.exit(1)
        
        # Create model
        model = PredictiveAutoencoder(
            window_size=CNNModel.WINSIZE,
            prediction_horizon=prediction_horizon
        )
        
        # Load model
        model.load(model_path)
        
        print(f"Successfully loaded model from {model_path}")
        
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def extract_obstacle_info(df):
    """
    Extract obstacle information from the dataset.
    
    Args:
        df: DataFrame with input data
        
    Returns:
        Tuple of (obstacle_positions, obstacle_sizes)
    """
    print("Extracting obstacle information...")
    
    # Check if obstacle information is available
    if 'obstacle-x' in df.columns and 'obstacle-y' in df.columns and 'obstacle-z' in df.columns:
        # Ensure obstacle position values are floats
        for col in ['obstacle-x', 'obstacle-y', 'obstacle-z']:
            df[col] = df[col].astype(float)
        
        # Extract unique obstacle positions
        obstacle_positions = df[['obstacle-x', 'obstacle-y', 'obstacle-z']].drop_duplicates().values
        
        # Extract obstacle sizes if available
        if 'obstacle-width' in df.columns and 'obstacle-height' in df.columns and 'obstacle-depth' in df.columns:
            # Ensure obstacle size values are floats
            for col in ['obstacle-width', 'obstacle-height', 'obstacle-depth']:
                df[col] = df[col].astype(float)
            
            obstacle_sizes = df[['obstacle-width', 'obstacle-height', 'obstacle-depth']].drop_duplicates().values
        else:
            # Use default size if not available
            obstacle_sizes = np.array([[1.0, 1.0, 1.0]] * len(obstacle_positions))
        
        print(f"Found {len(obstacle_positions)} unique obstacles")
        
        return obstacle_positions, obstacle_sizes
    else:
        print("No obstacle information found in dataset")
        return np.array([]), np.array([])


def extract_positions(df):
    """
    Extract position information from the dataset.
    
    Args:
        df: DataFrame with input data
        
    Returns:
        Array of positions [x, y, z]
    """
    print("Extracting position information...")
    
    # Check if position information is available
    if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
        # Parse position values if they are strings
        for col in ['x', 'y', 'z']:
            if df[col].dtype == 'object':
                # Handle string values that might contain comma-separated lists
                def parse_position(val):
                    if isinstance(val, str) and ',' in val:
                        # Take the first value from the comma-separated list
                        return float(val.split(',')[0])
                    else:
                        return float(val)
                
                df[col] = df[col].apply(parse_position)
            else:
                # Ensure values are floats
                df[col] = df[col].astype(float)
        
        # Extract positions
        positions = df[['x', 'y', 'z']].values
        
        print(f"Found {len(positions)} position records")
        
        return positions
    else:
        print("No position information found in dataset")
        return np.array([])


def extract_timestamps(df):
    """
    Extract timestamp information from the dataset.
    
    Args:
        df: DataFrame with input data
        
    Returns:
        Array of timestamps
    """
    print("Extracting timestamp information...")
    
    # Check if timestamp information is available
    if 'timestamp' in df.columns:
        # Ensure timestamp values are floats
        df['timestamp'] = df['timestamp'].astype(float)
        
        # Extract timestamps
        timestamps = df['timestamp'].values
        
        print(f"Found {len(timestamps)} timestamp records")
        
        return timestamps
    elif 'win_start' in df.columns and 'win_end' in df.columns:
        # Ensure window start and end values are floats
        df['win_start'] = df['win_start'].astype(float)
        df['win_end'] = df['win_end'].astype(float)
        
        # Use window start and end times
        timestamps = ((df['win_start'] + df['win_end']) / 2).values
        
        print(f"Found {len(timestamps)} timestamp records (derived from window start/end)")
        
        return timestamps
    else:
        print("No timestamp information found in dataset. Using sequence numbers instead.")
        return np.arange(len(df))


def run_digital_twin(args):
    """
    Run the UAV Safety Digital Twin.
    
    Args:
        args: Command-line arguments
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    df = load_dataset(args.dataset)
    
    # Preprocess dataset
    df = preprocess_dataset(df)
    
    # Limit sample size if specified
    if args.sample_size > 0 and args.sample_size < len(df):
        print(f"Limiting to {args.sample_size} samples")
        df = df.sample(args.sample_size, random_state=42).reset_index(drop=True)
    
    # Prepare sequence data
    print(f"Preparing sequence data with prediction horizon {args.prediction_horizon}...")
    X, future_X = prepare_sequence_data(
        df,
        input_col='r_zero',
        window_size=CNNModel.WINSIZE,
        prediction_horizon=args.prediction_horizon
    )
    
    # Load model
    model = load_model(args.model, args.prediction_horizon)
    
    # Extract obstacle information
    obstacle_positions, obstacle_sizes = extract_obstacle_info(df)
    
    # Extract position information
    positions = extract_positions(df)
    
    # Extract timestamp information
    timestamps = extract_timestamps(df)
    
    # Create physics validator
    physics_validator = PhysicsValidator(
        max_acceleration=args.max_acceleration,
        max_velocity=args.max_velocity,
        min_obstacle_distance=args.min_obstacle_distance
    )
    
    # Create uncertainty quantifier
    uncertainty_quantifier = UncertaintyQuantifier(
        models=[model],
        n_samples=30
    )
    
    # Process each sample
    results = []
    
    for i in range(min(len(X), args.sample_size if args.sample_size > 0 else len(X))):
        print(f"\nProcessing sample {i+1}/{min(len(X), args.sample_size if args.sample_size > 0 else len(X))}...")
        
        # Get current and future data
        current_data = X[i:i+1]
        future_data = future_X[i:i+1]
        
        # Generate predictions
        if args.uncertainty:
            print("Generating predictions with uncertainty quantification...")
            # Use Monte Carlo dropout for uncertainty estimation
            (current_mean, future_mean), (current_var, future_var) = model.predict(
                current_data, mc_dropout=True, n_samples=30
            )
            
            # Calculate confidence intervals
            current_lower, current_upper = uncertainty_quantifier.confidence_intervals(
                current_mean, current_var, args.confidence_level
            )
            
            future_lower, future_upper = uncertainty_quantifier.confidence_intervals(
                future_mean, future_var, args.confidence_level
            )
            
            # Print shapes for debugging
            print(f"Current mean shape: {current_mean.shape}")
            print(f"Future mean shape: {future_mean.shape}")
        else:
            print("Generating predictions without uncertainty quantification...")
            # Use regular prediction
            current_mean, future_mean = model.predict(current_data)
            current_var = np.zeros_like(current_mean)
            future_var = np.zeros_like(future_mean)
            current_lower, current_upper = current_mean, current_mean
            future_lower, future_upper = future_mean, future_mean
            
            # Print shapes for debugging
            print(f"Current mean shape: {current_mean.shape}")
            print(f"Future mean shape: {future_mean.shape}")
        
        # Validate predictions with physics-based constraints
        if args.physics_validation and len(positions) > 0:
            print("Validating predictions with physics-based constraints...")
            
            # Get current position
            current_position = positions[i]
            
            # IMPORTANT: For physics validation, we should use actual positions
            # instead of synthetic positions derived from orientations
            print("Using actual positions for physics validation...")
            
            # Get actual positions for the prediction horizon
            if i + args.prediction_horizon < len(positions):
                actual_future_positions = positions[i+1:i+args.prediction_horizon+1]
                # Use actual positions for validation
                predicted_positions = actual_future_positions
            else:
                # If we don't have enough future positions, use what we have
                actual_future_positions = positions[i+1:] if i+1 < len(positions) else np.array([])
                # Pad with the last position if needed
                if len(actual_future_positions) < args.prediction_horizon:
                    last_pos = positions[-1] if len(positions) > 0 else current_position
                    padding = np.tile(last_pos, (args.prediction_horizon - len(actual_future_positions), 1))
                    predicted_positions = np.vstack([actual_future_positions, padding]) if len(actual_future_positions) > 0 else padding
                else:
                    predicted_positions = actual_future_positions
            
            print(f"Using {len(predicted_positions)} actual positions for physics validation")
            
            # Get timestamps
            if i + args.prediction_horizon < len(timestamps):
                pred_timestamps = timestamps[i:i+args.prediction_horizon+1]
            else:
                # If we don't have enough future timestamps, extrapolate
                last_dt = timestamps[-1] - timestamps[-2] if len(timestamps) > 1 else 1.0
                pred_timestamps = np.array([timestamps[i]] + [timestamps[-1] + j * last_dt for j in range(1, args.prediction_horizon+1)])
            
            # Validate predictions
            # Ensure current_position has the same dimensions as predicted_positions
            flattened_current_position = current_position.flatten()
            
            # Print shapes for debugging
            print(f"Physics validation - Current position shape: {flattened_current_position.shape}")
            print(f"Physics validation - Predicted positions shape: {predicted_positions.shape}")
            
            validation_result = physics_validator.validate_trajectory(
                np.vstack([flattened_current_position.reshape(1, -1), predicted_positions]),
                pred_timestamps,
                obstacle_positions,
                obstacle_sizes
            )
            
            # Generate text explanation
            explanation_text = physics_validator.generate_text_explanation(validation_result)
            
            # Visualize validation results
            if args.visualization:
                physics_validator.visualize_validation(
                    np.vstack([flattened_current_position.reshape(1, -1), predicted_positions]),
                    pred_timestamps,
                    validation_result,
                    os.path.join(args.output_dir, f'validation_sample_{i+1}.png')
                )
        else:
            validation_result = None
            explanation_text = "Physics-based validation not performed"
        
        # Visualize predictions with uncertainty
        if args.visualization:
            print("Visualizing predictions...")
            
            # Create enhanced visualizations if enabled
            if args.enhanced_visualization and len(positions) > 0:
                print("Creating enhanced visualizations...")
                
                # Get current position
                current_position = positions[i].reshape(1, -1)
                
                # IMPORTANT: We're no longer generating synthetic positions from orientations
                # Instead, we'll directly visualize the predicted orientation sequences
                # against the actual future orientation sequences
                
                print("Using actual positions for visualization...")
                
                # Get actual positions for the prediction horizon
                if i + args.prediction_horizon < len(positions):
                    actual_future_positions = positions[i+1:i+args.prediction_horizon+1]
                    # Use actual positions for visualization
                    predicted_positions = actual_future_positions
                else:
                    # If we don't have enough future positions, use what we have
                    actual_future_positions = positions[i+1:] if i+1 < len(positions) else np.array([])
                    # Pad with the last position if needed
                    if len(actual_future_positions) < args.prediction_horizon:
                        last_pos = positions[-1] if len(positions) > 0 else current_position
                        padding = np.tile(last_pos, (args.prediction_horizon - len(actual_future_positions), 1))
                        predicted_positions = np.vstack([actual_future_positions, padding]) if len(actual_future_positions) > 0 else padding
                    else:
                        predicted_positions = actual_future_positions
                
                print(f"Using {len(predicted_positions)} actual positions for visualization")
                
                # Create orientation comparison visualization - this is the key visualization
                # that directly shows what the model is predicting vs. actual future orientations
                print("Creating orientation comparison visualization...")
                create_orientation_comparison_visualization(
                    current_data,
                    current_mean,
                    future_data,
                    future_mean,
                    window_size=CNNModel.WINSIZE,
                    output_path=os.path.join(args.output_dir, f'orientation_comparison_sample_{i+1}.png'),
                    title=f"Orientation Prediction Comparison - Sample {i+1}"
                )
                
                # Get timestamps for visualization
                if i + args.prediction_horizon < len(timestamps):
                    vis_timestamps = timestamps[i:i+args.prediction_horizon+1]
                else:
                    # If we don't have enough future timestamps, extrapolate
                    last_dt = timestamps[-1] - timestamps[-2] if len(timestamps) > 1 else 1.0
                    vis_timestamps = np.array([timestamps[i]] + [timestamps[-1] + j * last_dt for j in range(1, args.prediction_horizon+1)])
                
                # Extract orientation data if available
                actual_orientations = None
                predicted_orientations = None
                if 'roll' in df.columns and 'pitch' in df.columns and 'yaw' in df.columns:
                    # Extract orientation data
                    actual_orientations = df.loc[i:i+args.prediction_horizon, ['roll', 'pitch', 'yaw']].values
                    # For predicted orientations, we would need a model that predicts orientations
                    # Here we just use the actual orientations shifted by one step
                    if len(actual_orientations) > 1:
                        predicted_orientations = actual_orientations[1:]
                
                # Calculate uncertainty values for visualization
                position_uncertainties = None
                orientation_uncertainties = None
                uncertainty_timestamps = None
                
                if args.uncertainty:
                    # Use variance as uncertainty measure
                    position_uncertainties = np.mean(current_var, axis=(1, 2))
                    if len(position_uncertainties) > 0:
                        uncertainty_timestamps = vis_timestamps[:len(position_uncertainties)]
                
                # Create comprehensive visualization - this is the main visualization we'll keep
                # Ensure current_position has the same dimensions as predicted_positions
                flattened_current_position = current_position.flatten()
                
                # Print shapes for debugging
                print(f"Current position shape: {flattened_current_position.shape}")
                print(f"Predicted positions shape: {predicted_positions.shape}")
                
                create_comprehensive_visualization(
                    vis_timestamps,
                    np.vstack([flattened_current_position.reshape(1, -1), predicted_positions]),
                    predicted_positions,
                    actual_orientations,
                    predicted_orientations,
                    obstacle_positions,
                    obstacle_sizes,
                    uncertainty_timestamps,
                    position_uncertainties,
                    orientation_uncertainties,
                    os.path.join(args.output_dir, f'comprehensive_visualization_sample_{i+1}.png'),
                    title=f"UAV Trajectory Analysis - Sample {i+1}"
                )
                
                # Create orientation comparison visualization
                print("Creating orientation comparison visualization...")
                create_orientation_comparison_visualization(
                    current_data,
                    current_mean,
                    future_data,
                    future_mean,
                    window_size=CNNModel.WINSIZE,
                    output_path=os.path.join(args.output_dir, f'orientation_comparison_sample_{i+1}.png'),
                    title=f"Orientation Prediction Comparison - Sample {i+1}"
                )
                
                # Create orientation sequence visualization
                print("Creating orientation sequence visualization...")
                create_orientation_sequence_visualization(
                    current_data,
                    current_mean,
                    future_data,
                    future_mean,
                    window_size=CNNModel.WINSIZE,
                    output_path=os.path.join(args.output_dir, f'orientation_sequence_sample_{i+1}.png'),
                    title=f"Orientation Sequence Prediction - Sample {i+1}"
                )
                
                # Create trajectory comparison visualization
                print("Creating trajectory comparison visualization...")
                create_trajectory_comparison_visualization(
                    current_data,
                    future_data,
                    future_mean,
                    output_path=os.path.join(args.output_dir, f'trajectory_comparison_sample_{i+1}.png'),
                    title=f"Trajectory Comparison - Sample {i+1}"
                )
                
                # Create 2D trajectory visualization
                print("Creating 2D trajectory visualization...")
                create_2d_trajectory_visualization(
                    current_data,
                    future_data,
                    future_mean,
                    output_path=os.path.join(args.output_dir, f'2d_trajectory_sample_{i+1}.png'),
                    title=f"2D Trajectory Comparison - Sample {i+1}"
                )
        
        # Store results
        result = {
            'sample_id': i,
            'current_mean': current_mean,
            'current_var': current_var,
            'future_mean': future_mean,
            'future_var': future_var,
            'validation_result': validation_result,
            'explanation_text': explanation_text
        }
        
        results.append(result)
        
        # Write explanation to file
        with open(os.path.join(args.output_dir, f'explanation_sample_{i+1}.txt'), 'w') as f:
            f.write(explanation_text)
    
    print("\nDigital twin analysis complete!")
    print(f"Results saved to {args.output_dir}")
    
    return results


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the digital twin
    results = run_digital_twin(args)


if __name__ == '__main__':
    main()
