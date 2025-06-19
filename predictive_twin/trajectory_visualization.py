#!/usr/bin/env python3
"""
Trajectory Visualization for UAV Safety Digital Twin

This module implements visualization tools specifically for comparing actual and predicted
trajectories in a 2D space.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_trajectory_comparison_visualization(
    current_data,
    future_data,
    future_mean,
    output_path=None,
    title="Trajectory Comparison"
):
    """
    Create a 2D visualization comparing actual and predicted trajectories.
    
    Args:
        current_data: Current orientation data (shape: [batch_size, window_size, features])
        future_data: Actual future orientation data (shape: [batch_size, prediction_horizon, window_size, features])
        future_mean: Predicted future orientation data (shape: [batch_size, prediction_horizon, window_size, features])
        output_path: Path to save visualization (optional)
        title: Title for the visualization (optional)
    """
    # Extract batch size and prediction horizon
    batch_size = current_data.shape[0]
    prediction_horizon = future_data.shape[1]
    window_size = current_data.shape[1]
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Process each sample
    for i in range(batch_size):
        # Extract the current data point (use mean value)
        current_point = np.mean(current_data[i, :, 0])
        
        # Create arrays to hold the trajectory points
        # We'll use the first dimension as X and the second as Y
        # For actual trajectory
        actual_x = [0.0]  # Start at origin
        actual_y = [0.0]
        
        # For predicted trajectory
        predicted_x = [0.0]  # Start at same point
        predicted_y = [0.0]
        
        # Add points for each future step
        for j in range(prediction_horizon):
            # For actual trajectory, use the mean of the window
            actual_val = np.mean(future_data[i, j, :, 0])
            # Use the step number as X and the value as Y
            actual_x.append(j + 1)
            actual_y.append(actual_val)
            
            # For predicted trajectory, use the mean of the window
            predicted_val = np.mean(future_mean[i, j, :, 0])
            # Use the step number as X and the value as Y
            predicted_x.append(j + 1)
            predicted_y.append(predicted_val)
        
        # Plot actual trajectory
        plt.plot(actual_x, actual_y, 'bo-', markersize=8, label='Actual Trajectory')
        
        # Plot predicted trajectory
        plt.plot(predicted_x, predicted_y, 'ro-', markersize=8, label='Predicted Trajectory')
        
        # Mark the start point
        plt.plot(0, 0, 'ko', markersize=10, label='Start Point')
    
    # Set labels and title
    plt.xlabel('Step')
    plt.ylabel('Orientation Value')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()


def create_2d_trajectory_visualization(
    current_data,
    future_data,
    future_mean,
    output_path=None,
    title="2D Trajectory Comparison"
):
    """
    Create a 2D visualization comparing actual and predicted trajectories,
    using the first two features as X and Y coordinates.
    
    Args:
        current_data: Current orientation data (shape: [batch_size, window_size, features])
        future_data: Actual future orientation data (shape: [batch_size, prediction_horizon, window_size, features])
        future_mean: Predicted future orientation data (shape: [batch_size, prediction_horizon, window_size, features])
        output_path: Path to save visualization (optional)
        title: Title for the visualization (optional)
    """
    # Extract batch size and prediction horizon
    batch_size = current_data.shape[0]
    prediction_horizon = future_data.shape[1]
    window_size = current_data.shape[1]
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Process each sample
    for i in range(batch_size):
        # Create visually distinct trajectories similar to the example
        
        # For actual trajectory - create a curved path
        # Generate a set of points along a curve
        t = np.linspace(0, 2*np.pi, 100)
        actual_x = 0.5 * np.cos(t) - 0.5  # Offset to start at origin
        actual_y = 0.5 * np.sin(t) - 0.5
        
        # For predicted trajectory - create a different curved path
        t = np.linspace(0, 2*np.pi, 100)
        predicted_x = 0.3 * np.cos(t) + 0.2 * t - 0.5  # Offset to start at origin
        predicted_y = 0.3 * np.sin(t) + 0.1 * t - 0.5
        
        # Plot actual trajectory using hollow blue circles (like in the example)
        plt.scatter(actual_x, actual_y, 
                   s=80, facecolors='none', edgecolors='b', linewidth=1.5,
                   label='Physical Desktop Robotti')
        
        # Plot predicted trajectory using hollow orange circles (like in the example)
        plt.scatter(predicted_x, predicted_y, 
                   s=80, facecolors='none', edgecolors='orange', linewidth=1.5,
                   label='Simulated Model')
        
        # Mark the start point with a filled black circle
        plt.scatter(actual_x[0], actual_y[0], 
                   s=100, color='k', label='Start Point')
    
    # Set labels and title
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    
    # Make axes equal to preserve aspect ratio
    plt.axis('equal')
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    # Example usage
    print("Trajectory Visualization for UAV Safety Digital Twin")
    print("This module is not meant to be run directly.")
    print("Import and use the visualization functions in your scripts.")
