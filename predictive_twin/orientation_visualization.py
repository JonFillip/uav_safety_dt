#!/usr/bin/env python3
"""
Orientation Visualization for UAV Safety Digital Twin

This module implements visualization tools specifically for comparing predicted
orientation sequences with actual future orientation sequences.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add the project root to the path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_orientation_comparison_visualization(
    current_data,
    current_mean,
    future_data,
    future_mean,
    window_size=25,
    output_path=None,
    title="Orientation Prediction Comparison"
):
    """
    Create a visualization comparing predicted orientation sequences with actual future sequences.
    
    Args:
        current_data: Current orientation data (shape: [batch_size, window_size, features])
        current_mean: Reconstructed current orientation data (shape: [batch_size, window_size, features])
        future_data: Actual future orientation data (shape: [batch_size, prediction_horizon, window_size, features])
        future_mean: Predicted future orientation data (shape: [batch_size, prediction_horizon, window_size, features])
        window_size: Size of the orientation window (default: 25)
        output_path: Path to save visualization (optional)
        title: Title for the visualization (optional)
    """
    # Extract batch size and prediction horizon
    batch_size = current_data.shape[0]
    prediction_horizon = future_data.shape[1]
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(batch_size, prediction_horizon + 1, figure=fig)
    
    # Create time steps for x-axis
    time_steps = np.arange(window_size)
    
    # Plot current reconstruction for each sample
    for i in range(batch_size):
        ax_current = fig.add_subplot(gs[i, 0])
        
        # Plot actual current data
        ax_current.plot(time_steps, current_data[i, :, 0], 'b-', label='Actual')
        
        # Plot reconstructed current data
        ax_current.plot(time_steps, current_mean[i, :, 0], 'r--', label='Reconstructed')
        
        # Set labels and title
        ax_current.set_xlabel('Time Step')
        ax_current.set_ylabel('Orientation')
        ax_current.set_title(f'Current Reconstruction (Sample {i+1})')
        ax_current.legend()
        ax_current.grid(True)
        
        # Plot future predictions for each step
        for j in range(prediction_horizon):
            ax_future = fig.add_subplot(gs[i, j+1])
            
            # Plot actual future data
            ax_future.plot(time_steps, future_data[i, j, :, 0], 'b-', label='Actual')
            
            # Plot predicted future data
            ax_future.plot(time_steps, future_mean[i, j, :, 0], 'r--', label='Predicted')
            
            # Set labels and title
            ax_future.set_xlabel('Time Step')
            ax_future.set_ylabel('Orientation')
            ax_future.set_title(f'Future Step {j+1} (Sample {i+1})')
            ax_future.legend()
            ax_future.grid(True)
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()


def create_orientation_sequence_visualization(
    current_data,
    current_mean,
    future_data,
    future_mean,
    window_size=25,
    output_path=None,
    title="Orientation Sequence Prediction"
):
    """
    Create a visualization of orientation sequences over time.
    
    Args:
        current_data: Current orientation data (shape: [batch_size, window_size, features])
        current_mean: Reconstructed current orientation data (shape: [batch_size, window_size, features])
        future_data: Actual future orientation data (shape: [batch_size, prediction_horizon, window_size, features])
        future_mean: Predicted future orientation data (shape: [batch_size, prediction_horizon, window_size, features])
        window_size: Size of the orientation window (default: 25)
        output_path: Path to save visualization (optional)
        title: Title for the visualization (optional)
    """
    # Extract batch size and prediction horizon
    batch_size = current_data.shape[0]
    prediction_horizon = future_data.shape[1]
    
    # Create figure
    fig, axes = plt.subplots(batch_size, 1, figsize=(15, 5 * batch_size))
    
    # Handle single sample case
    if batch_size == 1:
        axes = [axes]
    
    # Plot orientation sequences for each sample
    for i in range(batch_size):
        ax = axes[i]
        
        # Create time steps for x-axis
        time_steps = np.arange(window_size + prediction_horizon * window_size)
        
        # Create arrays to hold the full sequences
        actual_sequence = np.zeros(window_size + prediction_horizon * window_size)
        predicted_sequence = np.zeros(window_size + prediction_horizon * window_size)
        
        # Fill in current data
        actual_sequence[:window_size] = current_data[i, :, 0]
        predicted_sequence[:window_size] = current_mean[i, :, 0]
        
        # Fill in future data
        for j in range(prediction_horizon):
            start_idx = window_size + j * window_size
            end_idx = start_idx + window_size
            actual_sequence[start_idx:end_idx] = future_data[i, j, :, 0]
            predicted_sequence[start_idx:end_idx] = future_mean[i, j, :, 0]
        
        # Plot actual sequence
        ax.plot(time_steps, actual_sequence, 'b-', label='Actual')
        
        # Plot predicted sequence
        ax.plot(time_steps, predicted_sequence, 'r--', label='Predicted')
        
        # Add vertical lines to separate windows
        for j in range(prediction_horizon + 1):
            ax.axvline(x=j * window_size, color='k', linestyle='--', alpha=0.3)
        
        # Set labels and title
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Orientation')
        ax.set_title(f'Orientation Sequence (Sample {i+1})')
        ax.legend()
        ax.grid(True)
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    # Example usage
    print("Orientation Visualization for UAV Safety Digital Twin")
    print("This module is not meant to be run directly.")
    print("Import and use the visualization functions in your scripts.")
