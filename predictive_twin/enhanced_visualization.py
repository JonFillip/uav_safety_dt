#!/usr/bin/env python3
"""
Enhanced Visualization for UAV Safety Digital Twin

This module implements comprehensive visualization tools for the UAV Safety Digital Twin,
providing detailed views of UAV trajectories, kinematics, and uncertainty.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, Ellipse
import matplotlib.colors as mcolors
from scipy.spatial.distance import cdist

# Add the project root to the path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def calculate_velocity_acceleration(positions, timestamps):
    """
    Calculate velocity and acceleration from position data.
    
    Args:
        positions: Array of positions [x, y, z]
        timestamps: Array of timestamps
        
    Returns:
        Tuple of (velocities, accelerations, velocity_magnitudes, acceleration_magnitudes)
    """
    # Calculate time differences
    dt = np.diff(timestamps)
    
    # Calculate velocities
    velocities = np.diff(positions, axis=0) / dt[:, np.newaxis]
    
    # Calculate accelerations
    accelerations = np.diff(velocities, axis=0) / dt[1:][:, np.newaxis]
    
    # Calculate magnitudes
    velocity_magnitudes = np.linalg.norm(velocities, axis=1)
    acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
    
    return velocities, accelerations, velocity_magnitudes, acceleration_magnitudes


def calculate_curvature(positions, timestamps):
    """
    Calculate curvature of the trajectory.
    
    Args:
        positions: Array of positions [x, y, z]
        timestamps: Array of timestamps
        
    Returns:
        Array of curvature values
    """
    # Calculate velocities and accelerations
    velocities, accelerations, _, _ = calculate_velocity_acceleration(positions, timestamps)
    
    # Calculate curvature using the formula κ = |r'×r''|/|r'|³
    curvature = np.zeros(len(velocities) - 1)
    
    for i in range(len(velocities) - 1):
        v = velocities[i]
        a = accelerations[i]
        
        # Calculate cross product |v × a|
        cross_product = np.linalg.norm(np.cross(v, a))
        
        # Calculate |v|³
        v_norm_cubed = np.linalg.norm(v) ** 3
        
        # Avoid division by zero
        if v_norm_cubed > 1e-10:
            curvature[i] = cross_product / v_norm_cubed
        else:
            curvature[i] = 0
    
    return curvature


def calculate_prediction_error(actual_positions, predicted_positions):
    """
    Calculate prediction error between actual and predicted positions.
    
    Args:
        actual_positions: Array of actual positions [x, y, z]
        predicted_positions: Array of predicted positions [x, y, z]
        
    Returns:
        Array of prediction error magnitudes
    """
    # Ensure the arrays have the same length
    min_length = min(len(actual_positions), len(predicted_positions))
    actual = actual_positions[:min_length]
    predicted = predicted_positions[:min_length]
    
    # Calculate error vectors
    error_vectors = actual - predicted
    
    # Calculate error magnitudes
    error_magnitudes = np.linalg.norm(error_vectors, axis=1)
    
    return error_magnitudes, error_vectors


def create_comprehensive_visualization(
    timestamps, 
    actual_positions, 
    predicted_positions=None, 
    actual_orientations=None,
    predicted_orientations=None,
    obstacle_positions=None, 
    obstacle_sizes=None,
    uncertainty_timestamps=None,
    position_uncertainties=None,
    orientation_uncertainties=None,
    output_path=None,
    title="UAV Trajectory with Uncertainty"
):
    """
    Create a comprehensive visualization of UAV trajectory with uncertainty.
    
    Args:
        timestamps: Array of timestamps
        actual_positions: Array of actual positions [x, y, z]
        predicted_positions: Array of predicted positions [x, y, z] (optional)
        actual_orientations: Array of actual orientations [roll, pitch, yaw] (optional)
        predicted_orientations: Array of predicted orientations [roll, pitch, yaw] (optional)
        obstacle_positions: Array of obstacle positions [x, y, z] (optional)
        obstacle_sizes: Array of obstacle sizes [width, height, depth] (optional)
        uncertainty_timestamps: Array of timestamps for uncertainty markers (optional)
        position_uncertainties: Array of position uncertainties (optional)
        orientation_uncertainties: Array of orientation uncertainties (optional)
        output_path: Path to save visualization (optional)
        title: Title for the visualization (optional)
    """
    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(5, 2, figure=fig, height_ratios=[1, 1, 1, 1, 1], width_ratios=[1, 1])
    
    # Calculate derived quantities
    velocities, accelerations, velocity_magnitudes, acceleration_magnitudes = calculate_velocity_acceleration(
        actual_positions, timestamps
    )
    
    curvature = calculate_curvature(actual_positions, timestamps)
    
    # Calculate prediction error if predicted positions are provided
    if predicted_positions is not None:
        prediction_error, error_vectors = calculate_prediction_error(
            actual_positions[1:], predicted_positions[:len(actual_positions)-1]
        )
    
    # Create time series plots (left column)
    
    # Plot 1: X position
    ax_x = fig.add_subplot(gs[0, 0])
    ax_x.plot(timestamps, actual_positions[:, 0], 'b-', label='Actual X')
    if predicted_positions is not None:
        # Plot predicted X positions, aligning with the actual timestamps
        pred_len = min(len(predicted_positions), len(timestamps) - 1)
        ax_x.plot(timestamps[1:pred_len+1], predicted_positions[:pred_len, 0], 'b--', label='Predicted X')
    
    # Add uncertainty bands if provided
    if uncertainty_timestamps is not None and position_uncertainties is not None:
        for t, u in zip(uncertainty_timestamps, position_uncertainties):
            ax_x.axvline(x=t, color='r', alpha=0.3)
    
    ax_x.set_ylabel('X (m)')
    ax_x.set_title('Position and Orientation over Time')
    ax_x.legend()
    ax_x.grid(True)
    
    # Plot 2: Y position
    ax_y = fig.add_subplot(gs[1, 0], sharex=ax_x)
    ax_y.plot(timestamps, actual_positions[:, 1], 'g-', label='Actual Y')
    if predicted_positions is not None:
        pred_len = min(len(predicted_positions), len(timestamps) - 1)
        ax_y.plot(timestamps[1:pred_len+1], predicted_positions[:pred_len, 1], 'g--', label='Predicted Y')
    
    # Add uncertainty bands if provided
    if uncertainty_timestamps is not None and position_uncertainties is not None:
        for t, u in zip(uncertainty_timestamps, position_uncertainties):
            ax_y.axvline(x=t, color='r', alpha=0.3)
    
    ax_y.set_ylabel('Y (m)')
    ax_y.legend()
    ax_y.grid(True)
    
    # Plot 3: Z position
    ax_z = fig.add_subplot(gs[2, 0], sharex=ax_x)
    ax_z.plot(timestamps, actual_positions[:, 2], 'r-', label='Actual Z')
    if predicted_positions is not None:
        pred_len = min(len(predicted_positions), len(timestamps) - 1)
        ax_z.plot(timestamps[1:pred_len+1], predicted_positions[:pred_len, 2], 'r--', label='Predicted Z')
    
    # Add uncertainty bands if provided
    if uncertainty_timestamps is not None and position_uncertainties is not None:
        for t, u in zip(uncertainty_timestamps, position_uncertainties):
            ax_z.axvline(x=t, color='r', alpha=0.3)
    
    ax_z.set_ylabel('Z (m)')
    ax_z.legend()
    ax_z.grid(True)
    
    # Plot 4: Yaw (if orientations are provided)
    ax_yaw = fig.add_subplot(gs[3, 0], sharex=ax_x)
    if actual_orientations is not None:
        ax_yaw.plot(timestamps, actual_orientations[:, 2], 'k-', label='Actual Yaw')
    if predicted_orientations is not None:
        pred_len = min(len(predicted_orientations), len(timestamps) - 1)
        ax_yaw.plot(timestamps[1:pred_len+1], predicted_orientations[:pred_len, 2], 'k--', label='Predicted Yaw')
    
    # Add uncertainty bands if provided
    if uncertainty_timestamps is not None and orientation_uncertainties is not None:
        for t, u in zip(uncertainty_timestamps, orientation_uncertainties):
            ax_yaw.axvline(x=t, color='r', alpha=0.3)
    
    ax_yaw.set_ylabel('Yaw (°)')
    # Only add legend if we have orientation data
    if actual_orientations is not None or predicted_orientations is not None:
        ax_yaw.legend()
    ax_yaw.grid(True)
    
    # Plot 5: Velocity, Acceleration, and Curvature
    ax_vel = fig.add_subplot(gs[4, 0], sharex=ax_x)
    
    # Plot velocity
    ax_vel.plot(timestamps[1:], velocity_magnitudes, 'b-', label='Velocity')
    ax_vel.set_ylabel('Velocity (m/s)', color='b')
    ax_vel.tick_params(axis='y', labelcolor='b')
    ax_vel.set_xlabel('Time (s)')
    ax_vel.grid(True)
    
    # Create second y-axis for acceleration
    ax_acc = ax_vel.twinx()
    ax_acc.plot(timestamps[2:], acceleration_magnitudes, 'r-', label='Acceleration')
    ax_acc.set_ylabel('Acceleration (m/s²)', color='r')
    ax_acc.tick_params(axis='y', labelcolor='r')
    
    # Create third y-axis for curvature
    ax_curv = ax_vel.twinx()
    # Offset the third y-axis to the right
    ax_curv.spines['right'].set_position(('outward', 60))
    
    # Only plot curvature if we have enough data points
    if len(timestamps) > 3 and len(curvature) > 0:
        # Make sure timestamps and curvature have the same length
        plot_timestamps = timestamps[2:2+len(curvature)]
        ax_curv.plot(plot_timestamps, curvature, 'g-', label='Curvature')
        ax_curv.set_ylabel('Curvature (1/m)', color='g')
        ax_curv.tick_params(axis='y', labelcolor='g')
    else:
        ax_curv.set_ylabel('Curvature (1/m)', color='g')
        ax_curv.tick_params(axis='y', labelcolor='g')
    
    # Add a combined legend
    lines_vel, labels_vel = ax_vel.get_legend_handles_labels()
    lines_acc, labels_acc = ax_acc.get_legend_handles_labels()
    lines_curv, labels_curv = ax_curv.get_legend_handles_labels()
    ax_vel.legend(lines_vel + lines_acc + lines_curv, labels_vel + labels_acc + labels_curv, loc='upper right')
    
    # Create 2D trajectory plot (top right)
    ax_traj = fig.add_subplot(gs[0:3, 1])
    
    # Create more dense points for visualization by interpolating between existing points
    def interpolate_positions(positions, factor=5):
        """Interpolate between positions to create more dense points for visualization"""
        if len(positions) <= 1:
            return positions
            
        dense_positions = []
        for i in range(len(positions) - 1):
            # Add the current position
            dense_positions.append(positions[i])
            
            # Add interpolated positions
            for j in range(1, factor):
                t = j / factor
                interp_pos = positions[i] * (1 - t) + positions[i + 1] * t
                dense_positions.append(interp_pos)
                
        # Add the last position
        dense_positions.append(positions[-1])
        
        return np.array(dense_positions)
    
    # Create more dense points for visualization
    dense_actual = interpolate_positions(actual_positions, factor=10)
    
    # Plot actual trajectory using hollow blue circles with connecting lines
    ax_traj.plot(dense_actual[:, 0], dense_actual[:, 1], 'b-', alpha=0.5, linewidth=1)  # Add connecting line
    ax_traj.scatter(dense_actual[:, 0], dense_actual[:, 1], 
                   s=80, facecolors='none', edgecolors='b', linewidth=1.5,
                   label='Actual Path')
    
    # Plot predicted trajectory if provided - using hollow orange circles with connecting lines
    if predicted_positions is not None:
        # Create more dense points for visualization
        dense_predicted = interpolate_positions(predicted_positions, factor=10)
        
        # Add connecting line first
        ax_traj.plot(dense_predicted[:, 0], dense_predicted[:, 1], 'orange', alpha=0.5, linewidth=1)
        
        # Then add scatter points
        ax_traj.scatter(dense_predicted[:, 0], dense_predicted[:, 1], 
                       s=80, facecolors='none', edgecolors='orange', linewidth=1.5,
                       label='Predicted Path')
        
        # Mark the start point with a filled black circle
        ax_traj.scatter(actual_positions[0, 0], actual_positions[0, 1], 
                       s=100, color='k', label='Start Point')
    
    # Plot obstacles if provided
    if obstacle_positions is not None and obstacle_sizes is not None:
        for pos, size in zip(obstacle_positions, obstacle_sizes):
            # Create rectangle for obstacle
            rect = Rectangle(
                (pos[0] - size[0]/2, pos[1] - size[1]/2),
                size[0], size[1],
                color='gray', alpha=0.5
            )
            ax_traj.add_patch(rect)
    
    # Plot uncertainty ellipses if provided - make them smaller and more transparent
    if position_uncertainties is not None and uncertainty_timestamps is not None:
        for i, t in enumerate(uncertainty_timestamps):
            # Find the closest timestamp
            idx = np.argmin(np.abs(timestamps - t))
            if idx < len(actual_positions):
                pos = actual_positions[idx]
                # Scale down the uncertainty to make the ellipse smaller
                unc = min(position_uncertainties[i], 0.01)  # Limit the size
                
                # Create uncertainty ellipse with reduced size and opacity
                ellipse = Ellipse(
                    (pos[0], pos[1]),
                    width=unc*2, height=unc*2,
                    color='r', alpha=0.1  # Reduced opacity
                )
                ax_traj.add_patch(ellipse)
    
    # Add current position marker
    ax_traj.plot(actual_positions[-1, 0], actual_positions[-1, 1], 'ro', markersize=8, label='Current Position')
    
    # Calculate and display distance to nearest obstacle
    if obstacle_positions is not None and len(obstacle_positions) > 0:
        current_pos = actual_positions[-1]
        distances = []
        
        for pos, size in zip(obstacle_positions, obstacle_sizes):
            # Calculate distance to obstacle surface
            dx = max(0, abs(current_pos[0] - pos[0]) - size[0]/2)
            dy = max(0, abs(current_pos[1] - pos[1]) - size[1]/2)
            dz = max(0, abs(current_pos[2] - pos[2]) - size[2]/2)
            
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            distances.append(distance)
        
        if distances:  # Check if distances list is not empty
            min_distance = min(distances)
            min_idx = np.argmin(distances)
            
            # Add distance text
            ax_traj.text(
                0.05, 0.05,
                f'Distance: {min_distance:.2f} m',
                transform=ax_traj.transAxes,
                bbox=dict(facecolor='white', alpha=0.7)
            )
            
            # Draw line to nearest obstacle
            nearest_obs = obstacle_positions[min_idx]
            ax_traj.plot([current_pos[0], nearest_obs[0]], [current_pos[1], nearest_obs[1]], 'k--', alpha=0.5)
    
    ax_traj.set_xlabel('X (m)')
    ax_traj.set_ylabel('Y (m)')
    ax_traj.set_title('2D Trajectory View')
    ax_traj.legend()
    ax_traj.grid(True)
    ax_traj.axis('equal')
    
    # Create prediction error plot (bottom right)
    ax_error = fig.add_subplot(gs[3:5, 1])
    
    if predicted_positions is not None:
        # Plot prediction error
        ax_error.plot(timestamps[1:len(prediction_error)+1], prediction_error, 'r-', label='Prediction Error')
        ax_error.set_xlabel('Time (s)')
        ax_error.set_ylabel('Error (m)')
        ax_error.set_title('Prediction Error')
        ax_error.grid(True)
        
        # Add error statistics
        mean_error = np.mean(prediction_error)
        max_error = np.max(prediction_error)
        ax_error.text(
            0.05, 0.95,
            f'Mean Error: {mean_error:.2f} m\nMax Error: {max_error:.2f} m',
            transform=ax_error.transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7)
        )
    else:
        ax_error.text(
            0.5, 0.5,
            'No prediction data available',
            transform=ax_error.transAxes,
            horizontalalignment='center',
            verticalalignment='center'
        )
    
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


def create_uncertainty_heatmap(
    actual_positions,
    timestamps,
    uncertainties,
    obstacle_positions=None,
    obstacle_sizes=None,
    output_path=None
):
    """
    Create a heatmap visualization of uncertainty along the trajectory.
    
    Args:
        actual_positions: Array of actual positions [x, y, z]
        timestamps: Array of timestamps
        uncertainties: Array of uncertainty values
        obstacle_positions: Array of obstacle positions [x, y, z] (optional)
        obstacle_sizes: Array of obstacle sizes [width, height, depth] (optional)
        output_path: Path to save visualization (optional)
    """
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Check if we have enough data
    if len(uncertainties) == 0 or len(actual_positions) <= 1:
        plt.text(0.5, 0.5, 'Not enough data for uncertainty heatmap',
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
        
        # Save or show plot
        if output_path:
            plt.savefig(output_path, dpi=300)
            plt.close()
        else:
            plt.show()
        return
    
    # Create colormap for uncertainty
    cmap = plt.cm.get_cmap('viridis')
    norm = plt.Normalize(vmin=min(uncertainties), vmax=max(uncertainties))
    
    # Plot trajectory with color based on uncertainty
    for i in range(min(len(actual_positions) - 1, len(uncertainties))):
        plt.plot(
            [actual_positions[i, 0], actual_positions[i+1, 0]],
            [actual_positions[i, 1], actual_positions[i+1, 1]],
            color=cmap(norm(uncertainties[i])),
            linewidth=2
        )
    
    # Plot obstacles if provided
    if obstacle_positions is not None and obstacle_sizes is not None:
        for pos, size in zip(obstacle_positions, obstacle_sizes):
            # Create rectangle for obstacle
            rect = Rectangle(
                (pos[0] - size[0]/2, pos[1] - size[1]/2),
                size[0], size[1],
                color='gray', alpha=0.5
            )
            plt.gca().add_patch(rect)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Uncertainty')
    
    # Set labels and title
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Uncertainty Heatmap')
    plt.grid(True)
    plt.axis('equal')
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()


def create_safety_visualization_enhanced(
    timestamps,
    positions,
    safety_margins,
    safety_probs,
    threshold=0,
    obstacle_positions=None,
    obstacle_sizes=None,
    output_path=None
):
    """
    Create an enhanced visualization of safety margins and probabilities.
    
    Args:
        timestamps: Array of timestamps
        positions: Array of positions [x, y, z]
        safety_margins: Array of safety margins
        safety_probs: Array of safety probabilities
        threshold: Safety threshold (default: 0)
        obstacle_positions: Array of obstacle positions [x, y, z] (optional)
        obstacle_sizes: Array of obstacle sizes [width, height, depth] (optional)
        output_path: Path to save visualization (optional)
    """
    # Create figure with GridSpec
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1])
    
    # Check if we have enough data
    if len(safety_margins) == 0 or len(safety_probs) == 0 or len(positions) <= 1:
        plt.text(0.5, 0.5, 'Not enough data for safety visualization',
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
        
        # Save or show plot
        if output_path:
            plt.savefig(output_path, dpi=300)
            plt.close()
        else:
            plt.show()
        return
    
    # Ensure all arrays have compatible lengths
    min_length = min(len(timestamps), len(safety_margins), len(safety_probs))
    plot_timestamps = timestamps[:min_length]
    plot_safety_margins = safety_margins[:min_length]
    plot_safety_probs = safety_probs[:min_length]
    
    # Plot 1: Safety margin over time
    ax_margin = fig.add_subplot(gs[0, 0])
    ax_margin.plot(plot_timestamps, plot_safety_margins, 'b-')
    ax_margin.axhline(y=threshold, color='r', linestyle='--', label='Safety Threshold')
    
    # Highlight unsafe regions
    unsafe_regions = plot_safety_margins < threshold
    if np.any(unsafe_regions):
        ax_margin.fill_between(
            plot_timestamps,
            plot_safety_margins,
            threshold,
            where=unsafe_regions,
            color='r',
            alpha=0.3
        )
    
    ax_margin.set_xlabel('Time (s)')
    ax_margin.set_ylabel('Safety Margin')
    ax_margin.set_title('Safety Margin over Time')
    ax_margin.legend()
    ax_margin.grid(True)
    
    # Plot 2: Safety probability over time
    ax_prob = fig.add_subplot(gs[1, 0], sharex=ax_margin)
    ax_prob.plot(plot_timestamps, plot_safety_probs, 'g-')
    ax_prob.set_ylim(0, 1)
    
    # Highlight unsafe regions
    if np.any(unsafe_regions):
        ax_prob.fill_between(
            plot_timestamps,
            0,
            plot_safety_probs,
            where=unsafe_regions,
            color='r',
            alpha=0.3
        )
    
    # Highlight uncertain regions (0.05 < P(safe) < 0.95)
    uncertain_regions = (plot_safety_probs > 0.05) & (plot_safety_probs < 0.95)
    if np.any(uncertain_regions):
        ax_prob.fill_between(
            plot_timestamps,
            0,
            plot_safety_probs,
            where=uncertain_regions,
            color='y',
            alpha=0.3
        )
    
    ax_prob.set_xlabel('Time (s)')
    ax_prob.set_ylabel('P(Safe)')
    ax_prob.set_title('Safety Probability over Time')
    ax_prob.grid(True)
    
    # Plot 3: 2D trajectory with safety coloring
    ax_traj = fig.add_subplot(gs[:, 1])
    
    # Create colormap for safety probability
    cmap = plt.cm.get_cmap('RdYlGn')
    norm = plt.Normalize(vmin=0, vmax=1)
    
    # Plot trajectory with color based on safety probability
    # Ensure we don't go out of bounds with safety_probs
    for i in range(min(len(positions) - 1, len(plot_safety_probs))):
        ax_traj.plot(
            [positions[i, 0], positions[i+1, 0]],
            [positions[i, 1], positions[i+1, 1]],
            color=cmap(norm(plot_safety_probs[i])),
            linewidth=2
        )
    
    # Plot obstacles if provided
    if obstacle_positions is not None and obstacle_sizes is not None:
        for pos, size in zip(obstacle_positions, obstacle_sizes):
            # Create rectangle for obstacle
            rect = Rectangle(
                (pos[0] - size[0]/2, pos[1] - size[1]/2),
                size[0], size[1],
                color='gray', alpha=0.5
            )
            ax_traj.add_patch(rect)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_traj)
    cbar.set_label('Safety Probability')
    
    # Add current position marker
    ax_traj.plot(positions[-1, 0], positions[-1, 1], 'ko', markersize=8, label='Current Position')
    
    # Calculate and display distance to nearest obstacle
    if obstacle_positions is not None and len(obstacle_positions) > 0:
        current_pos = positions[-1]
        distances = []
        
        for pos, size in zip(obstacle_positions, obstacle_sizes):
            # Calculate distance to obstacle surface
            dx = max(0, abs(current_pos[0] - pos[0]) - size[0]/2)
            dy = max(0, abs(current_pos[1] - pos[1]) - size[1]/2)
            dz = max(0, abs(current_pos[2] - pos[2]) - size[2]/2)
            
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            distances.append(distance)
        
        if distances:  # Check if distances list is not empty
            min_distance = min(distances)
            min_idx = np.argmin(distances)
            
            # Add distance text
            ax_traj.text(
                0.05, 0.05,
                f'Distance: {min_distance:.2f} m',
                transform=ax_traj.transAxes,
                bbox=dict(facecolor='white', alpha=0.7)
            )
            
            # Draw line to nearest obstacle
            nearest_obs = obstacle_positions[min_idx]
            ax_traj.plot([current_pos[0], nearest_obs[0]], [current_pos[1], nearest_obs[1]], 'k--', alpha=0.5)
    
    ax_traj.set_xlabel('X (m)')
    ax_traj.set_ylabel('Y (m)')
    ax_traj.set_title('Trajectory with Safety Probability')
    ax_traj.legend()
    ax_traj.grid(True)
    ax_traj.axis('equal')
    
    # Set overall title
    fig.suptitle('UAV Safety Analysis', fontsize=16)
    
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
    print("Enhanced Visualization for UAV Safety Digital Twin")
    print("This module is not meant to be run directly.")
    print("Import and use the visualization functions in your scripts.")
