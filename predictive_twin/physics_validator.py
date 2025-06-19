#!/usr/bin/env python3
"""
Physics Validator for UAV Safety Digital Twin

This module implements physics-based validation of neural network predictions for the
UAV Safety Digital Twin. It includes:
- Feature attribution techniques to connect neural network activations to physical quantities
- Physics-based constraints to flag predictions that violate physical constraints
- Explainability layer to provide human-readable explanations for safety decisions
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import shap
import lime
import lime.lime_tabular

# Add the project root to the path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PhysicsValidator:
    """
    Class for physics-based validation of neural network predictions.
    
    This class implements various methods for validating neural network predictions
    using physics-based constraints and providing explanations for safety decisions.
    """
    
    def __init__(self, max_acceleration=10.0, max_velocity=20.0, min_obstacle_distance=0.5):
        """
        Initialize the physics validator.
        
        Args:
            max_acceleration: Maximum physically possible acceleration (m/s^2)
            max_velocity: Maximum physically possible velocity (m/s)
            min_obstacle_distance: Minimum physically possible obstacle distance (m)
        """
        self.max_acceleration = max_acceleration
        self.max_velocity = max_velocity
        self.min_obstacle_distance = min_obstacle_distance
        self.explainer = None
    
    def validate_kinematics(self, positions, timestamps, max_jerk=None):
        """
        Validate kinematic constraints on position data.
        
        Args:
            positions: Array of positions [x, y, z]
            timestamps: Array of timestamps
            max_jerk: Maximum physically possible jerk (m/s^3) (optional)
            
        Returns:
            Dictionary with validation results
        """
        # Calculate time differences
        dt = np.diff(timestamps)
        
        # Calculate velocities
        velocities = np.diff(positions, axis=0) / dt[:, np.newaxis]
        
        # Calculate accelerations
        accelerations = np.diff(velocities, axis=0) / dt[1:][:, np.newaxis]
        
        # Calculate jerks if max_jerk is provided
        if max_jerk is not None:
            jerks = np.diff(accelerations, axis=0) / dt[2:][:, np.newaxis]
            jerk_magnitudes = np.linalg.norm(jerks, axis=1)
        
        # Calculate magnitudes
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
        
        # Check constraints
        velocity_violations = velocity_magnitudes > self.max_velocity
        acceleration_violations = acceleration_magnitudes > self.max_acceleration
        
        if max_jerk is not None:
            jerk_violations = jerk_magnitudes > max_jerk
        else:
            jerk_violations = np.zeros_like(acceleration_violations)
        
        # Create result dictionary
        result = {
            'velocity_magnitudes': velocity_magnitudes,
            'acceleration_magnitudes': acceleration_magnitudes,
            'velocity_violations': velocity_violations,
            'acceleration_violations': acceleration_violations,
            'jerk_violations': jerk_violations if max_jerk is not None else None,
            'max_velocity': np.max(velocity_magnitudes),
            'max_acceleration': np.max(acceleration_magnitudes),
            'max_jerk': np.max(jerk_magnitudes) if max_jerk is not None else None,
            'any_violations': np.any(velocity_violations) or np.any(acceleration_violations) or 
                             (np.any(jerk_violations) if max_jerk is not None else False)
        }
        
        return result
    
    def validate_obstacle_distance(self, positions, obstacle_positions, obstacle_sizes):
        """
        Validate obstacle distance constraints.
        
        Args:
            positions: Array of positions [x, y, z]
            obstacle_positions: Array of obstacle positions [x, y, z]
            obstacle_sizes: Array of obstacle sizes [width, height, depth]
            
        Returns:
            Dictionary with validation results
        """
        # Calculate distances to obstacles
        distances = []
        violations = []
        
        for i, position in enumerate(positions):
            min_distance = float('inf')
            violation = False
            
            for j, obstacle_position in enumerate(obstacle_positions):
                # Calculate distance to obstacle
                obstacle_size = obstacle_sizes[j]
                
                # Calculate distance to obstacle surface
                dx = max(0, abs(position[0] - obstacle_position[0]) - obstacle_size[0] / 2)
                dy = max(0, abs(position[1] - obstacle_position[1]) - obstacle_size[1] / 2)
                dz = max(0, abs(position[2] - obstacle_position[2]) - obstacle_size[2] / 2)
                
                distance = np.sqrt(dx**2 + dy**2 + dz**2)
                
                if distance < min_distance:
                    min_distance = distance
                
                if distance < self.min_obstacle_distance:
                    violation = True
            
            distances.append(min_distance)
            violations.append(violation)
        
        # Create result dictionary
        result = {
            'distances': np.array(distances),
            'violations': np.array(violations),
            'min_distance': np.min(distances),
            'any_violations': np.any(violations)
        }
        
        return result
    
    def validate_trajectory(self, positions, timestamps, obstacle_positions, obstacle_sizes, max_jerk=None):
        """
        Validate a complete trajectory.
        
        Args:
            positions: Array of positions [x, y, z]
            timestamps: Array of timestamps
            obstacle_positions: Array of obstacle positions [x, y, z]
            obstacle_sizes: Array of obstacle sizes [width, height, depth]
            max_jerk: Maximum physically possible jerk (m/s^3) (optional)
            
        Returns:
            Dictionary with validation results
        """
        # Validate kinematics
        kinematics_result = self.validate_kinematics(positions, timestamps, max_jerk)
        
        # Validate obstacle distances
        obstacle_result = self.validate_obstacle_distance(positions, obstacle_positions, obstacle_sizes)
        
        # Combine results
        result = {
            'kinematics': kinematics_result,
            'obstacles': obstacle_result,
            'any_violations': kinematics_result['any_violations'] or obstacle_result['any_violations']
        }
        
        return result
    
    def validate_predictions(self, current_state, predicted_states, timestamps, 
                           obstacle_positions, obstacle_sizes, max_jerk=None):
        """
        Validate predicted states.
        
        Args:
            current_state: Current state [position, velocity, acceleration]
            predicted_states: Predicted states [position, velocity, acceleration]
            timestamps: Array of timestamps
            obstacle_positions: Array of obstacle positions [x, y, z]
            obstacle_sizes: Array of obstacle sizes [width, height, depth]
            max_jerk: Maximum physically possible jerk (m/s^3) (optional)
            
        Returns:
            Dictionary with validation results
        """
        # Extract positions
        current_position = current_state[0]
        predicted_positions = [state[0] for state in predicted_states]
        
        # Combine positions
        positions = np.vstack([current_position[np.newaxis, :], predicted_positions])
        
        # Validate trajectory
        result = self.validate_trajectory(positions, timestamps, obstacle_positions, obstacle_sizes, max_jerk)
        
        return result
    
    def create_shap_explainer(self, model, background_data):
        """
        Create a SHAP explainer for the model.
        
        Args:
            model: Model to explain
            background_data: Background data for SHAP explainer
            
        Returns:
            SHAP explainer
        """
        # Create SHAP explainer
        self.explainer = shap.KernelExplainer(model, background_data)
        
        return self.explainer
    
    def explain_prediction(self, model, input_data, feature_names=None):
        """
        Explain a prediction using SHAP values.
        
        Args:
            model: Model to explain
            input_data: Input data to explain
            feature_names: Names of features (optional)
            
        Returns:
            Dictionary with explanation
        """
        if self.explainer is None:
            raise ValueError("SHAP explainer not created. Call create_shap_explainer() first.")
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(input_data)
        
        # Create explanation
        explanation = {
            'shap_values': shap_values,
            'base_value': self.explainer.expected_value,
            'feature_names': feature_names
        }
        
        return explanation
    
    def create_lime_explainer(self, feature_names, categorical_features=None):
        """
        Create a LIME explainer.
        
        Args:
            feature_names: Names of features
            categorical_features: Indices of categorical features (optional)
            
        Returns:
            LIME explainer
        """
        # Create LIME explainer
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            np.zeros((1, len(feature_names))),  # Dummy training data
            feature_names=feature_names,
            categorical_features=categorical_features,
            verbose=False,
            mode='regression'
        )
        
        return self.lime_explainer
    
    def explain_with_lime(self, model, input_data, num_features=5):
        """
        Explain a prediction using LIME.
        
        Args:
            model: Model to explain
            input_data: Input data to explain
            num_features: Number of features to include in explanation
            
        Returns:
            LIME explanation
        """
        if not hasattr(self, 'lime_explainer'):
            raise ValueError("LIME explainer not created. Call create_lime_explainer() first.")
        
        # Create prediction function
        def predict_fn(x):
            return model(x)
        
        # Generate explanation
        explanation = self.lime_explainer.explain_instance(
            input_data, predict_fn, num_features=num_features
        )
        
        return explanation
    
    def generate_text_explanation(self, validation_result, explanation=None, feature_names=None):
        """
        Generate a human-readable explanation for validation results.
        
        Args:
            validation_result: Validation result from validate_predictions
            explanation: Explanation from explain_prediction (optional)
            feature_names: Names of features (optional)
            
        Returns:
            Human-readable explanation
        """
        # Start with a summary
        if validation_result['any_violations']:
            text = "The predicted trajectory violates physical constraints:\n\n"
        else:
            text = "The predicted trajectory satisfies all physical constraints:\n\n"
        
        # Add kinematic violations
        kinematics = validation_result['kinematics']
        if kinematics['any_violations']:
            text += "Kinematic violations:\n"
            
            if np.any(kinematics['velocity_violations']):
                text += f"- Maximum velocity ({kinematics['max_velocity']:.2f} m/s) exceeds physical limit ({self.max_velocity:.2f} m/s)\n"
            
            if np.any(kinematics['acceleration_violations']):
                text += f"- Maximum acceleration ({kinematics['max_acceleration']:.2f} m/s²) exceeds physical limit ({self.max_acceleration:.2f} m/s²)\n"
            
            if kinematics['jerk_violations'] is not None and np.any(kinematics['jerk_violations']):
                text += f"- Maximum jerk ({kinematics['max_jerk']:.2f} m/s³) exceeds physical limit\n"
        else:
            text += "Kinematic constraints satisfied:\n"
            text += f"- Maximum velocity: {kinematics['max_velocity']:.2f} m/s (limit: {self.max_velocity:.2f} m/s)\n"
            text += f"- Maximum acceleration: {kinematics['max_acceleration']:.2f} m/s² (limit: {self.max_acceleration:.2f} m/s²)\n"
            
            if kinematics['max_jerk'] is not None:
                text += f"- Maximum jerk: {kinematics['max_jerk']:.2f} m/s³\n"
        
        # Add obstacle violations
        obstacles = validation_result['obstacles']
        if obstacles['any_violations']:
            text += "\nObstacle violations:\n"
            text += f"- Minimum distance to obstacle ({obstacles['min_distance']:.2f} m) is less than safety threshold ({self.min_obstacle_distance:.2f} m)\n"
        else:
            text += "\nObstacle constraints satisfied:\n"
            text += f"- Minimum distance to obstacle: {obstacles['min_distance']:.2f} m (threshold: {self.min_obstacle_distance:.2f} m)\n"
        
        # Add explanation if provided
        if explanation is not None and feature_names is not None:
            text += "\nKey factors influencing the prediction:\n"
            
            # Sort features by absolute SHAP value
            shap_values = explanation['shap_values']
            if isinstance(shap_values, list):
                # For multi-output models, use the first output
                shap_values = shap_values[0]
            
            # Get absolute SHAP values
            abs_shap_values = np.abs(shap_values)
            
            # Sort features by absolute SHAP value
            sorted_idx = np.argsort(abs_shap_values)[::-1]
            
            # Add top 5 features
            for i in range(min(5, len(sorted_idx))):
                idx = sorted_idx[i]
                feature_name = feature_names[idx]
                shap_value = shap_values[idx]
                
                if shap_value > 0:
                    text += f"- {feature_name}: Increases risk by {shap_value:.4f}\n"
                else:
                    text += f"- {feature_name}: Decreases risk by {-shap_value:.4f}\n"
        
        return text
    
    def visualize_validation(self, positions, timestamps, validation_result, output_path=None):
        """
        Visualize validation results.
        
        Args:
            positions: Array of positions [x, y, z]
            timestamps: Array of timestamps
            validation_result: Validation result from validate_predictions
            output_path: Path to save visualization (optional)
        """
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Plot 1: Position trajectory
        ax = axes[0]
        
        # Plot positions
        ax.plot(timestamps, positions[:, 0], 'r-', label='X')
        ax.plot(timestamps, positions[:, 1], 'g-', label='Y')
        ax.plot(timestamps, positions[:, 2], 'b-', label='Z')
        
        ax.set_ylabel('Position (m)')
        ax.set_title('Position Trajectory')
        ax.legend()
        
        # Plot 2: Velocity and acceleration
        ax = axes[1]
        
        # Calculate velocities and accelerations
        dt = np.diff(timestamps)
        velocities = np.diff(positions, axis=0) / dt[:, np.newaxis]
        accelerations = np.diff(velocities, axis=0) / dt[1:][:, np.newaxis]
        
        # Calculate magnitudes
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
        
        # Plot velocity magnitudes
        ax.plot(timestamps[1:], velocity_magnitudes, 'b-', label='Velocity')
        
        # Add velocity limit
        ax.axhline(y=self.max_velocity, color='b', linestyle='--', label=f'Max Velocity ({self.max_velocity} m/s)')
        
        # Highlight velocity violations
        velocity_violations = validation_result['kinematics']['velocity_violations']
        if np.any(velocity_violations):
            ax.fill_between(
                timestamps[1:],
                velocity_magnitudes,
                self.max_velocity,
                where=velocity_violations,
                color='b',
                alpha=0.3
            )
        
        # Create second y-axis for acceleration
        ax2 = ax.twinx()
        
        # Plot acceleration magnitudes
        ax2.plot(timestamps[2:], acceleration_magnitudes, 'r-', label='Acceleration')
        
        # Add acceleration limit
        ax2.axhline(y=self.max_acceleration, color='r', linestyle='--', label=f'Max Acceleration ({self.max_acceleration} m/s²)')
        
        # Highlight acceleration violations
        acceleration_violations = validation_result['kinematics']['acceleration_violations']
        if np.any(acceleration_violations):
            ax2.fill_between(
                timestamps[2:],
                acceleration_magnitudes,
                self.max_acceleration,
                where=acceleration_violations,
                color='r',
                alpha=0.3
            )
        
        ax.set_ylabel('Velocity (m/s)')
        ax2.set_ylabel('Acceleration (m/s²)')
        ax.set_title('Velocity and Acceleration')
        
        # Add legends for both y-axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Plot 3: Obstacle distances
        ax = axes[2]
        
        # Plot obstacle distances
        distances = validation_result['obstacles']['distances']
        ax.plot(timestamps, distances, 'g-', label='Obstacle Distance')
        
        # Add distance threshold
        ax.axhline(y=self.min_obstacle_distance, color='r', linestyle='--', label=f'Min Distance ({self.min_obstacle_distance} m)')
        
        # Highlight distance violations
        distance_violations = validation_result['obstacles']['violations']
        if np.any(distance_violations):
            ax.fill_between(
                timestamps,
                distances,
                self.min_obstacle_distance,
                where=distance_violations,
                color='r',
                alpha=0.3
            )
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance (m)')
        ax.set_title('Distance to Nearest Obstacle')
        ax.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show plot
        if output_path:
            plt.savefig(output_path)
        
        plt.close()
    
    def visualize_explanation(self, explanation, feature_names, output_path=None):
        """
        Visualize SHAP explanation.
        
        Args:
            explanation: Explanation from explain_prediction
            feature_names: Names of features
            output_path: Path to save visualization (optional)
        """
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Extract SHAP values
        shap_values = explanation['shap_values']
        if isinstance(shap_values, list):
            # For multi-output models, use the first output
            shap_values = shap_values[0]
        
        # Sort features by absolute SHAP value
        abs_shap_values = np.abs(shap_values)
        sorted_idx = np.argsort(abs_shap_values)
        
        # Plot SHAP values
        plt.barh(
            np.array(feature_names)[sorted_idx],
            shap_values[sorted_idx],
            color=['r' if x > 0 else 'b' for x in shap_values[sorted_idx]]
        )
        
        plt.xlabel('SHAP Value (Impact on Prediction)')
        plt.title('Feature Importance')
        
        # Add base value
        plt.axvline(x=0, color='k', linestyle='--')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show plot
        if output_path:
            plt.savefig(output_path)
        
        plt.close()


def create_feature_attribution_visualization(feature_names, feature_values, importances, output_path=None):
    """
    Create a visualization of feature attributions.
    
    Args:
        feature_names: Names of features
        feature_values: Values of features
        importances: Importance scores for features
        output_path: Path to save visualization (optional)
    """
    # Sort features by absolute importance
    abs_importances = np.abs(importances)
    sorted_idx = np.argsort(abs_importances)[::-1]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot feature importances
    bars = plt.barh(
        np.array(feature_names)[sorted_idx],
        importances[sorted_idx],
        color=['r' if x > 0 else 'b' for x in importances[sorted_idx]]
    )
    
    # Add feature values as text
    for i, bar in enumerate(bars):
        idx = sorted_idx[i]
        value = feature_values[idx]
        
        # Format value based on magnitude
        if abs(value) < 0.01:
            value_str = f"{value:.2e}"
        else:
            value_str = f"{value:.2f}"
        
        # Add text at the end of the bar
        if importances[idx] > 0:
            plt.text(
                importances[idx] + 0.01 * max(abs_importances),
                bar.get_y() + bar.get_height() / 2,
                value_str,
                va='center'
            )
        else:
            plt.text(
                importances[idx] - 0.05 * max(abs_importances),
                bar.get_y() + bar.get_height() / 2,
                value_str,
                va='center',
                ha='right'
            )
    
    plt.xlabel('Feature Importance')
    plt.title('Feature Attribution')
    
    # Add zero line
    plt.axvline(x=0, color='k', linestyle='--')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path)
    
    plt.close()


if __name__ == '__main__':
    # Example usage
    print("Physics Validator for UAV Safety Digital Twin")
    print("This module is not meant to be run directly.")
    print("Import and use the PhysicsValidator class in your scripts.")
