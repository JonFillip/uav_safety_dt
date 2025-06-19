#!/usr/bin/env python3
"""
Uncertainty Quantification for UAV Safety Digital Twin

This module implements advanced methods for quantifying prediction uncertainty in the
UAV Safety Digital Twin. It includes:
- Monte Carlo dropout for uncertainty estimation
- Ensemble methods for uncertainty quantification
- Calibration techniques for uncertainty estimates
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
import matplotlib.pyplot as plt
from scipy.stats import norm, beta
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

# Add the project root to the path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the predictive autoencoder
from predictive_twin.predictive_autoencoder import PredictiveAutoencoder


class UncertaintyQuantifier:
    """
    Class for quantifying uncertainty in predictive models.
    
    This class implements various methods for quantifying uncertainty in predictions,
    including Monte Carlo dropout, ensemble methods, and calibration techniques.
    """
    
    def __init__(self, models=None, dropout_rate=0.2, n_samples=30):
        """
        Initialize the uncertainty quantifier.
        
        Args:
            models: List of models for ensemble methods (optional)
            dropout_rate: Dropout rate for Monte Carlo dropout (default: 0.2)
            n_samples: Number of Monte Carlo samples (default: 30)
        """
        self.models = models if models is not None else []
        self.dropout_rate = dropout_rate
        self.n_samples = n_samples
        self.calibration_model = None
    
    def add_model(self, model):
        """
        Add a model to the ensemble.
        
        Args:
            model: Model to add
        """
        self.models.append(model)
    
    def monte_carlo_dropout(self, model, data, training=True):
        """
        Perform Monte Carlo dropout for uncertainty estimation.
        
        Args:
            model: Model with dropout layers
            data: Input data
            training: Whether to run in training mode (enables dropout)
            
        Returns:
            Tuple of (mean_predictions, variance_predictions)
        """
        # Enable dropout during inference
        predictions = []
        
        # Create a function that runs the model in training mode
        if isinstance(model, PredictiveAutoencoder):
            # For PredictiveAutoencoder, use the model's predict method with mc_dropout=True
            (current_mean, future_mean), (current_var, future_var) = model.predict(
                data, mc_dropout=True, n_samples=self.n_samples
            )
            return (current_mean, future_mean), (current_var, future_var)
        else:
            # For other models, manually run multiple forward passes
            for _ in range(self.n_samples):
                pred = model(data, training=training)
                predictions.append(pred)
            
            # Calculate mean and variance
            mean_pred = tf.reduce_mean(predictions, axis=0)
            var_pred = tf.math.reduce_variance(predictions, axis=0)
            
            return mean_pred, var_pred
    
    def ensemble_prediction(self, data):
        """
        Generate predictions using an ensemble of models.
        
        Args:
            data: Input data
            
        Returns:
            Tuple of (mean_predictions, variance_predictions)
        """
        if not self.models:
            raise ValueError("No models in ensemble. Add models using add_model().")
        
        # Get predictions from each model
        predictions = []
        
        for model in self.models:
            if isinstance(model, PredictiveAutoencoder):
                # For PredictiveAutoencoder, use the model's predict method
                current_pred, future_pred = model.predict(data)
                predictions.append((current_pred, future_pred))
            else:
                # For other models, use the model directly
                pred = model(data)
                predictions.append(pred)
        
        # Calculate mean and variance
        if isinstance(self.models[0], PredictiveAutoencoder):
            # For PredictiveAutoencoder, handle current and future predictions separately
            current_preds = [p[0] for p in predictions]
            future_preds = [p[1] for p in predictions]
            
            current_mean = np.mean(current_preds, axis=0)
            current_var = np.var(current_preds, axis=0)
            
            future_mean = np.mean(future_preds, axis=0)
            future_var = np.var(future_preds, axis=0)
            
            return (current_mean, future_mean), (current_var, future_var)
        else:
            # For other models, calculate mean and variance directly
            mean_pred = np.mean(predictions, axis=0)
            var_pred = np.var(predictions, axis=0)
            
            return mean_pred, var_pred
    
    def calibrate_uncertainty(self, true_values, predicted_values, predicted_uncertainties, method='isotonic'):
        """
        Calibrate uncertainty estimates.
        
        Args:
            true_values: Ground truth values
            predicted_values: Predicted values
            predicted_uncertainties: Predicted uncertainties (variances)
            method: Calibration method ('isotonic' or 'temperature')
            
        Returns:
            Calibrated uncertainties
        """
        if method == 'isotonic':
            # Use isotonic regression to calibrate uncertainties
            # First, calculate normalized errors
            normalized_errors = np.abs(true_values - predicted_values) / np.sqrt(predicted_uncertainties)
            
            # Fit isotonic regression
            self.calibration_model = IsotonicRegression(out_of_bounds='clip')
            self.calibration_model.fit(predicted_uncertainties, normalized_errors**2)
            
            # Calibrate uncertainties
            calibrated_uncertainties = self.calibration_model.predict(predicted_uncertainties)
            
            return calibrated_uncertainties
        
        elif method == 'temperature':
            # Use temperature scaling to calibrate uncertainties
            # This is more suitable for classification problems
            # For regression, we use a simple scaling factor
            
            # Calculate normalized errors
            normalized_errors = np.abs(true_values - predicted_values) / np.sqrt(predicted_uncertainties)
            
            # Find optimal temperature (scaling factor)
            temperature = np.mean(normalized_errors**2)
            
            # Calibrate uncertainties
            calibrated_uncertainties = predicted_uncertainties * temperature
            
            # Store temperature for future use
            self.calibration_model = temperature
            
            return calibrated_uncertainties
        
        else:
            raise ValueError(f"Unknown calibration method: {method}")
    
    def apply_calibration(self, predicted_uncertainties):
        """
        Apply calibration to new uncertainty estimates.
        
        Args:
            predicted_uncertainties: Predicted uncertainties (variances)
            
        Returns:
            Calibrated uncertainties
        """
        if self.calibration_model is None:
            raise ValueError("Calibration model not trained. Call calibrate_uncertainty() first.")
        
        if isinstance(self.calibration_model, IsotonicRegression):
            # Apply isotonic regression
            return self.calibration_model.predict(predicted_uncertainties)
        else:
            # Apply temperature scaling
            return predicted_uncertainties * self.calibration_model
    
    def confidence_intervals(self, predicted_values, predicted_uncertainties, confidence_level=0.95):
        """
        Calculate confidence intervals for predictions.
        
        Args:
            predicted_values: Predicted values
            predicted_uncertainties: Predicted uncertainties (variances)
            confidence_level: Confidence level (default: 0.95)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Calculate z-score for the given confidence level
        z_score = norm.ppf((1 + confidence_level) / 2)
        
        # Calculate standard deviation from variance
        predicted_std = np.sqrt(predicted_uncertainties)
        
        # Calculate confidence intervals
        lower_bound = predicted_values - z_score * predicted_std
        upper_bound = predicted_values + z_score * predicted_std
        
        return lower_bound, upper_bound
    
    def prediction_intervals(self, predicted_values, predicted_uncertainties, confidence_level=0.95, df=None):
        """
        Calculate prediction intervals for predictions.
        
        Args:
            predicted_values: Predicted values
            predicted_uncertainties: Predicted uncertainties (variances)
            confidence_level: Confidence level (default: 0.95)
            df: Degrees of freedom (default: None, use normal distribution)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if df is None:
            # Use normal distribution (same as confidence_intervals)
            return self.confidence_intervals(predicted_values, predicted_uncertainties, confidence_level)
        else:
            # Use t-distribution with df degrees of freedom
            from scipy.stats import t
            t_score = t.ppf((1 + confidence_level) / 2, df)
            
            # Calculate standard deviation from variance
            predicted_std = np.sqrt(predicted_uncertainties)
            
            # Calculate prediction intervals
            lower_bound = predicted_values - t_score * predicted_std
            upper_bound = predicted_values + t_score * predicted_std
            
            return lower_bound, upper_bound
    
    def visualize_uncertainty(self, true_values, predicted_values, predicted_uncertainties, 
                             calibrated_uncertainties=None, output_path=None):
        """
        Visualize uncertainty estimates.
        
        Args:
            true_values: Ground truth values
            predicted_values: Predicted values
            predicted_uncertainties: Predicted uncertainties (variances)
            calibrated_uncertainties: Calibrated uncertainties (optional)
            output_path: Path to save visualization (optional)
        """
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: True vs Predicted values with uncertainty
        ax = axes[0, 0]
        
        # Calculate standard deviation from variance
        predicted_std = np.sqrt(predicted_uncertainties)
        
        # Calculate confidence intervals
        lower_bound, upper_bound = self.confidence_intervals(predicted_values, predicted_uncertainties)
        
        # Sort by true values for better visualization
        sort_idx = np.argsort(true_values)
        true_sorted = true_values[sort_idx]
        pred_sorted = predicted_values[sort_idx]
        lower_sorted = lower_bound[sort_idx]
        upper_sorted = upper_bound[sort_idx]
        
        # Plot true vs predicted with confidence intervals
        ax.plot(true_sorted, pred_sorted, 'b.', alpha=0.5, label='Predictions')
        ax.fill_between(true_sorted, lower_sorted, upper_sorted, color='b', alpha=0.2, label='95% CI')
        
        # Plot perfect prediction line
        min_val = min(true_values.min(), predicted_values.min())
        max_val = max(true_values.max(), predicted_values.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('True vs Predicted Values with Uncertainty')
        ax.legend()
        
        # Plot 2: Error vs Uncertainty
        ax = axes[0, 1]
        
        # Calculate errors
        errors = np.abs(true_values - predicted_values)
        
        # Plot errors vs uncertainties
        ax.scatter(predicted_std, errors, alpha=0.5)
        
        # Plot y=x line for reference
        max_val = max(predicted_std.max(), errors.max())
        ax.plot([0, max_val], [0, max_val], 'r--', label='y=x')
        
        ax.set_xlabel('Predicted Uncertainty (Std)')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Error vs Uncertainty')
        ax.legend()
        
        # Plot 3: Calibration curve
        ax = axes[1, 0]
        
        # Calculate normalized errors
        normalized_errors = errors / predicted_std
        
        # Create bins for normalized errors
        bins = np.linspace(0, 3, 10)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Calculate fraction of errors in each bin
        bin_indices = np.digitize(normalized_errors, bins) - 1
        bin_counts = np.bincount(bin_indices, minlength=len(bins)-1)
        bin_fractions = bin_counts / len(normalized_errors)
        
        # Plot calibration curve
        ax.bar(bin_centers, bin_fractions, width=bins[1]-bins[0], alpha=0.5)
        
        # Plot expected distribution (chi-squared with 1 degree of freedom)
        x = np.linspace(0, 3, 100)
        y = np.exp(-x**2/2) / np.sqrt(2*np.pi)
        ax.plot(x, y, 'r-', label='Expected')
        
        ax.set_xlabel('Normalized Error (|y-ŷ|/σ)')
        ax.set_ylabel('Fraction of Predictions')
        ax.set_title('Calibration Curve')
        ax.legend()
        
        # Plot 4: Calibration comparison (if calibrated uncertainties provided)
        ax = axes[1, 1]
        
        if calibrated_uncertainties is not None:
            # Calculate calibrated standard deviation
            calibrated_std = np.sqrt(calibrated_uncertainties)
            
            # Calculate normalized errors with calibrated uncertainties
            calibrated_normalized_errors = errors / calibrated_std
            
            # Calculate fraction of errors in each bin for calibrated uncertainties
            calibrated_bin_indices = np.digitize(calibrated_normalized_errors, bins) - 1
            calibrated_bin_counts = np.bincount(calibrated_bin_indices, minlength=len(bins)-1)
            calibrated_bin_fractions = calibrated_bin_counts / len(calibrated_normalized_errors)
            
            # Plot calibration curves
            ax.bar(bin_centers - 0.1, bin_fractions, width=(bins[1]-bins[0])/2, alpha=0.5, label='Original')
            ax.bar(bin_centers + 0.1, calibrated_bin_fractions, width=(bins[1]-bins[0])/2, alpha=0.5, label='Calibrated')
            
            # Plot expected distribution
            ax.plot(x, y, 'r-', label='Expected')
            
            ax.set_xlabel('Normalized Error')
            ax.set_ylabel('Fraction of Predictions')
            ax.set_title('Calibration Comparison')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No calibrated uncertainties provided', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
        
        # Adjust layout and save/show plot
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
        
        plt.close()
    
    def decompose_uncertainty(self, ensemble_var, mc_dropout_var):
        """
        Decompose uncertainty into epistemic and aleatoric components.
        
        Args:
            ensemble_var: Variance from ensemble predictions (epistemic uncertainty)
            mc_dropout_var: Variance from Monte Carlo dropout (total uncertainty)
            
        Returns:
            Tuple of (epistemic_uncertainty, aleatoric_uncertainty)
        """
        # Epistemic uncertainty is captured by the ensemble variance
        epistemic_uncertainty = ensemble_var
        
        # Aleatoric uncertainty is the difference between total and epistemic uncertainty
        # (assuming they are independent)
        aleatoric_uncertainty = np.maximum(0, mc_dropout_var - ensemble_var)
        
        return epistemic_uncertainty, aleatoric_uncertainty
    
    def safety_probability(self, safety_margin, uncertainty, threshold=0):
        """
        Calculate the probability of safety given a safety margin and uncertainty.
        
        Args:
            safety_margin: Safety margin (distance to threshold)
            uncertainty: Uncertainty in the safety margin
            threshold: Safety threshold (default: 0)
            
        Returns:
            Probability of safety
        """
        # Calculate the probability that the true safety margin is above the threshold
        # using the normal CDF
        z_score = (safety_margin - threshold) / np.sqrt(uncertainty)
        safety_prob = norm.cdf(z_score)
        
        return safety_prob


def create_uncertainty_visualization(time_steps, true_values, predicted_values, 
                                    lower_bound, upper_bound, output_path=None):
    """
    Create a visualization of predictions with uncertainty.
    
    Args:
        time_steps: Time steps for x-axis
        true_values: Ground truth values
        predicted_values: Predicted values
        lower_bound: Lower bound of prediction interval
        upper_bound: Upper bound of prediction interval
        output_path: Path to save visualization (optional)
    """
    plt.figure(figsize=(10, 6))
    
    # Plot true values
    plt.plot(time_steps, true_values, 'k-', label='True Values')
    
    # Plot predicted values
    plt.plot(time_steps, predicted_values, 'b-', label='Predicted Values')
    
    # Plot prediction intervals
    plt.fill_between(time_steps, lower_bound, upper_bound, color='b', alpha=0.2, label='95% Prediction Interval')
    
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Predictions with Uncertainty')
    plt.legend()
    
    if output_path:
        plt.savefig(output_path)
    
    plt.close()


def create_safety_visualization(time_steps, safety_margin, safety_prob, threshold=0, output_path=None):
    """
    Create a visualization of safety margin and probability.
    
    Args:
        time_steps: Time steps for x-axis
        safety_margin: Safety margin (distance to threshold)
        safety_prob: Probability of safety
        threshold: Safety threshold (default: 0)
        output_path: Path to save visualization (optional)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot safety margin
    ax1.plot(time_steps, safety_margin, 'b-')
    ax1.axhline(y=threshold, color='r', linestyle='--', label='Safety Threshold')
    ax1.set_ylabel('Safety Margin')
    ax1.set_title('Safety Margin and Probability')
    ax1.legend()
    
    # Plot safety probability
    ax2.plot(time_steps, safety_prob, 'g-')
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('P(Safe)')
    
    # Highlight unsafe regions
    unsafe_regions = safety_margin < threshold
    if np.any(unsafe_regions):
        ax1.fill_between(time_steps, safety_margin, threshold, where=unsafe_regions, color='r', alpha=0.3)
        ax2.fill_between(time_steps, 0, safety_prob, where=unsafe_regions, color='r', alpha=0.3)
    
    # Highlight uncertain regions (0.05 < P(safe) < 0.95)
    uncertain_regions = (safety_prob > 0.05) & (safety_prob < 0.95)
    if np.any(uncertain_regions):
        ax2.fill_between(time_steps, 0, safety_prob, where=uncertain_regions, color='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    
    plt.close()


if __name__ == '__main__':
    # Example usage
    print("Uncertainty Quantification for UAV Safety Digital Twin")
    print("This module is not meant to be run directly.")
    print("Import and use the UncertaintyQuantifier class in your scripts.")
