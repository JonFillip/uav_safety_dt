import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import CNNModel

def feature_based_error_analysis():
    print("--- Starting Feature-Based Error Analysis ---")

    # --- 1. Load the Model and Data ---
    model_path = 'models/tuned_autoencoder.keras'
    eval_file = 'datasets/test1_dataset.csv'
    threshold = -0.15
    target_label = 'unsafe'

    print(f"\nLoading model: {model_path}")
    model_wrapper = CNNModel()
    model_wrapper.load(model_path)
    
    print(f"Loading data: {eval_file}")
    test_df = model_wrapper.extract_dataset(eval_file)

    # --- 2. Generate Predictions and Anomaly Scores ---
    print("Generating predictions and anomaly scores...")
    pred_df = model_wrapper.predict_encoder(test_df, inputs=["r_zero"])
    pred_df['anomaly_score'] = pred_df['mean_loss_4'] # Using the negative loss

    # --- 3. Identify TP, FP, and FN on a Per-Log Basis ---
    pred_df['predicted_positive'] = pred_df['anomaly_score'] < threshold
    
    log_labels = pred_df.groupby(['log_folder', 'log_name'])[target_label].any()
    log_predictions = pred_df.groupby(['log_folder', 'log_name'])['predicted_positive'].any()
    
    comparison_df = pd.DataFrame({'actual': log_labels, 'predicted': log_predictions})
    
    # Isolate the names of the logs for each category
    tp_logs = comparison_df[(comparison_df['actual'] == True) & (comparison_df['predicted'] == True)].index
    fp_logs = comparison_df[(comparison_df['actual'] == False) & (comparison_df['predicted'] == True)].index
    fn_logs = comparison_df[(comparison_df['actual'] == True) & (comparison_df['predicted'] == False)].index

    # Get the anomaly scores for all windows belonging to those logs
    tp_scores = pred_df[pred_df.set_index(['log_folder', 'log_name']).index.isin(tp_logs)]['anomaly_score']
    fp_scores = pred_df[pred_df.set_index(['log_folder', 'log_name']).index.isin(fp_logs)]['anomaly_score']
    fn_scores = pred_df[pred_df.set_index(['log_folder', 'log_name']).index.isin(fn_logs)]['anomaly_score']
    
    # --- 4. Perform Visual Analysis ---
    print("\n--- Plotting Anomaly Score Distributions ---")
    plt.figure(figsize=(12, 7))
    plt.hist(tp_scores, bins=50, alpha=0.7, label='True Positives (Unsafe, Correctly Flagged)', color='green')
    plt.hist(fp_scores, bins=50, alpha=0.7, label='False Positives (Safe, Incorrectly Flagged)', color='orange')
    plt.hist(fn_scores, bins=50, alpha=0.7, label='False Negatives (Unsafe, Missed)', color='red')
    
    plt.axvline(threshold, color='black', linestyle='dashed', linewidth=2, label=f'Threshold ({threshold})')
    
    plt.title('Distribution of Anomaly Scores for Prediction Outcomes (test1_dataset)')
    plt.xlabel('Anomaly Score (Lower is More Anomalous)')
    plt.ylabel('Number of Time Windows')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

    # --- 5. Print Descriptive Statistics ---
    print("\n--- Descriptive Statistics for Anomaly Scores ---")
    print("\n--- True Positives ---")
    print(tp_scores.describe())
    print("\n--- False Positives ---")
    print(fp_scores.describe())
    print("\n--- False Negatives ---")
    print(fn_scores.describe())

if __name__ == "__main__":
    feature_based_error_analysis()