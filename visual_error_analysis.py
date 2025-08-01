import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import CNNModel

def plot_trajectory_from_dataframe(df_row, title=""):
    """
    Generates a simple 2D plot of the flight trajectory using the x and y
    columns from a single row of the pre-processed dataframe.
    """
    plt.figure(figsize=(8, 8))
    
    # The 'x' and 'y' columns contain the list of coordinates for the whole flight
    x_coords = df_row['x']
    y_coords = df_row['y']
    
    plt.plot(x_coords, y_coords, label='Flight Path', color='blue')
    
    # Mark the start and end points
    plt.scatter(x_coords[0], y_coords[0], color='green', s=100, zorder=5, label='Start')
    plt.scatter(x_coords[-1], y_coords[-1], color='red', s=100, zorder=5, label='End')
    
    plt.title(title)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def visual_error_analysis():
    print("--- Starting Visual Error Analysis ---")

    # --- 1. Setup ---
    model_path = 'models/tuned_autoencoder.keras'
    eval_file = 'datasets/test1_dataset.csv'
    threshold = -0.15
    target_label = 'unsafe'

    model_wrapper = CNNModel()
    model_wrapper.load(model_path)
    test_df = model_wrapper.extract_dataset(eval_file)

    # --- 2. Generate Predictions ---
    print("Generating predictions to identify error cases...")
    pred_df = model_wrapper.predict_encoder(test_df, inputs=["r_zero"])
    pred_df['anomaly_score'] = pred_df['mean_loss_4']
    pred_df['predicted_positive'] = pred_df['anomaly_score'] < threshold
    
    # --- 3. Identify FP and FN Logs ---
    log_labels = pred_df.groupby(['log_folder', 'log_name'])[target_label].any()
    log_predictions = pred_df.groupby(['log_folder', 'log_name'])['predicted_positive'].any()
    
    comparison_df = pd.DataFrame({'actual': log_labels, 'predicted': log_predictions})
    
    fp_logs = comparison_df[(comparison_df['actual'] == False) & (comparison_df['predicted'] == True)].index
    fn_logs = comparison_df[(comparison_df['actual'] == True) & (comparison_df['predicted'] == False)].index

    print(f"\nFound {len(fp_logs)} False Positive logs and {len(fn_logs)} False Negative logs.")

    # --- 4. Plot Sample Cases ---

    # Plot up to 2 examples of False Positives
    if not fp_logs.empty:
        print("\n--- Visualizing False Positive Cases (Safe but Flagged as Unsafe) ---")
        for log_id in fp_logs[:2]:
            # Find the first row for this log to get the trajectory data
            log_data_row = pred_df[(pred_df['log_folder'] == log_id[0]) & (pred_df['log_name'] == log_id[1])].iloc[0]
            plot_trajectory_from_dataframe(log_data_row, title=f"False Positive: {log_id[1]}")

    # Plot up to 2 examples of False Negatives
    if not fn_logs.empty:
        print("\n--- Visualizing False Negative Cases (Unsafe but Missed) ---")
        for log_id in fn_logs[:2]:
            log_data_row = pred_df[(pred_df['log_folder'] == log_id[0]) & (pred_df['log_name'] == log_id[1])].iloc[0]
            plot_trajectory_from_dataframe(log_data_row, title=f"False Negative: {log_id[1]}")


if __name__ == "__main__":
    visual_error_analysis()