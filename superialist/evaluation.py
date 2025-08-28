import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from superialist_model import CNNModel
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import f_oneway

# --- Helper Function for Statistics (replaces data_analysis.stats) ---
def calculate_stats(df, prediction_col, threshold, ground_truth_col):
    """
    Calculates and prints performance metrics for the model.
    """
    # For autoencoders, a lower score (more negative) is a stronger anomaly signal.
    # Therefore, a prediction is positive if the score is LESS THAN the threshold.
    if "loss" in prediction_col:
        df['predicted_positive'] = df[prediction_col] < threshold
    else:
        df['predicted_positive'] = df[prediction_col] > threshold

    # Aggregate predictions on a per-flight (log) basis
    log_labels = df.groupby(['log_folder', 'log_name'])[ground_truth_col].any()
    log_predictions = df.groupby(['log_folder', 'log_name'])['predicted_positive'].any()

    # Calculate metrics
    tn = np.sum((log_labels == False) & (log_predictions == False))
    tp = np.sum((log_labels == True) & (log_predictions == True))
    fn = np.sum((log_labels == True) & (log_predictions == False))
    fp = np.sum((log_labels == False) & (log_predictions == True))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    print(f"\n--- Performance for '{ground_truth_col}' ---")
    print(f"  - F1-Score:  {f1:.3f}")
    print(f"  - Precision: {precision:.3f}")
    print(f"  - Recall:    {recall:.3f}")
    print(f"  - Accuracy:  {accuracy:.3f}")
    print(f"  - TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

# -- LSTM Autoencoder Model Definition
def create_lstm_autoencoder(n_steps=25, n_features=1):
    """Creates an LSTM-based autoencoder for anomaly detection."""
    model = Sequential([
        Input(shape=(n_steps, n_features)),
        LSTM(64, activation='relu', return_sequences=False),
        RepeatVector(n_steps),
        LSTM(24, activation='relu', return_sequences=True),
        TimeDistributed(Dense(n_features))
    ])
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model

# --- Main Functions ---

# --- Function to Find Optimal Threshold ---

def find_optimal_threshold(args):
    """Finds the best threshold by optimizing a weighted average of F1-scores."""
    print(f"--- Finding Optimal Threshold for Model: {args.model_path} ---")

    model_wrapper = CNNModel()
    model_wrapper.load(args.model_path)

    # --- Load the associated scaler ---
    base_name = os.path.splitext(os.path.basename(args.model_path))[0]
    scaler_path = os.path.join(os.path.dirname(args.model_path), f'{base_name}_scaler.joblib')
    if not os.path.exists(scaler_path):
        print(f"Error: Could not find the required scaler file at {scaler_path}")
        return
    scaler = joblib.load(scaler_path)
    print(f"Loaded scaler from {scaler_path}")

    print(f"--- Using evaluation file: {args.eval_file} ---")
    eval_data = model_wrapper.extract_dataset(args.eval_file)
    
    # --- Prepare and scale the features ---
    eval_data['dist_vec'] = eval_data['win_obstacle-distance'].apply(lambda x: [x] * CNNModel.WINSIZE)
    inputs = ['r_zero'] # Using the best feature set
    n_features = len(inputs)
    
    feature_array = np.dstack([np.array(eval_data[col].to_list()) for col in inputs])
    original_shape = feature_array.shape
    reshaped_for_scaling = feature_array.reshape(-1, n_features)
    scaled_data = scaler.transform(reshaped_for_scaling)
    X_eval_scaled = scaled_data.reshape(original_shape)
    
    scaled_inputs = [f'{col}_scaled' for col in inputs]
    for i, col_name in enumerate(scaled_inputs):
        eval_data[col_name] = list(X_eval_scaled[:, :, i])
    
    pred_data = model_wrapper.predict_encoder(eval_data, inputs=scaled_inputs)
    
    # --- NEW: Tuning logic with combined, weighted score ---
    best_combined_score = 0
    best_threshold = 0
    best_f1_unsafe = 0
    best_f1_uncertain = 0

    print(f"\nTuning threshold to optimize a weighted F1-score (unsafe_weight={args.unsafe_weight:.2f})...")
    
    for threshold in np.arange(-1.0, 0.0, 0.01):
        pred_data['predicted_positive'] = pred_data["mean_loss_4"] < threshold
        log_predictions = pred_data.groupby(['log_folder', 'log_name'])['predicted_positive'].any()

        # Calculate F1 for 'unsafe'
        log_labels_unsafe = pred_data.groupby(['log_folder', 'log_name'])['unsafe'].any()
        f1_unsafe = f1_score(log_labels_unsafe, log_predictions, zero_division=0)

        # Calculate F1 for 'uncertain'
        log_labels_uncertain = pred_data.groupby(['log_folder', 'log_name'])['uncertain'].any()
        f1_uncertain = f1_score(log_labels_uncertain, log_predictions, zero_division=0)
        
        # Calculate the combined, weighted score
        combined_score = (args.unsafe_weight * f1_unsafe) + ((1 - args.unsafe_weight) * f1_uncertain)
        
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_threshold = threshold
            best_f1_unsafe = f1_unsafe
            best_f1_uncertain = f1_uncertain
            
    print("\n--- Optimal Threshold Found ---")
    print(f"Best Combined Score: {best_combined_score:.4f}")
    print(f"Corresponding Threshold: {best_threshold:.2f}")
    print(f"  - F1-Score for 'unsafe' at this threshold:    {best_f1_unsafe:.4f}")
    print(f"  - F1-Score for 'uncertain' at this threshold:  {best_f1_uncertain:.4f}")

    print("\n--- Final Performance at Optimal Threshold ---")
    calculate_stats(pred_data, "mean_loss_4", best_threshold, "unsafe")
    calculate_stats(pred_data, "mean_loss_4", best_threshold, "uncertain")


def train_superialist(args):
    """Trains the SUPERIALIST autoencoder model with a scaled, expanded feature set."""
    print("--- Training ENHANCED SUPERIALIST Autoencoder ---")
    
    model_wrapper = CNNModel()
    
    print("Loading training data...")
    train_data = model_wrapper.extract_dataset(args.train_file)
    
    nominal_data = train_data[train_data["win_dist_0_10"] > 3.5].copy()
    print(f"Training on {len(nominal_data)} nominal data windows...")

    # --- Prepare the expanded feature set ---
    nominal_data['dist_vec'] = nominal_data['win_obstacle-distance'].apply(lambda x: [x] * CNNModel.WINSIZE)
    inputs = ['r_zero']
    n_features = len(inputs)
    print(f"Preparing {n_features} features for scaling: {', '.join(inputs)}")

    # --- NEW: SCALING LOGIC ---
    # 1. Stack features into a NumPy array and reshape for the scaler
    feature_array = np.dstack([np.array(nominal_data[col].to_list()) for col in inputs])
    original_shape = feature_array.shape
    reshaped_for_scaling = feature_array.reshape(-1, n_features)

    # 2. Fit the scaler and transform the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(reshaped_for_scaling)
    
    # 3. Reshape the data back to its original 3D shape
    X_train_scaled = scaled_data.reshape(original_shape)
    
    # 4. Put the scaled data back into the DataFrame in new columns
    scaled_inputs = [f'{col}_scaled' for col in inputs]
    for i, col_name in enumerate(scaled_inputs):
        nominal_data[col_name] = list(X_train_scaled[:, :, i])

    # --- Save the model and the scaler ---
    base_name = f'enhanced_scaled_{args.model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    model_path = os.path.join(args.model_dir, f'{base_name}.keras')
    scaler_path = os.path.join(args.model_dir, f'{base_name}_scaler.joblib')
    
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # --- Train the model using the NEW scaled features ---
    model = model_wrapper.get_autoencoder_model(n_input_features=n_features)
    model_wrapper.fit(
        nominal_data,
        model=model,
        inputs=scaled_inputs,
        outputs=scaled_inputs, # Autoencoder output must match input
        output_is_list=True,
        epochs=args.epochs
    )
    
    model_wrapper.save(model_path)
    print(f"\nSuccessfully trained and saved ENHANCED SCALED model to: {model_path}")

def evaluate_superialist(args):
    """Evaluates a trained ENHANCED SUPERIALIST model with scaled features."""
    print(f"--- Evaluating ENHANCED SCALED SUPERIALIST Model: {args.model_path} ---")

    model_wrapper = CNNModel()
    model_wrapper.load(args.model_path)
    
    # --- NEW: LOAD THE SCALER ---
    base_name = os.path.splitext(os.path.basename(args.model_path))[0]
    scaler_path = os.path.join(os.path.dirname(args.model_path), f'{base_name}_scaler.joblib')
    if not os.path.exists(scaler_path):
        print(f"Error: Could not find the scaler file at {scaler_path}")
        return
    scaler = joblib.load(scaler_path)
    print(f"Loaded scaler from {scaler_path}")

    print(f"\n--- Evaluating on {args.eval_file} ---")
    eval_data = model_wrapper.extract_dataset(args.eval_file)
    
    # --- Prepare and scale the evaluation features ---
    eval_data['dist_vec'] = eval_data['win_obstacle-distance'].apply(lambda x: [x] * CNNModel.WINSIZE)
    inputs = ['r_zero' ]
    n_features = len(inputs)
    
    feature_array = np.dstack([np.array(eval_data[col].to_list()) for col in inputs])
    original_shape = feature_array.shape
    reshaped_for_scaling = feature_array.reshape(-1, n_features)
    
    # Use the loaded scaler to transform the evaluation data
    scaled_data = scaler.transform(reshaped_for_scaling)
    X_eval_scaled = scaled_data.reshape(original_shape)
    
    scaled_inputs = [f'{col}_scaled' for col in inputs]
    for i, col_name in enumerate(scaled_inputs):
        eval_data[col_name] = list(X_eval_scaled[:, :, i])
    
    print(f"Evaluating with {len(scaled_inputs)} scaled features.")
    
    # Get anomaly scores using the scaled data
    pred_data = model_wrapper.predict_encoder(eval_data, inputs=scaled_inputs)

    # Evaluate performance
    print(f"Using anomaly threshold: {args.threshold}")
    calculate_stats(pred_data, "mean_loss_4", args.threshold, "uncertain")
    calculate_stats(pred_data, "mean_loss_4", args.threshold, "unsafe")

def analyze_superialist_uncertainty(args):
    """Analyzes the correlation between the Superialist model's error and the 'uncertain' label."""
    print(f"--- Analyzing Uncertainty for SUPERIALIST Model: {args.model_path} ---")
    
    model_wrapper = CNNModel()
    model_wrapper.load(args.model_path)

    print(f"Loading and processing {args.eval_file}...")
    eval_data = model_wrapper.extract_dataset(args.eval_file)
    
    # Get reconstruction errors from the Superialist model (trained on r_zero)
    pred_data = model_wrapper.predict_encoder(eval_data, inputs=['r_zero'])

    # Create a DataFrame for analysis
    analysis_df = pd.DataFrame({
        'reconstruction_error': pred_data['mean_loss'], # The predict_encoder function already calculates and negates the loss
        'is_uncertain': eval_data['uncertain']
    })

    # --- Correlation and Statistical Analysis ---
    print("\n--- Correlation and Significance Analysis ---")
    
    # 1. Linear Correlation
    analysis_df['is_uncertain_numeric'] = analysis_df['is_uncertain'].astype(int)
    correlation_score = analysis_df['reconstruction_error'].corr(analysis_df['is_uncertain_numeric'])
    print(f"Linear Correlation (Point-Biserial) Score: {correlation_score:.4f}")
    
    # 2. Non-Linear Correlation
    X_feature = analysis_df[['reconstruction_error']]
    y_labels = analysis_df['is_uncertain']
    mi_score = mutual_info_classif(X_feature, y_labels, discrete_features=False, random_state=42)[0]
    print(f"Non-Linear Correlation (Mutual Info) Score: {mi_score:.4f}")
    
    # 3. Statistical Significance Test
    errors_certain = analysis_df[analysis_df['is_uncertain'] == False]['reconstruction_error']
    errors_uncertain = analysis_df[analysis_df['is_uncertain'] == True]['reconstruction_error']
    f_statistic, p_value = f_oneway(errors_certain, errors_uncertain)
    print(f"ANOVA F-statistic: {f_statistic:.2f} | P-value: {p_value:.4e}")
    if p_value < 0.05:
        print("Conclusion: The difference in error between groups is statistically significant.")
    else:
        print("Conclusion: The difference in error between groups is not statistically significant.")

    # --- Visualization ---
    print("\nGenerating visualization...")
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='is_uncertain', y='reconstruction_error', data=analysis_df)
    title = f"Reconstruction Error Distribution on {os.path.basename(args.eval_file)}"
    plt.title(title)
    plt.xlabel('Is the Window Labeled as Uncertain?')
    plt.ylabel('Reconstruction Error (Anomaly Score)')
    plt.grid(True)
    
    plot_path = os.path.join(os.path.dirname(args.model_path), f"superialist_uncertainty_correlation_{os.path.basename(args.eval_file).split('.')[0]}.png")
    plt.savefig(plot_path)
    plt.show()
    plt.close()

    print(f"\nAnalysis complete. Plot saved to: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the SUPERIALIST model from the original paper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Train Command ---
    parser_train = subparsers.add_parser('train', help='Train a new SUPERIALIST model.')
    parser_train.add_argument('--model-type', choices=['superialist', 'lstm_ae'], default='superialist', help='Type of autoencoder to train.')
    parser_train.add_argument('--train-file', default='datasets/train_dataset.csv', help='Path to the training dataset.')
    parser_train.add_argument('--model-dir', default='models', help='Directory to save the trained model.')
    parser_train.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser_train.set_defaults(func=train_superialist)

    # --- Evaluate Command ---
    parser_eval = subparsers.add_parser('evaluate', help='Evaluate a trained SUPERIALIST model.')
    parser_eval.add_argument('model_path', help='Path to the saved .keras model file.')
    parser_eval.add_argument('--eval-file', default='datasets/test1_dataset.csv', help='Path to the evaluation dataset (test1 or test2).')
    parser_eval.add_argument('--threshold', type=float, default=-0.3, help='Anomaly threshold for classification.')
    parser_eval.set_defaults(func=evaluate_superialist)

    # --- Tune Threshold Command ---
    parser_tune = subparsers.add_parser('tune-threshold', help='Find the optimal decision threshold.')
    parser_tune.add_argument('model_path', help='Path to the saved .keras model file.')
    parser_tune.add_argument('--eval-file', default='datasets/test1_dataset.csv', help='Dataset to use for tuning.')
    parser_tune.add_argument('--unsafe-weight', type=float, default=0.7, help='Weight for the unsafe F1-score in the combined tuning metric (e.g., 0.7 for 70%).')
    parser_tune.set_defaults(func=find_optimal_threshold)


    # --- Correlation Analysis Command ---
    parser_analyze = subparsers.add_parser('analyze', help='Analyze correlation between Superialist error and uncertainty.')
    parser_analyze.add_argument('model_path', help='Path to the saved .keras model file.')
    parser_analyze.add_argument('--eval-file', default='datasets/test1_dataset.csv')
    parser_analyze.set_defaults(func=analyze_superialist_uncertainty)

    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    if hasattr(args, 'model_dir'):
        os.makedirs(args.model_dir, exist_ok=True)
        
    args.func(args)

if __name__ == "__main__":
    main()
