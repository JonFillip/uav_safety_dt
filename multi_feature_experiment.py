# FAILED EXPERIMENT - PERFORMS WORSE THAN SUPERIALIST, TUNED SUPERIALIST, VAE, CNN-CLASSIFIER, AND TFT
# BEST RESULT - F1 SCORE = 0.234


import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, load_model, save_model
from keras.layers import Input, Conv1D, Dropout, Conv1DTranspose
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# --- Helper Functions ---
def handle_rotation(headings, threshold=np.pi):
    for i in range(1, len(headings)):
        diff = headings[i] - headings[i - 1]
        if diff > threshold:
            headings[i] -= 2 * np.pi
        elif diff < -threshold:
            headings[i] += 2 * np.pi
    return headings

def rolling_ave(group, size, col):
    result = []
    for i in range(len(group)):
        end_idx = i
        start_idx = max(i - size, 0)
        mean_value = group.iloc[start_idx:end_idx][col].mean()
        result.append(mean_value)
    return result

def extract_and_process_data(file_path):
    """
    Robustly loads and processes a dataset for the multi-feature experiment.
    """
    print(f"--- Loading and processing {file_path} ---")
    column_names = [
        'unnamed_col', 'log_folder', 'log_name', 'obstacle-distance', 'risky',
        'win_start', 'win_end', 'win_idx', 'win_obstacle-distance',
        'win_risky', 'x', 'y', 'z', 'r'
    ]
    df = pd.read_csv(file_path, header=None, names=column_names, skiprows=1)

    df.dropna(subset=['x', 'y', 'z', 'r'], inplace=True)
    df = df[df['r'] != 'r'].copy()

    def safe_float_convert(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def process_string_to_list(column_string):
        if not isinstance(column_string, str): return []
        return [v for v in [safe_float_convert(val) for val in column_string.split(",")] if v is not None]

    for col in ['r', 'x', 'y', 'z']:
        if col == 'r':
            df[col] = df[col].apply(lambda s: handle_rotation(process_string_to_list(s)))
        else:
            df[col] = df[col].apply(process_string_to_list)

    # --- FINAL FIX: Ensure all windows have the correct length ---
    window_size = 25
    df = df[df['r'].str.len() == window_size].copy()
    df = df[df['x'].str.len() == window_size].copy()
    df = df[df['y'].str.len() == window_size].copy()
    df = df[df['z'].str.len() == window_size].copy()
    
    print(f"Data cleaned. {len(df)} valid windows remaining.")
    # --- END OF FIX ---

    for col in ['r', 'x', 'y', 'z']:
        df[f"{col}_zero"] = df[col].apply(lambda c: [val - np.mean(c) for val in c])

    df.reset_index(drop=True, inplace=True)
    df["win_dist_0_10"] = df["win_obstacle-distance"] # Simplified for this context
    return df

def build_multi_feature_autoencoder(n_input_features, window_size):
    """Builds the autoencoder using the best-tuned architecture."""
    model = Sequential()
    model.add(Input(shape=(window_size, n_input_features)))
    model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"))
    model.add(Dropout(rate=0.1))
    model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv1DTranspose(filters=32, kernel_size=3, padding="same", activation="relu"))
    model.add(Dropout(rate=0.2))
    model.add(Conv1DTranspose(filters=64, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv1DTranspose(filters=n_input_features, kernel_size=3, padding="same"))
    model.compile(optimizer="adam", loss="mse")
    return model

def print_stats(title, y_true, y_pred):
    """Prints a confusion matrix and key classification metrics."""
    print(f"\n--- {title} ---")
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"Confusion Matrix:\n{cm}")
    print(f"  - True Negatives (TN): {tn}")
    print(f"  - False Positives (FP): {fp}")
    print(f"  - False Negatives (FN): {fn}")
    print(f"  - True Positives (TP): {tp}")
    print(f"\n  - Precision: {precision:.3f}")
    print(f"  - Recall: {recall:.3f}")
    print(f"  - F1-Score: {f1:.3f}")

def train(args):
    """Trains an autoencoder on multiple input features."""
    print("--- Training Multi-Feature Autoencoder ---")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    train_df = extract_and_process_data('datasets/train_dataset.csv')
    nominal_df = train_df[train_df["win_dist_0_10"] > 3.5]
    
    input_features = ['r_zero', 'x_zero', 'y_zero', 'z_zero']
    print(f"Training on {len(nominal_df)} windows with {len(input_features)} features...")

    model = build_multi_feature_autoencoder(n_input_features=len(input_features), window_size=25)
    
    input_values = [np.array(nominal_df[col].to_list()) for col in input_features]
    x_train = np.dstack(input_values)
    
    model.fit(x_train, x_train, epochs=args.epochs, verbose=1, batch_size=128)
    save_model(model, args.output_path)
    print(f"\nMulti-feature autoencoder saved to {args.output_path}")

def evaluate(args):
    """Evaluates the model using a user-specified anomaly score metric."""
    print("--- Evaluating with Different Anomaly Scores ---")
    
    model = load_model(args.model_path)
    test_df = extract_and_process_data(args.eval_file)
    
    input_features = ['r_zero', 'x_zero', 'y_zero', 'z_zero']

    input_values = [np.array(test_df[col].to_list()) for col in input_features]
    x_test = np.dstack(input_values)

    reconstructions = model.predict(x_test, verbose=0)
    loss = tf.keras.losses.mae(reconstructions, x_test)
    mean_loss = tf.reduce_mean(loss, axis=1)
    test_df["mean_loss"] = -mean_loss # Store as negative to match previous logic

    test_df["mean_loss_4"] = (
        test_df.groupby(["log_folder", "log_name"])["mean_loss"]
        .apply(rolling_ave, size=4)
        .explode()
        .tolist()
    )
    
    anomaly_score_col = args.score_metric
    print(f"\nUsing '{anomaly_score_col}' as the anomaly score.")
    test_df['predicted_positive'] = test_df[anomaly_score_col] < args.threshold

    log_labels = test_df.groupby(['log_folder', 'log_name'])[args.target_label].any()
    log_predictions = test_df.groupby(['log_folder', 'log_name'])['predicted_positive'].any()
    
    print_stats(
        f"Prediction for '{args.target_label}' on {os.path.basename(args.eval_file)}",
        log_labels,
        log_predictions
    )

def main():
    parser = argparse.ArgumentParser(description="UAV Multi-Feature Experiment Runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    parser_train = subparsers.add_parser('train', help='Train the multi-feature autoencoder')
    parser_train.add_argument('-o', '--output-path', required=True)
    parser_train.add_argument('--epochs', type=int, default=100)
    parser_train.set_defaults(func=train)

    parser_eval = subparsers.add_parser('evaluate', help='Evaluate the model')
    parser_eval.add_argument('--model-path', required=True)
    parser_eval.add_argument('--eval-file', default='datasets/test1_dataset.csv')
    parser_eval.add_argument('--score-metric', choices=['mean_loss', 'mean_loss_4'], default='mean_loss_4')
    parser_eval.add_argument('--threshold', type=float, required=True)
    parser_eval.add_argument('--target-label', choices=['unsafe', 'uncertain'], default='unsafe')
    parser_eval.set_defaults(func=evaluate)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()