import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Import from your existing, trusted scripts
from superialist.superialist_model import CNNModel
from temporal_fusion_transformer import TemporalFusionTransformer, prepare_tft_data
from superialist.evaluation import calculate_stats
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

def train_tft(args):
    """Trains the Temporal Fusion Transformer model."""
    print("--- Training Temporal Fusion Transformer ---")

    # 1. Use the trusted extract_dataset to get the data
    print("Loading and processing training data...")
    full_train_df = CNNModel.extract_dataset(args.train_file)

    # 2. Filter for nominal data
    nominal_df = full_train_df[full_train_df["win_dist_0_10"] > 3.5].reset_index(drop=True)
    print(f"Training on {len(nominal_df)} nominal data windows...")

    # 3. Prepare data specifically for the TFT
    X_train, y_train = prepare_tft_data(
        nominal_df,
        past_history_len=args.window_size,
        future_target_len=args.horizon,
        features=['r_zero'],
        target='r_zero'
    )

    # 4. Build, train, and save the model
    tft = TemporalFusionTransformer(window_size=args.window_size, prediction_horizon=args.horizon)
    tft.get_best_model(num_features=1) # Using best-tuned hyperparameters
    tft.fit(X_train, y_train, epochs=args.epochs)

    model_path = os.path.join(args.model_dir, f'tft_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.weights.h5')
    tft.save(model_path)
    print(f"\nSuccessfully trained and saved model weights to: {model_path}")


def tune_threshold_tft(args):
    """Finds the optimal F1 threshold on a tuning dataset."""
    print(f"--- Tuning Threshold for TFT Model: {args.model_path} ---")

    # 1. Load the model with the correct architecture
    tft = TemporalFusionTransformer(window_size=args.window_size, prediction_horizon=args.horizon)
    tft.load(args.model_path, num_features=1, num_attention_heads=2)

    # 2. Load the full tuning data
    print(f"--- Loading tuning file: {args.tune_file} ---")
    full_tune_df = CNNModel.extract_dataset(args.tune_file)

    # 3. If tune_frac is less than 1.0, create a representative subset
    if args.tune_frac < 1.0:
        print(f"Original tuning set has {len(full_tune_df)} windows. Creating a {args.tune_frac:.0%} subset.")
        # Identify unique flights and determine if they contain an 'unsafe' event for stratification
        flight_groups = full_tune_df.groupby(['log_folder', 'log_name'])
        flight_labels = flight_groups['unsafe'].any()
        
        unique_flights = flight_labels.index.to_frame(index=False)
        labels_for_stratify = flight_labels.values

        # Perform a stratified split to select a fraction of the flights
        tune_flights_df, _ = train_test_split(
            unique_flights,
            train_size=args.tune_frac,
            stratify=labels_for_stratify,
            random_state=42
        )

        # Filter the original dataframe to keep only the windows from the selected flights
        tune_df = pd.merge(full_tune_df, tune_flights_df, on=['log_folder', 'log_name'], how='inner')
        print(f"Using subset of {len(tune_df)} windows from {len(tune_flights_df)} flights for tuning.")
    else:
        tune_df = full_tune_df
        print(f"Using the entire tuning set ({len(tune_df)} windows) for tuning.")

    # 4. Prepare data for the model from the (potentially smaller) tuning dataframe
    X_tune, _ = prepare_tft_data(
        tune_df,
        past_history_len=args.window_size,
        future_target_len=args.horizon,
        features=['r_zero'],
        target='r_zero'
    )

    # 5. Get anomaly scores
    anomaly_scores = tft.predict_anomaly_score(X_tune)
    pred_df = tune_df.iloc[:len(anomaly_scores)].copy()
    pred_df['anomaly_score'] = anomaly_scores

    # 6. Find the optimal threshold
    best_f1 = 0
    best_threshold = 0
    for threshold in np.arange(0.0, 1.0, 0.01):
        pred_df['predicted_positive'] = pred_df['anomaly_score'] > threshold
        log_labels = pred_df.groupby(['log_folder', 'log_name'])['unsafe'].any()
        log_predictions = pred_df.groupby(['log_folder', 'log_name'])['predicted_positive'].any()
        f1 = f1_score(log_labels, log_predictions, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print("\n--- Optimal Threshold Found ---")
    print(f"Best Per-Flight F1-Score on '{os.path.basename(args.tune_file)}' (using a {args.tune_frac:.0%} sample): {best_f1:.4f}")
    print(f"Corresponding Threshold: {best_threshold:.2f}")
    print("\nUse this threshold for the 'evaluate' command on a separate test set.")

def evaluate_tft(args):
    """Evaluates a trained TFT model using a fixed, pre-tuned threshold."""
    print(f"--- Evaluating TFT Model: {args.model_path} ---")

    # 1. Load the model with its architecture
    tft = TemporalFusionTransformer(window_size=args.window_size, prediction_horizon=args.horizon)
    tft.load(args.model_path, num_features=1, num_attention_heads=2) # Rebuilds model and loads weights

    # 2. Load and process evaluation data
    print(f"\n--- Evaluating on {args.eval_file} with threshold {args.threshold:.2f} ---")
    eval_df = CNNModel.extract_dataset(args.eval_file)

    X_eval, _ = prepare_tft_data(
        eval_df,
        past_history_len=args.window_size,
        future_target_len=args.horizon,
        features=['r_zero'],
        target='r_zero'
    )

    # 3. Get anomaly scores (prediction uncertainty)
    anomaly_scores = tft.predict_anomaly_score(X_eval)
    pred_df = eval_df.iloc[:len(anomaly_scores)].copy()
    pred_df['anomaly_score'] = anomaly_scores

    # 4. Calculate and print final stats using the best threshold
    print("\n--- Final Performance ---")
    calculate_stats(pred_df, "anomaly_score", args.threshold, "unsafe")
    calculate_stats(pred_df, "anomaly_score", args.threshold, "uncertain")

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the Temporal Fusion Transformer model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Train Command ---
    parser_train = subparsers.add_parser('train', help='Train a new TFT model.')
    parser_train.add_argument('--train-file', default='datasets/train_dataset.csv')
    parser_train.add_argument('--model-dir', default='models')
    parser_train.add_argument('--epochs', type=int, default=50)
    parser_train.add_argument('--window-size', type=int, default=20)
    parser_train.add_argument('--horizon', type=int, default=5)
    parser_train.set_defaults(func=train_tft)

    # --- Evaluate Command (Modified) ---
    parser_eval = subparsers.add_parser('evaluate', help='Evaluate a trained TFT model with a fixed threshold.')
    parser_eval.add_argument('model_path', help='Path to the saved .weights.h5 model file.')
    parser_eval.add_argument('--eval-file', default='datasets/test2_dataset.csv')
    parser_eval.add_argument('--threshold', type=float, required=True, help='Anomaly threshold obtained from the tune-threshold command.')
    parser_eval.add_argument('--window-size', type=int, default=20)
    parser_eval.add_argument('--horizon', type=int, default=5)
    parser_eval.set_defaults(func=evaluate_tft)

    # --- Tune Threshold Command (NEW) ---
    parser_tune = subparsers.add_parser('tune-threshold', help='Find the optimal decision threshold for a TFT model.')
    parser_tune.add_argument('model_path', help='Path to the saved .weights.h5 model file.')
    parser_tune.add_argument('--tune-file', default='datasets/test1_dataset.csv', help='Dataset to use for tuning the threshold.')
    parser_tune.add_argument('--tune-frac', type=float, default=1.0, help='Fraction of the tuning file to use (e.g., 0.3 for 30%).')
    parser_tune.add_argument('--window-size', type=int, default=20)
    parser_tune.add_argument('--horizon', type=int, default=5)
    parser_tune.set_defaults(func=tune_threshold_tft)
    

    args = parser.parse_args()

    if hasattr(args, 'model_dir'):
        os.makedirs(args.model_dir, exist_ok=True)

    args.func(args)

if __name__ == "__main__":
    main()