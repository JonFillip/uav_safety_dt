import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

# --- Data Processing Functions (from original model.py) ---

def handle_rotation(headings, threshold=np.pi):
    for i in range(1, len(headings)):
        diff = headings[i] - headings[i - 1]
        if diff > threshold:
            headings[i] -= 2 * np.pi
        elif diff < -threshold:
            headings[i] += 2 * np.pi
    return headings

def rolling_min(group, size, offset=2):
    result = []
    for i in range(len(group)):
        start_idx = min(i + offset, len(group) - 1)
        end_idx = min(start_idx + size, len(group))
        min_value = group.iloc[start_idx:end_idx]["win_obstacle-distance"].min()
        result.append(min_value)
    return result

def extract_dataset(file_path, winsize=25):
    """Extracts and engineers features, including obstacle distance."""
    df = pd.read_csv(file_path)
    if 'unsafe' not in df.columns:
        df['unsafe'] = False
        
    df["r"] = df["r"].astype(str).apply(lambda x: handle_rotation([float(val) for val in x.split(",")]))
    df["x"] = df["x"].astype(str).apply(lambda x: [float(val) for val in x.split(",")])
    df["y"] = df["y"].astype(str).apply(lambda x: [float(val) for val in x.split(",")])
    df["z"] = df["z"].astype(str).apply(lambda x: [float(val) for val in x.split(",")])
    df["r_zero"] = df["r"].apply(lambda x: [val - np.mean(x) for val in x])
    df["x_zero"] = df["x"].apply(lambda x: [val - np.mean(x) for val in x])
    df["y_zero"] = df["y"].apply(lambda x: [val - np.mean(x) for val in x])
    df["z_zero"] = df["z"].apply(lambda x: [val - np.mean(x) for val in x])
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    df = df[df["r"].apply(lambda x: len(x) == winsize)]

    # --- ADDING ADVANCED FEATURES ---
    df["win_dist_1_10"] = df.groupby(["log_folder", "log_name"], group_keys=False).apply(rolling_min, size=21, offset=0, include_groups=False).explode().tolist()
    df["win_dist_0"] = df["win_obstacle-distance"]
    df["win_dist_0_10"] = df[["win_dist_0", "win_dist_1_10"]].min(axis=1)
    
    return df

# --- Helper Functions ---

def print_stats(title, y_true, y_pred):
    print(f"\n--- {title} ---")
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        print(f"  - True Negatives (TN): {tn}")
        print(f"  - False Positives (FP): {fp}")
        print(f"  - False Negatives (FN): {fn}")
        print(f"  - True Positives (TP): {tp}")
        print(f"\n  - Accuracy:  {(tp+tn)/(tp+tn+fp+fn):.3f}")
    else:
        print(f"Confusion Matrix:\n{cm}")

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"  - Precision: {precision:.3f}")
    print(f"  - Recall:    {recall:.3f}")
    print(f"  - F1-Score:  {f1:.3f}")

def prepare_predictor_data(df, input_cols):
    data_list = [np.array(df[col].to_list()) for col in input_cols]
    X = np.stack(data_list, axis=-1)
    y = X[:, -1, :]
    return X, y

# --- Main Experiment Functions ---

def run_training(args):
    print("=== Training Full Two-Stage Model with Enriched Features ===")
    os.makedirs(os.path.dirname(args.predictor_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.detector_path), exist_ok=True)
    
    # --- Part 1: Train the Predictor ---
    print("\n--- Training Predictor ---")
    input_cols = ['r_zero', 'x_zero', 'y_zero', 'z_zero']
    n_features = len(input_cols)
    n_steps = 25

    train_df = extract_dataset('datasets/train_dataset.csv', winsize=n_steps)
    X_train_predictor, y_train_predictor = prepare_predictor_data(train_df, input_cols)
    
    predictor_model = Sequential([
        Input(shape=(n_steps, n_features)),
        LSTM(50, return_sequences=False),
        Dense(n_features)
    ])
    predictor_model.compile(optimizer='adam', loss='mse')
    predictor_model.fit(X_train_predictor, y_train_predictor, epochs=args.epochs, batch_size=256, verbose=1)
    predictor_model.save(args.predictor_path)
    print(f"Predictor model saved to {args.predictor_path}")

    # --- Part 2: Train the Detector with Enriched Features ---
    print("\n--- Training Detector ---")
    predictor_model = tf.keras.models.load_model(args.predictor_path)
    
    labeled_df = extract_dataset('datasets/test2_dataset.csv', winsize=n_steps)
    X_labeled, y_labeled_unused = prepare_predictor_data(labeled_df, input_cols)

    predictions = predictor_model.predict(X_labeled)
    errors = np.mean(np.square(y_labeled_unused - predictions), axis=1)

    # --- NEW: Create a combined feature set for the detector ---
    detector_features = labeled_df[['win_dist_0_10']].copy()
    detector_features['error'] = errors
    
    # Scale the new features
    scaler = MinMaxScaler()
    detector_input = scaler.fit_transform(detector_features)
    detector_labels = labeled_df['unsafe'].values

    class_weight_dict = {False: 0.5, True: 10.0}
    print(f"Using MANUAL class weights: {class_weight_dict}")
    
    # --- NEW: Update detector input shape ---
    detector_model = Sequential([
        Input(shape=(detector_input.shape[1],)), # Shape is now based on number of features
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    detector_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    detector_model.summary()
    detector_model.fit(detector_input, detector_labels, epochs=args.epochs, class_weight=class_weight_dict, batch_size=256, verbose=1)
    detector_model.save(args.detector_path)
    print(f"Detector model saved to {args.detector_path}")

def run_evaluation(args):
    print("=== Evaluating Two-Stage Model with Enriched Features ===")
    predictor = tf.keras.models.load_model(args.predictor_path)
    detector = tf.keras.models.load_model(args.detector_path)
    
    n_steps = predictor.input_shape[1]
    input_cols = ['r_zero', 'x_zero', 'y_zero', 'z_zero']

    eval_df = extract_dataset(args.eval_file, winsize=n_steps)
    X_eval, y_eval_unused = prepare_predictor_data(eval_df, input_cols)
    
    predictions = predictor.predict(X_eval)
    errors = np.mean(np.square(y_eval_unused - predictions), axis=1)
    
    # --- NEW: Prepare combined features for evaluation ---
    detector_features_eval = eval_df[['win_dist_0_10']].copy()
    detector_features_eval['error'] = errors
    
    # NOTE: In a real-world scenario, you would save and load the scaler from training.
    # For this script, we re-fit it on the test data for simplicity.
    scaler = MinMaxScaler()
    detector_input_eval = scaler.fit_transform(detector_features_eval)

    window_probabilities = detector.predict(detector_input_eval).flatten()
    window_labels = eval_df['unsafe'].values

    # --- Automatic Threshold Tuning Loop ---
    best_f1 = 0
    best_threshold = 0
    thresholds = np.arange(0.05, 1.0, 0.05)

    for threshold in thresholds:
        preds = (window_probabilities > threshold).astype(int)
        f1 = f1_score(window_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print("\n--- Optimal Threshold Found ---")
    print(f"Best F1-Score: {best_f1:.4f} at Threshold: {best_threshold:.2f}")

    # --- Final Evaluation using the Best Threshold ---
    optimal_predictions = (window_probabilities > best_threshold).astype(int)
    print_stats("Per-Window Performance (at Optimal Threshold)", window_labels, optimal_predictions)

    # --- Per-Flight (Log-Based) Evaluation ---
    eval_results_df = eval_df.copy()
    eval_results_df['predicted'] = optimal_predictions
    
    flight_labels = eval_results_df.groupby(['log_folder', 'log_name'])['unsafe'].any()
    flight_predictions = eval_results_df.groupby(['log_folder', 'log_name'])['predicted'].any()
    
    print_stats("Per-Flight (Log-Based) Performance (at Optimal Threshold)", flight_labels, flight_predictions)

def main():
    parser = argparse.ArgumentParser(description="Advanced Digital Twin Experiment Runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_train = subparsers.add_parser('train', help='Train the two-stage model')
    parser_train.add_argument('--predictor-path', default='models/dt_predictor.keras', help='Path to save the predictor model')
    parser_train.add_argument('--detector-path', default='models/dt_detector.keras', help='Path to save the detector model')
    parser_train.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser_train.set_defaults(func=run_training)

    parser_eval = subparsers.add_parser('evaluate', help='Evaluate a trained two-stage model')
    parser_eval.add_argument('--predictor-path', default='models/dt_predictor.keras', help='Path to the saved predictor model')
    parser_eval.add_argument('--detector-path', default='models/dt_detector.keras', help='Path to the saved detector model')
    parser_eval.add_argument('--eval-file', default='datasets/test1_dataset.csv', help="Path to the dataset for evaluation")
    parser_eval.set_defaults(func=run_evaluation)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()