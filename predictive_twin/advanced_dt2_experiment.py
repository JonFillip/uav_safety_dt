import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, Add, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from datetime import datetime
import csv
import json
import matplotlib.pyplot as plt
import joblib

# --- Experiment Logging Class ---
class ExperimentLogger:
    def __init__(self, file_path='experiment_lineage.csv'):
        self.file_path = file_path
        self.fieldnames = ['run_id', 'timestamp', 'event_type', 'details', 'artifact_path']
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, run_id, event_type, details, artifact_path=''):
        with open(self.file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({
                'run_id': run_id,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'event_type': event_type,
                'details': str(details),
                'artifact_path': artifact_path
            })

# --- Data Processing and Helper Functions ---
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
    
def calculate_distance_trend(group, window_size=5):
    distances = group["win_obstacle-distance"]
    trend = distances.diff(periods=window_size)
    return trend.fillna(0)

def parse_string_to_list(s):
    """Safely parses a string representation of a list of floats."""
    # Remove brackets and split by comma
    cleaned_string = s.strip('[]')
    if not cleaned_string:
        return []
    return [float(item) for item in cleaned_string.split(',')]

def extract_dataset(file_path, winsize=25):
    """Extracts and engineers features, including obstacle distance."""
    df = pd.read_csv(file_path)
    if 'unsafe' not in df.columns:
        df['unsafe'] = False
    
    # CORRECTED: Use the robust parser for list-like columns
    list_columns = ["r", "x", "y", "z"]
    for col in list_columns:
        df[col] = df[col].astype(str).apply(parse_string_to_list)

    # Apply handle_rotation after parsing
    df["r"] = df["r"].apply(handle_rotation)

    df["r_zero"] = df["r"].apply(lambda x: [val - np.mean(x) for val in x])
    df["x_zero"] = df["x"].apply(lambda x: [val - np.mean(x) for val in x])
    df["y_zero"] = df["y"].apply(lambda x: [val - np.mean(x) for val in x])
    df["z_zero"] = df["z"].apply(lambda x: [val - np.mean(x) for val in x])
    df = df.dropna(subset=['r', 'x', 'y', 'z', 'win_obstacle-distance'])
    df.reset_index(drop=True, inplace=True)
    df = df[df["r"].apply(lambda x: len(x) == winsize)]
    df["win_dist_1_10"] = df.groupby(["log_folder", "log_name"], group_keys=False).apply(rolling_min, size=21, offset=0, include_groups=False).explode().tolist()
    df["dist_trend"] = df.groupby(["log_folder", "log_name"])["win_obstacle-distance"].diff(periods=5).fillna(0)
    return df

def print_stats(title, y_true, y_pred):
    print(f"\n--- {title} ---")
    cm = confusion_matrix(y_true, y_pred)
    metrics = {}
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        print(f"  - True Negatives (TN): {tn}")
        print(f"  - False Positives (FP): {fp}")
        print(f"  - False Negatives (FN): {fn}")
        print(f"  - True Positives (TP): {tp}")
        accuracy = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn) > 0 else 0
        print(f"\n  - Accuracy:  {accuracy:.3f}")
        metrics['accuracy'] = accuracy
        metrics['confusion_matrix'] = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    else:
        print(f"Confusion Matrix:\n{cm}")
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"  - Precision: {precision:.3f}")
    print(f"  - Recall:    {recall:.3f}")
    print(f"  - F1-Score:  {f1:.3f}")
    metrics.update({'precision': precision, 'recall': recall, 'f1_score': f1})
    return metrics

def prepare_predictor_data(df, input_cols):
    data_list = [np.array(df[col].to_list()) for col in input_cols]
    X = np.stack(data_list, axis=-1)
    y = X[:, -1, :]
    return X, y

def get_consecutive_flight_predictions(df, n_consecutive):
    temp_df = df[['log_folder', 'log_name', 'predicted_window']].copy()
    temp_df['consecutive_sum'] = temp_df.groupby(['log_folder', 'log_name'])['predicted_window'].rolling(window=n_consecutive).sum().reset_index(level=[0,1], drop=True)
    temp_df['consecutive_sum'] = temp_df['consecutive_sum'].fillna(0)
    flight_predictions = temp_df.groupby(['log_folder', 'log_name'])['consecutive_sum'].max() >= n_consecutive
    return flight_predictions
    
def plot_loss_history(history, save_path):
    plt.figure(figsize=(10, 6)); plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Detector Model Training and Validation Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True); plt.savefig(save_path); plt.show(); plt.close()
    print(f"Training history plot saved to {save_path}")

def plot_roc_curve(y_true, y_pred_proba, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 6)); plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--'); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right"); plt.grid(True); plt.savefig(save_path); plt.show(); plt.close()
    print(f"AUC-ROC curve plot saved to {save_path}")

# --- Predictor Model Definitions ---
def create_lstm_predictor(n_steps, n_features):
    return Sequential([Input(shape=(n_steps, n_features)), LSTM(50, return_sequences=False), Dense(n_features)])

def create_tft_predictor(n_steps, n_features):
    inputs = Input(shape=(n_steps, n_features)); lstm_out = LSTM(64, return_sequences=True)(inputs)
    attention_out = MultiHeadAttention(num_heads=4, key_dim=32)(lstm_out, lstm_out)
    res_out = Add()([lstm_out, attention_out]); norm_out = LayerNormalization()(res_out)
    flatten_out = Flatten()(norm_out); outputs = Dense(n_features)(flatten_out)
    return Model(inputs=inputs, outputs=outputs)

# --- Main Experiment Functions ---
def run_training(args):
    logger = ExperimentLogger()
    run_id = args.run_id or f"dt_{args.predictor_type}_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_path = os.path.join('experiments', f'run_{run_id}')
    os.makedirs(run_path, exist_ok=True)
    
    print(f"=== Starting Training Run ID: {run_id} (Mode: {args.mode}) ===")
    print(f"All artifacts will be saved in: {run_path}")

    predictor_path = os.path.join(run_path, 'dt_predictor.keras')
    detector_path = os.path.join(run_path, 'dt_detector.keras')

    print(f"\n--- Training Predictor (Type: {args.predictor_type.upper()}) ---")
    input_cols = ['r_zero', 'x_zero', 'y_zero', 'z_zero']
    n_features = len(input_cols); n_steps = 25
    train_df_predictor = extract_dataset('datasets/train_dataset.csv', winsize=n_steps)
    X_train_predictor, y_train_predictor = prepare_predictor_data(train_df_predictor, input_cols)
    if args.predictor_type == 'lstm': predictor_model = create_lstm_predictor(n_steps, n_features)
    elif args.predictor_type == 'tft': predictor_model = create_tft_predictor(n_steps, n_features)
    predictor_model.compile(optimizer='adam', loss='mse')
    predictor_model.fit(X_train_predictor, y_train_predictor, epochs=args.epochs, batch_size=256, verbose=1)
    predictor_model.save(predictor_path)
    
    print("\n--- Training Detector ---")
    labeled_df_full = extract_dataset('datasets/test2_dataset.csv', winsize=n_steps)
    train_detector_df, validation_df = train_test_split(labeled_df_full, test_size=0.3, random_state=42, stratify=labeled_df_full['unsafe'])
    
    # Save the validation set for later use
    validation_path = os.path.join(run_path, 'validation_set.csv')
    validation_df.to_csv(validation_path, index=False)
    print(f"Validation set saved to {validation_path}")
    
    X_labeled_train, y_labeled_train_unused = prepare_predictor_data(train_detector_df, input_cols)
    predictions_train = predictor_model.predict(X_labeled_train)
    errors_train = np.mean(np.square(y_labeled_train_unused - predictions_train), axis=1)
    
    if args.mode == 'proactive': feature_cols = ['win_dist_1_10', 'win_obstacle-distance']
    elif args.mode == 'causal': feature_cols = ['dist_trend', 'win_obstacle-distance']
    else: feature_cols = ['win_obstacle-distance']
    
    detector_features_train = train_detector_df[feature_cols].copy()
    detector_features_train['error'] = errors_train
    scaler = MinMaxScaler()
    detector_input_train = scaler.fit_transform(detector_features_train)
    detector_labels_train = train_detector_df['unsafe'].values
    joblib.dump(scaler, os.path.join(run_path, 'detector_scaler.joblib')) # Save the scaler
    
    detector_model = Sequential([
            Input(shape=(detector_input_train.shape[1],)), 
            Dense(64, activation='relu'), 
            Dense(24, activation='relu'), 
            Dropout(0.4), 
            Dense(1, activation='sigmoid')
        ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    detector_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    detector_model.fit(detector_input_train, detector_labels_train, epochs=args.epochs, class_weight={False: 0.5, True: 10.0}, batch_size=256, verbose=1)
    detector_model.save(detector_path)
    
    logger.log(run_id, 'train', {'predictor': args.predictor_type, 'mode': args.mode, 'epochs': args.epochs}, run_path)
    print(f"\n=== Finished Training Run ID: {run_id} ===")

def run_evaluation(args):
    logger = ExperimentLogger()
    run_id = args.run_id
    run_path = os.path.join('experiments', f'run_{run_id}')
    if not os.path.isdir(run_path): print(f"Error: Run directory not found for run_id '{run_id}'. Please train first."); return

    print(f"=== Evaluating Models from Run ID: {run_id} (Mode: {args.mode}) on {args.eval_file} ===")
    
    predictor = tf.keras.models.load_model(os.path.join(run_path, 'dt_predictor.keras'))
    detector = tf.keras.models.load_model(os.path.join(run_path, 'dt_detector.keras'))
    scaler = joblib.load(os.path.join(run_path, 'detector_scaler.joblib')) # Load the scaler
    
    # --- Threshold Tuning on Validation Set ---
    print("\n--- Tuning Threshold on Validation Set ---")
    validation_df = extract_dataset(os.path.join(run_path, 'validation_set.csv'))
    n_steps = predictor.input_shape[1]; input_cols = ['r_zero', 'x_zero', 'y_zero', 'z_zero']
    X_val, y_val_unused = prepare_predictor_data(validation_df, input_cols)
    predictions_val = predictor.predict(X_val)
    errors_val = np.mean(np.square(y_val_unused - predictions_val), axis=1)
    
    if args.mode == 'proactive': feature_cols = ['win_dist_1_10', 'win_obstacle-distance']
    elif args.mode == 'causal': feature_cols = ['dist_trend', 'win_obstacle-distance']
    else: feature_cols = ['win_obstacle-distance']
    
    detector_features_val = validation_df[feature_cols].copy(); detector_features_val['error'] = errors_val
    detector_input_val = scaler.transform(detector_features_val)
    window_probabilities_val = detector.predict(detector_input_val).flatten()
    
    best_f1 = 0; best_threshold = 0
    val_results_df = validation_df.copy()
    flight_labels_val = val_results_df.groupby(['log_folder', 'log_name'])['unsafe'].any()
    for threshold in np.arange(0.05, 1.0, 0.05):
        val_results_df['predicted_window'] = (window_probabilities_val > threshold).astype(int)
        flight_predictions = get_consecutive_flight_predictions(val_results_df, args.consecutive_windows)
        f1 = f1_score(flight_labels_val, flight_predictions, zero_division=0)
        if f1 > best_f1: best_f1, best_threshold = f1, threshold
            
    print(f"\n--- Optimal Threshold Found: {best_threshold:.2f} (Validation F1-Score: {best_f1:.4f}) ---")
    
    # --- Final Evaluation on Test Set ---
    print(f"\n\n=== Final Evaluation on Unseen Data ({args.eval_file}) ===")
    test_df = extract_dataset(args.eval_file, winsize=n_steps)
    X_test, y_test_unused = prepare_predictor_data(test_df, input_cols)
    predictions_test = predictor.predict(X_test)
    errors_test = np.mean(np.square(y_test_unused - predictions_test), axis=1)
    
    detector_features_test = test_df[feature_cols].copy(); detector_features_test['error'] = errors_test
    detector_input_test = scaler.transform(detector_features_test)
    window_probabilities_test = detector.predict(detector_input_test).flatten()
    window_labels_test = test_df['unsafe'].values
    
    optimal_predictions = (window_probabilities_test > best_threshold).astype(int)
    
    window_metrics = print_stats("Per-Window Performance", window_labels_test, optimal_predictions)
    test_results_df = test_df.copy(); test_results_df['predicted_window'] = optimal_predictions
    final_flight_predictions = get_consecutive_flight_predictions(test_results_df, args.consecutive_windows)
    flight_metrics = print_stats(f"Per-Flight Performance (Consecutive Windows: {args.consecutive_windows})", test_df.groupby(['log_folder', 'log_name'])['unsafe'].any(), final_flight_predictions)
    
    log_details = {'mode': args.mode, 'eval_file': args.eval_file, 'optimal_threshold': best_threshold, 'per_window_metrics': window_metrics, 'per_flight_metrics': flight_metrics}
    eval_results_path = os.path.join(run_path, f"evaluation_results_{os.path.basename(args.eval_file).split('.')[0]}.json")
    with open(eval_results_path, 'w') as f: json.dump(log_details, f, indent=4)
    logger.log(run_id, 'evaluate', log_details, run_path)

def main():
    parser = argparse.ArgumentParser(description="Advanced Digital Twin Experiment Runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    modes = ['proactive', 'causal', 'active']

    parser_train = subparsers.add_parser('train', help='Train models and create validation set')
    parser_train.add_argument('--mode', choices=modes, default='active')
    parser_train.add_argument('--predictor-type', choices=['lstm', 'tft'], default='tft')
    parser_train.add_argument('--run-id', help='[Optional] Specify a run ID.')
    parser_train.add_argument('--epochs', type=int, default=50)
    parser_train.set_defaults(func=run_training)

    parser_eval = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    parser_eval.add_argument('run_id', help='The Run ID to evaluate.')
    parser_eval.add_argument('--mode', choices=modes, default='active')
    parser_eval.add_argument('--eval-file', default='datasets/test1_dataset.csv')
    parser_eval.add_argument('--consecutive-windows', type=int, default=4)
    parser_eval.set_defaults(func=run_evaluation)
    
    # Tuning function is omitted for brevity but can be added back if needed
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()