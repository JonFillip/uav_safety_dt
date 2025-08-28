import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Flatten, Dense, LSTM, MultiHeadAttention, LayerNormalization, Add, GRU, TimeDistributed, Concatenate, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from datetime import datetime
import csv
import json

# --- NEW: Custom Layer Definition for GatedLinearUnit ---
class GatedLinearUnit(tf.keras.layers.Layer):
    def __init__(self, units):
        super(GatedLinearUnit, self).__init__()
        self.linear = Dense(units)
        self.sigmoid = Dense(units, activation='sigmoid')

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)

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

def extract_enriched_window_data(file_path, winsize=25):
    df = pd.read_csv(file_path)
    if 'unsafe' not in df.columns:
        df['unsafe'] = False
    
    df["r"] = df["r"].astype(str).apply(lambda x: [float(val) for val in x.split(",")])
    df["x"] = df["x"].astype(str).apply(lambda x: [float(val) for val in x.split(",")])
    df["y"] = df["y"].astype(str).apply(lambda x: [float(val) for val in x.split(",")])
    df["z"] = df["z"].astype(str).apply(lambda x: [float(val) for val in x.split(",")])
    df["r_zero"] = df["r"].apply(lambda x: [val - np.mean(x) for val in x])
    df["x_zero"] = df["x"].apply(lambda x: [val - np.mean(x) for val in x])
    df["y_zero"] = df["y"].apply(lambda x: [val - np.mean(x) for val in x])
    df["z_zero"] = df["z"].apply(lambda x: [val - np.mean(x) for val in x])
    df = df.dropna(subset=['r', 'x', 'y', 'z', 'win_obstacle-distance'])
    df.reset_index(drop=True, inplace=True)
    df = df[df["r"].apply(lambda x: len(x) == winsize)]

    for col in ['win_obstacle-distance']:
        df[col] = df[col].apply(lambda x: [x] * winsize)

    return df

from advanced_dt_experiment import print_stats

def prepare_end_to_end_data(df, input_cols):
    data_list = [np.array(df[col].to_list()) for col in input_cols]
    X = np.stack(data_list, axis=-1)
    y = df['unsafe'].values
    return X, y

# --- Custom Layers for SOTA TFT ---
class GatedResidualNetwork(tf.keras.layers.Layer):
    def __init__(self, units, dropout_rate=0.1, **kwargs):
        super(GatedResidualNetwork, self).__init__(**kwargs)
        self.units = units
        self.dense1 = Dense(units, activation="relu")
        self.dense2 = Dense(units)
        self.gate = Dense(units, activation="sigmoid")
        self.norm = LayerNormalization()
        self.dropout = Dropout(dropout_rate)
        self.skip = Dense(units)  # projection for skip connection

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        x = self.dense2(x)

        # gating
        g = self.gate(inputs)
        x = x * g

        # skip connection
        s = self.skip(inputs)

        return self.norm(Add()([x, s]))

class VariableSelectionNetwork(tf.keras.layers.Layer):
    def __init__(self, num_features, units, dropout_rate):
        super(VariableSelectionNetwork, self).__init__()
        self.grns = [GatedResidualNetwork(units, dropout_rate) for _ in range(num_features)]
        self.softmax = Dense(num_features, activation='softmax')

    def call(self, inputs):
        v = Concatenate(axis=-1)([grn(tf.expand_dims(inputs[..., i], axis=-1)) for i, grn in enumerate(self.grns)])
        return self.softmax(v)

# --- Model Definitions ---
def create_cnn_classifier(n_steps=25, n_features=5):
    model = Sequential([
        Input(shape=(n_steps, n_features)),
        Conv1D(filters=48, kernel_size=6, activation="relu"),
        MaxPooling1D(pool_size=3),
        BatchNormalization(),
        Conv1D(filters=64, kernel_size=2, activation="relu"),
        MaxPooling1D(pool_size=3),
        BatchNormalization(),
        Flatten(),
        Dense(112, activation="tanh"),
        Dense(80, activation="tanh"),
        Dense(1, activation="sigmoid")
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model

def create_sota_tft_classifier(n_steps=25, n_features=5, hidden_units=64, dropout_rate=0.1):
    """Creates a full, state-of-the-art TFT for classification."""
    inputs = Input(shape=(n_steps, n_features))
    
    vsn = VariableSelectionNetwork(n_features, hidden_units, dropout_rate)
    selected_features = vsn(inputs)
    
    processed_features = GatedResidualNetwork(hidden_units, dropout_rate)(selected_features)

    lstm_out = GRU(hidden_units, return_sequences=True)(processed_features)
    
    attention_temp = GatedResidualNetwork(hidden_units, dropout_rate)(lstm_out)
    attention_out = MultiHeadAttention(num_heads=4, key_dim=hidden_units)(attention_temp, attention_temp)
    attention_out = Add()([attention_temp, attention_out])
    attention_out = LayerNormalization()(attention_out)
    
    final_processing = GatedResidualNetwork(hidden_units, dropout_rate)(attention_out)
    flattened = Flatten()(final_processing)
    outputs = Dense(1, activation='sigmoid')(flattened)
    
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_lstm_classifier(n_steps=25, n_features=5):
    """Creates an end-to-end LSTM classifier."""
    model = Sequential([
        Input(shape=(n_steps, n_features)),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model

# --- Main Experiment Function ---
def run_baseline_experiment(args):
    logger = ExperimentLogger()
    run_id = args.run_id or f"baseline_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_path = os.path.join('experiments', run_id)
    os.makedirs(run_path, exist_ok=True)
    
    print(f"=== Starting Baseline Experiment Run ID: {run_id} ===")
    print(f"All artifacts for this run will be saved in: {run_path}")

    input_cols = ['r_zero', 'x_zero', 'y_zero', 'z_zero', 'win_obstacle-distance']
    n_steps = 25
    n_features = len(input_cols)
    
    labeled_df_full = extract_enriched_window_data('datasets/test2_dataset.csv', winsize=n_steps)
    train_df, validation_df = train_test_split(
        labeled_df_full, 
        train_size=args.train_size, 
        random_state=42, 
        stratify=labeled_df_full['unsafe']
    )
    print(f"\n--- Using {args.train_size*100:.0f}% of labeled data for training ({len(train_df)} samples) ---")
    
    X_train, y_train = prepare_end_to_end_data(train_df, input_cols)

    if args.model_type == 'cnn':
        model = create_cnn_classifier(n_steps, n_features)
    elif args.model_type == 'tft':
        model = create_sota_tft_classifier(n_steps, n_features)
    elif args.model_type == 'lstm':
        model = create_lstm_classifier(n_steps, n_features)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
        
    model.summary()
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    print(f"Using class weights: {class_weight_dict}")
    
    model.fit(X_train, y_train, epochs=args.epochs, batch_size=256, class_weight=class_weight_dict, verbose=1)
    
    model_path = os.path.join(run_path, f'{args.model_type}_baseline.keras')
    model.save(model_path)
    logger.log(run_id, 'train', {'model_type': args.model_type, 'epochs': args.epochs}, run_path)
    
    print(f"\n\n=== Tuning Threshold on Validation Set (from test1_dataset.csv) ===")
    X_val, y_val = prepare_end_to_end_data(validation_df, input_cols)
    best_threshold = tune_threshold(model, X_val, y_val)

    print(f"\n\n=== Evaluating Baseline on Final Unseen Data (test1_dataset.csv) ===")
    final_eval_df = extract_enriched_window_data('datasets/test1_dataset.csv', winsize=n_steps)
    X_test, y_test = prepare_end_to_end_data(final_eval_df, input_cols)
    evaluate_model(model, X_test, y_test, final_eval_df, best_threshold, run_id, logger, run_path, 'test1')

def tune_threshold(model, X_val, y_val):
    window_probabilities = model.predict(X_val).flatten()
    best_f1 = 0
    best_threshold = 0
    for threshold in np.arange(0.05, 1.0, 0.05):
        preds = (window_probabilities > threshold).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    print(f"--- Optimal Threshold Found: {best_threshold:.2f} (Validation F1-Score: {best_f1:.4f}) ---")
    return best_threshold

def evaluate_model(model, X_test, y_test, eval_df, threshold, run_id, logger, run_path, eval_name):
    window_probabilities = model.predict(X_test).flatten()
    
    optimal_predictions = (window_probabilities > threshold).astype(int)
    window_metrics = print_stats("Per-Window Performance", y_test, optimal_predictions)

    eval_results_df = eval_df.copy()
    eval_results_df['predicted'] = optimal_predictions
    flight_labels = eval_results_df.groupby(['log_folder', 'log_name'])['unsafe'].any()
    flight_predictions = eval_results_df.groupby(['log_folder', 'log_name'])['predicted'].any()
    flight_metrics = print_stats("Per-Flight (Log-Based) Performance", flight_labels, flight_predictions)

    log_details = {'eval_file': eval_name, 'optimal_threshold': threshold, 'per_window_metrics': window_metrics, 'per_flight_metrics': flight_metrics}
    
    results_path = os.path.join(run_path, f"evaluation_results_{eval_name}.json")
    with open(results_path, 'w') as f: json.dump(log_details, f, indent=4)
    logger.log(run_id, f'evaluate_{eval_name}', log_details, run_path)

def main():
    parser = argparse.ArgumentParser(description="End-to-End Baseline Experiment Runner")
    parser.add_argument('--model-type', choices=['cnn', 'tft', 'lstm'], default='cnn', help='Type of baseline model to run')
    parser.add_argument('--train-size', type=float, default=0.7, help='Proportion of labeled data to use for training (e.g., 0.1 for 10%)')
    parser.add_argument('--run-id', help='[Optional] Specify a run ID.')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    run_baseline_experiment(args)

if __name__ == "__main__":
    main()