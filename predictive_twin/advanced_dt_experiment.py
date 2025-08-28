import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Flatten, Dense, LSTM, MultiHeadAttention, LayerNormalization, Add, GRU, TimeDistributed, Concatenate, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from keras_tuner import BayesianOptimization
from keras.saving import register_keras_serializable
from datetime import datetime
import csv
import json
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import f_oneway


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
    """Calculates the change in distance over a past window."""
    distances = group["win_obstacle-distance"]
    # The trend is the current distance minus the distance 'window_size' steps ago.
    # A negative value means the drone is getting closer.
    trend = distances.diff(periods=window_size)
    return trend.fillna(0) # Fill initial NaNs with 0


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

# --- Visualization Helper Function ---
def plot_loss_history(history, save_path):
    """Plots and saves the training and validation loss."""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Detector Model Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
    plt.close()
    print(f"Training history plot saved to {save_path}")

def plot_roc_curve(y_true, y_pred_proba, save_path):
    """Plots and saves the AUC-ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
    plt.close()
    print(f"AUC-ROC curve plot saved to {save_path}")

# --- Consecutive Window Rule ---
def get_consecutive_flight_predictions(df, n_consecutive):
    """Aggregates window predictions based on a consecutive rule."""
    # Create a temporary DataFrame with only the necessary columns
    temp_df = df[['log_folder', 'log_name', 'predicted_window']].copy()
    
    # Calculate the rolling sum within each group
    # The result has a MultiIndex, which we reset to align
    temp_df['consecutive_sum'] = temp_df.groupby(['log_folder', 'log_name'])['predicted_window'] \
                                        .rolling(window=n_consecutive) \
                                        .sum() \
                                        .reset_index(level=[0,1], drop=True)
    
    # Fill the initial NaN values that result from the rolling window
    temp_df['consecutive_sum'] = temp_df['consecutive_sum'].fillna(0)

    # If the max sum for a flight is ever >= N, that flight is unsafe
    flight_predictions = temp_df.groupby(['log_folder', 'log_name'])['consecutive_sum'].max() >= n_consecutive
    
    return flight_predictions

# --- Predictor Model Definitions ---
def create_lstm_predictor(n_steps, n_features):
    return Sequential([Input(shape=(n_steps, n_features)), LSTM(50, return_sequences=False), Dense(n_features)])

@register_keras_serializable()
class GatedLinearUnit(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(GatedLinearUnit, self).__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        self.linear = Dense(self.units)
        self.sigmoid = Dense(self.units, activation='sigmoid')
        super(GatedLinearUnit, self).build(input_shape)

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)
    
    def get_config(self):
        config = super(GatedLinearUnit, self).get_config()
        config.update({'units': self.units})
        return config

@register_keras_serializable()
class GatedResidualNetwork(tf.keras.layers.Layer):
    def __init__(self, units, dropout_rate, **kwargs):
        super(GatedResidualNetwork, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # Define all sub-layers here to resolve the build warning
        self.dense1 = Dense(self.units, activation="relu")
        self.dense2 = Dense(self.units)
        self.dropout = Dropout(self.dropout_rate)
        # The 'gate' is a Dense layer with a sigmoid activation
        self.gate = Dense(self.units, activation="sigmoid") 
        self.skip_projection = Dense(self.units)
        self.norm = LayerNormalization()
        self.add = Add()
        super(GatedResidualNetwork, self).build(input_shape)

    def call(self, inputs, training=False):
        # Main data path
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        x = self.dense2(x)

        # Gating mechanism
        g = self.gate(inputs)
        x = x * g

        # Skip connection
        s = self.skip_projection(inputs)

        # Add & Norm
        return self.norm(self.add([x, s]))

    def get_config(self):
        config = super(GatedResidualNetwork, self).get_config()
        config.update({
            'units': self.units,
            'dropout_rate': self.dropout_rate
        })
        return config

@register_keras_serializable()
class VariableSelectionNetwork(tf.keras.layers.Layer):
    def __init__(self, num_features, units, dropout_rate, **kwargs):
        super(VariableSelectionNetwork, self).__init__(**kwargs)
        self.num_features = num_features
        self.units = units
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.grns = [GatedResidualNetwork(self.units, self.dropout_rate) for _ in range(self.num_features)]
        self.softmax = Dense(self.num_features, activation='softmax')
        super(VariableSelectionNetwork, self).build(input_shape)

    def call(self, inputs):
        v = Concatenate(axis=-1)([grn(tf.expand_dims(inputs[..., i], axis=-1)) for i, grn in enumerate(self.grns)])
        return self.softmax(v)
    
    def get_config(self):
        config = super(VariableSelectionNetwork, self).get_config()
        config.update({'num_features': self.num_features, 'units': self.units, 'dropout_rate': self.dropout_rate})
        return config

def create_tft_predictor(n_steps=25, n_features=5, hidden_units=64, dropout_rate=0.1):
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

def get_config(self):
        config = super(VariableSelectionNetwork, self).get_config()
        config.update({'num_features': self.num_features, 'units': self.units, 'dropout_rate': self.dropout_rate})
        return config

def create_cnn_predictor(n_steps, n_features):
    """Creates a 1D-CNN model for state prediction."""
    return Sequential([
        Input(shape=(n_steps, n_features)),
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(n_features)
    ])


# --- Main Experiment Functions ---
def run_training(args):
    logger = ExperimentLogger()
    run_id = args.run_id or f"dt_{args.predictor_type}_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_path = os.path.join('experiments', run_id)
    os.makedirs(run_path, exist_ok=True)
    
    print(f"=== Starting Training Run ID: {run_id} (Mode: {args.mode}) ===")
    print(f"All artifacts will be saved in: {run_path}")

    predictor_path = os.path.join(run_path, 'dt_predictor.keras')
    detector_path = os.path.join(run_path, 'dt_detector.keras')

    # Subnetwork 1: The Predictor (Learns normal flight behavior)
    # Input: Drone kinematics
    # Output: Predicted next state, an similarity_error (anomaly error)
    print(f"\n--- Training Predictor (Type: {args.predictor_type.upper()}) ---")
    input_cols = ['r_zero', 'x_zero', 'y_zero', 'z_zero'] # , 'x_zero', 'y_zero', 'z_zero'
    n_features = len(input_cols); n_steps = 25
    train_df = extract_dataset('datasets/train_dataset.csv', winsize=n_steps)
    X_train_predictor, y_train_predictor = prepare_predictor_data(train_df, input_cols)
    
    if args.predictor_type == 'lstm': 
        predictor_model = create_lstm_predictor(n_steps, n_features)
    elif args.predictor_type == 'tft': 
        predictor_model = create_tft_predictor(n_steps, n_features)
    elif args.predictor_type == 'cnn':
        predictor_model = create_cnn_predictor(n_steps, n_features)
    else: 
        raise ValueError("Invalid predictor type specified.")
        
    predictor_model.compile(optimizer='adam', loss='mse')
    predictor_model.fit(X_train_predictor, y_train_predictor, epochs=args.epochs, batch_size=256, verbose=1)
    predictor_model.save(predictor_path)

    # Subnetwork 2: The Detectors
    # Input: current distance to obstacle + Optional Params (X_last_steps_win_obstacle or X_future_steps_win_obstacle)
    # Output: Risky Score, Best threshold, Safety classification
    print("\n--- Training Detector ---")
    labeled_df_full = extract_dataset('datasets/test2_dataset.csv', winsize=n_steps)
    train_detector_df, validation_df = train_test_split(labeled_df_full, test_size=0.2, random_state=42, stratify=labeled_df_full['unsafe'])
    
    # Save the validation set for later use
    validation_path = os.path.join(run_path, 'validation_set.csv')
    validation_df.to_csv(validation_path, index=False)
    print(f"Validation set saved to {validation_path}")
    
    X_labeled_train, y_labeled_train_unused = prepare_predictor_data(train_detector_df, input_cols)
    predictions_train = predictor_model.predict(X_labeled_train)
    errors_train = np.mean(np.square(y_labeled_train_unused - predictions_train), axis=1)
    
    if args.mode == 'causal':
        feature_cols = ['dist_trend']
    elif args.mode == 'proactive':
        feature_cols = ['win_dist_1_10']
    
    detector_features_train = train_detector_df[feature_cols].copy()
    # detector_features_train['error'] = errors_train
    scaler = MinMaxScaler()
    detector_input_train = scaler.fit_transform(detector_features_train)
    detector_labels_train = train_detector_df['unsafe'].values
    joblib.dump(scaler, os.path.join(run_path, 'detector_scaler.joblib')) # Save the scaler
    
    detector_model = Sequential([
        Input(shape=(detector_input_train.shape[1],)),
        Dense(64, activation='relu'),   # Updated from 56 (units_1: 64)
        Dense(24, activation='relu'),  # Updated from 8 (units_2: 24)
        Dropout(0.3),                  # Updated from 0.5 (dropout: 0.4)
        Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) 
    detector_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    detector_model.fit(detector_input_train, detector_labels_train, epochs=args.epochs, class_weight={False: 0.5, True: 10.0}, batch_size=256, verbose=1)
    detector_model.save(detector_path)
    
    logger.log(run_id, 'train', {'predictor': args.predictor_type, 'mode': args.mode, 'epochs': args.epochs}, run_path)
    print(f"\n=== Finished Training Run ID: {run_id} ===")

    # plot_loss_history(history, os.path.join(run_path, 'detector_loss_history.png'))


def run_evaluation(args):
    logger = ExperimentLogger()
    run_id = args.run_id or f"dt_{args.predictor_type}_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_path = os.path.join('experiments', run_id)
    
    if not os.path.isdir(run_path):
        print(f"Error: Run directory not found for run_id '{run_id}'. Please train first.")
        return

    print(f"=== Evaluating Models from Run ID: {run_id} on {args.eval_file} ===")
    
    predictor = tf.keras.models.load_model(os.path.join(run_path, 'dt_predictor.keras'))
    detector = tf.keras.models.load_model(os.path.join(run_path, 'dt_detector.keras'))
    scaler = joblib.load(os.path.join(run_path, 'detector_scaler.joblib'))
    
    # --- Threshold Tuning on Validation Set ---
    print("\n--- Tuning Threshold on Validation Set ---")
    validation_df = extract_dataset(os.path.join(run_path, 'validation_set.csv'))
    n_steps = predictor.input_shape[1]; input_cols = ['r_zero', 'x_zero', 'y_zero', 'z_zero'] # , 'x_zero', 'y_zero', 'z_zero'
    X_val, y_val_unused = prepare_predictor_data(validation_df, input_cols)
    predictions_val = predictor.predict(X_val)
    errors_val = np.mean(np.square(y_val_unused - predictions_val), axis=1)
    
    if args.mode == 'causal':
        feature_cols = ['dist_trend']
    elif args.mode == 'proactive':
        feature_cols = ['win_dist_1_10']
        
    detector_features_val = validation_df[feature_cols].copy()
    # detector_features_val['error'] = errors_val
    detector_input_val = scaler.transform(detector_features_val)
    window_probabilities_val = detector.predict(detector_input_val).flatten()
    window_labels_val = validation_df['unsafe'].values

    # --- UPDATED: Threshold tuning loop now optimizes for PER-WINDOW F1-Score ---
    best_f1 = 0
    best_threshold = 0
    
    for threshold in np.arange(0.05, 1.0, 0.05):
        preds = (window_probabilities_val > threshold).astype(int)
        f1 = f1_score(window_labels_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            
    print(f"\n--- Optimal Threshold Found: {best_threshold:.2f} (Validation F1-Score: {best_f1:.4f}) ---")
    
    # --- Final Evaluation on Test Set ---
    print(f"\n\n=== Final Evaluation on Unseen Data ({args.eval_file}) ===")
    test_df = extract_dataset(args.eval_file, winsize=n_steps)
    X_test, y_test_unused = prepare_predictor_data(test_df, input_cols)
    predictions_test = predictor.predict(X_test)
    errors_test = np.mean(np.square(y_test_unused - predictions_test), axis=1)
    
    detector_features_test = test_df[feature_cols].copy() 
    # detector_features_test['error'] = errors_test
    detector_input_test = scaler.transform(detector_features_test)
    window_probabilities_test = detector.predict(detector_input_test).flatten()
    window_labels_test = test_df['unsafe'].values
    
    optimal_predictions = (window_probabilities_test > best_threshold).astype(int)
    
    window_metrics = print_stats("Per-Window Performance", window_labels_test, optimal_predictions)
    test_results_df = test_df.copy()
    test_results_df['predicted_window'] = optimal_predictions
    final_flight_predictions = get_consecutive_flight_predictions(test_results_df, args.consecutive_windows)
    flight_metrics = print_stats(f"Per-Flight Performance (Consecutive Windows: {args.consecutive_windows})", test_df.groupby(['log_folder', 'log_name'])['unsafe'].any(), final_flight_predictions)
    
    log_details = {
        'mode': args.mode, 
        'eval_file': args.eval_file, 
        'optimal_threshold': best_threshold, 
        'per_window_metrics': window_metrics, 
        'per_flight_metrics': flight_metrics
        }
    eval_results_path = os.path.join(run_path, f"evaluation_results_{os.path.basename(args.eval_file).split('.')[0]}.json")
    with open(eval_results_path, 'w') as f: json.dump(log_details, f, indent=4)
    logger.log(run_id, 'evaluate', log_details, run_path)

def detector_model_builder(hp, n_features):
    # n_features = hp.Int('n_features', min_value=2, max_value=3)
    model = Sequential([
        Input(shape=(n_features,)),
        Dense(units=hp.Int('units_1', 16, 64, 16), activation='relu'),
        Dropout(hp.Float('dropout', 0.1, 0.5, 0.1)),
        Dense(units=hp.Int('units_2', 8, 32, 8), activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-3, 1e-4])), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def run_tuning(args):
    logger = ExperimentLogger()
    run_id = args.run_id
    run_path = os.path.join('experiments', run_id)
    if not os.path.isdir(run_path):
        print(f"Error: Run directory not found for run_id '{run_id}'. Please train the base models for this run first.")
        return

    print(f"=== Tuning Detector for Run ID: {run_id} ===")
    
    n_steps = 25
    input_cols = ['r_zero', 'x_zero', 'y_zero', 'z_zero']
    
    predictor_model = tf.keras.models.load_model(os.path.join(run_path, 'dt_predictor.keras'))
    
    labeled_df = extract_dataset('datasets/test2_dataset.csv', winsize=n_steps)
    X_labeled, y_labeled_unused = prepare_predictor_data(labeled_df, input_cols)
    predictions = predictor_model.predict(X_labeled)
    errors = np.mean(np.square(y_labeled_unused - predictions), axis=1)
    
    if args.mode == 'causal':
        feature_cols = ['dist_trend']
    elif args.mode == 'proactive':
        feature_cols = ['win_dist_1_10']
    
        
    detector_features = labeled_df[feature_cols].copy()
    # detector_features['error'] = errors
    
    scaler = MinMaxScaler()
    detector_input = scaler.fit_transform(detector_features)
    detector_labels = labeled_df['unsafe'].values
    
    X_train, X_val, y_train, y_val = train_test_split(detector_input, detector_labels, test_size=0.2, random_state=42, stratify=detector_labels)
    
    tuner_dir = os.path.join(run_path, 'tuning')
    # Get the number of features from the data
    n_features = detector_input.shape[1]
    
    # Correctly instantiate the tuner, passing n_features to the model builder
    tuner = BayesianOptimization(
        lambda hp: detector_model_builder(hp, n_features=n_features), # Pass n_features here
        objective='val_accuracy',
        max_trials=args.trials,
        directory=tuner_dir,
        project_name='detector_tuning',
        overwrite=True
    )

    class_weight_dict = {False: 0.5, True: 10.0}
    tuner.search(X_train, y_train, epochs=30, validation_data=(X_val, y_val), class_weight=class_weight_dict)
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\n--- Best Hyperparameters Found ---")
    for param, value in best_hps.values.items(): print(f"{param}: {value}")
        
    results_path = os.path.join(tuner_dir, 'best_hyperparameters.json')
    with open(results_path, 'w') as f: json.dump(best_hps.values, f, indent=4)
    logger.log(run_id, 'tune', best_hps.values, run_path)

def analyze_uncertainty(args):
    """Analyzes the correlation between predictor error and the 'uncertain' label."""
    print(f"--- Analyzing Uncertainty for Run ID: {args.run_id} ---")
    run_path = os.path.join('experiments', args.run_id)
    if not os.path.isdir(run_path):
        print(f"Error: Run directory not found for run_id '{args.run_id}'.")
        return

    # Load the trained predictor model
    predictor = tf.keras.models.load_model(os.path.join(run_path, 'dt_predictor.keras'))
    n_steps = predictor.input_shape[1]
    
    # This list must match the features the loaded model was trained on.
    input_cols = ['r_zero', 'x_zero', 'y_zero', 'z_zero'] 

    # Load a dataset with uncertain labels
    print(f"Loading and processing {args.eval_file}...")
    df = extract_dataset(args.eval_file, winsize=n_steps)

    # Get prediction errors from the predictor
    X_data, y_data_unused = prepare_predictor_data(df, input_cols)
    predictions = predictor.predict(X_data)
    errors = np.mean(np.square(y_data_unused - predictions), axis=1)

    # Create a new DataFrame for analysis
    analysis_df = pd.DataFrame({
        'prediction_error': errors,
        'is_uncertain': df['uncertain']
    })

    # --- Correlation and Statistical Analysis ---
    print("\n--- Correlation and Significance Analysis ---")
    
    # 1. Linear Correlation (Point-Biserial)
    analysis_df['is_uncertain_numeric'] = analysis_df['is_uncertain'].astype(int)
    correlation_score = analysis_df['prediction_error'].corr(analysis_df['is_uncertain_numeric'])
    print(f"Linear Correlation (Point-Biserial) Score: {correlation_score:.4f}")
    
    # 2. Non-Linear Correlation (Mutual Information)
    X_feature = analysis_df[['prediction_error']]
    y_labels = analysis_df['is_uncertain']
    mi_score = mutual_info_classif(X_feature, y_labels, discrete_features=False, random_state=42)[0]
    print(f"Non-Linear Correlation (Mutual Info) Score: {mi_score:.4f}")
    
    # 3. Statistical Significance Test (ANOVA)
    errors_certain = analysis_df[analysis_df['is_uncertain'] == False]['prediction_error']
    errors_uncertain = analysis_df[analysis_df['is_uncertain'] == True]['prediction_error']
    f_statistic, p_value = f_oneway(errors_certain, errors_uncertain)
    print(f"ANOVA F-statistic: {f_statistic:.2f} | P-value: {p_value:.4e}")
    if p_value < 0.05:
        print("Conclusion: The difference in error between groups is statistically significant.")
    else:
        print("Conclusion: The difference in error between groups is not statistically significant.")

    # --- Visualization ---
    print("\nGenerating visualization...")
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='is_uncertain', y='prediction_error', data=analysis_df)
    plt.yscale('log')
    title = f"Prediction Error Distribution on {os.path.basename(args.eval_file)}"
    plt.title(title)
    plt.xlabel('Is the Window Labeled as Uncertain?')
    plt.ylabel('Prediction Error (MSE) - Log Scale')
    plt.grid(True)
    
    plot_path = os.path.join(run_path, f"uncertainty_correlation_plot_{os.path.basename(args.eval_file).split('.')[0]}.png")
    plt.savefig(plot_path)
    plt.show()
    plt.close()

    print(f"\nAnalysis complete. Plot saved to: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Advanced Digital Twin Experiment Runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    modes = ['proactive', 'causal', 'active']
    predictors = ['lstm', 'tft', 'cnn']

    parser_train = subparsers.add_parser('train', help='Train the two-stage model and create a new run directory')
    parser_train.add_argument('--mode', choices=modes, default='active', help='Feature mode for the detector.')
    parser_train.add_argument('--predictor-type', choices=predictors, default='tft', help='Type of predictor model to train.')
    parser_train.add_argument('--run-id', help='[Optional] Specify a run ID.')
    parser_train.add_argument('--epochs', type=int, default=50)
    parser_train.set_defaults(func=run_training)

    parser_eval = subparsers.add_parser('evaluate', help='Evaluate a trained model from a specific run')
    parser_eval.add_argument('run_id', help='The Run ID of the trained models to evaluate.')
    parser_eval.add_argument('--mode', choices=modes, default='active', help='Feature mode to use for evaluation.')
    parser_eval.add_argument('--eval-file', default='datasets/test1_dataset.csv')
    parser_eval.add_argument('--consecutive-windows', type=int, default=4, help='Number of consecutive positive windows to flag a flight.')
    parser_eval.set_defaults(func=run_evaluation)
    
    parser_tune = subparsers.add_parser('tune', help='Tune detector hyperparameters for a specific run')
    parser_tune.add_argument('--mode', choices=modes, default='active', help='Feature mode for the detector.')
    parser_tune.add_argument('run_id', help='The Run ID to associate this tuning session with.')
    parser_tune.add_argument('--trials', type=int, default=20)
    parser_tune.set_defaults(func=run_tuning)

    parser_analyze = subparsers.add_parser('analyze', help='Analyze the correlation between predictor error and uncertainty.')
    parser_analyze.add_argument('run_id', help='The Run ID of the trained models to analyze.')
    parser_analyze.add_argument('--eval-file', default='datasets/test1_dataset.csv')
    parser_analyze.set_defaults(func=analyze_uncertainty)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()