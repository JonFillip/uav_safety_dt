import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, GRU, Dense, Dropout, Concatenate, Add, Multiply
from keras.saving import register_keras_serializable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_score, recall_score
import joblib
from datetime import datetime
import csv
import json

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
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'event_type': event_type,
                'details': str(details),
                'artifact_path': artifact_path
            })

# --- Data Loading and Feature Engineering ---
def parse_string_to_list(s):
    cleaned_string = s.strip('[]')
    if not cleaned_string: return []
    return [float(item) for item in cleaned_string.split(',')]

def add_derived_kinematics(df):
    for axis in ['x', 'y', 'z']:
        df[f'vel_{axis}'] = df[axis].apply(lambda pos: np.diff(pos, prepend=pos[0]))
        df[f'accel_{axis}'] = df[f'vel_{axis}'].apply(lambda vel: np.diff(vel, prepend=vel[0]))
    return df

def extract_dataset(file_path, winsize=25):
    df = pd.read_csv(file_path)
    if 'unsafe' not in df.columns: df['unsafe'] = False
    list_columns = ["r", "x", "y", "z"]
    for col in list_columns:
        df[col] = df[col].astype(str).apply(parse_string_to_list)
    df["r_zero"] = df["r"].apply(lambda x: [val - np.mean(x) for val in x])
    df["x_zero"] = df["x"].apply(lambda x: [val - np.mean(x) for val in x])
    df["y_zero"] = df["y"].apply(lambda x: [val - np.mean(x) for val in x])
    df["z_zero"] = df["z"].apply(lambda x: [val - np.mean(x) for val in x])
    df.dropna(subset=['r_zero', 'x_zero', 'y_zero', 'z_zero', 'win_obstacle-distance'], inplace=True)
    df = df[df["r_zero"].apply(lambda x: len(x) == winsize)]
    df["distance_trend"] = df.groupby(["log_folder", "log_name"])["win_obstacle-distance"].diff(periods=5).fillna(0)
    df = add_derived_kinematics(df)
    df.reset_index(drop=True, inplace=True)
    return df

def prepare_dsg_dt_data(df, kinematic_cols, context_cols, kinematic_scaler=None, context_scaler=None):
    X_kinematics_raw = np.dstack([np.array(df[col].to_list()) for col in kinematic_cols])
    k_shape = X_kinematics_raw.shape
    k_reshaped = X_kinematics_raw.reshape(-1, k_shape[2])
    if kinematic_scaler is None:
        kinematic_scaler = MinMaxScaler()
        X_kinematics_scaled = kinematic_scaler.fit_transform(k_reshaped).reshape(k_shape)
    else:
        X_kinematics_scaled = kinematic_scaler.transform(k_reshaped).reshape(k_shape)
    context_df = pd.DataFrame()
    scalar_context_cols = [col for col in context_cols if df[col].apply(np.isscalar).all()]
    if scalar_context_cols:
        context_df = pd.concat([context_df, df[scalar_context_cols]], axis=1)
    sequence_context_cols = [col for col in context_cols if not df[col].apply(np.isscalar).all()]
    for col in sequence_context_cols:
        context_df[f'{col}_mean'] = df[col].apply(np.mean)
    X_context_raw = context_df.values
    if context_scaler is None:
        context_scaler = MinMaxScaler()
        X_context_scaled = context_scaler.fit_transform(X_context_raw)
    else:
        X_context_scaled = context_scaler.transform(X_context_raw)
    return X_kinematics_scaled, X_context_scaled, kinematic_scaler, context_scaler, list(context_df.columns)

# --- NEW: Helper functions for evaluation ---
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

def get_consecutive_flight_predictions(df, n_consecutive):
    temp_df = df[['log_folder', 'log_name', 'predicted_window']].copy()
    temp_df['consecutive_sum'] = temp_df.groupby(['log_folder', 'log_name'])['predicted_window'] \
                                        .rolling(window=n_consecutive, min_periods=1) \
                                        .sum() \
                                        .reset_index(level=[0,1], drop=True)
    temp_df['consecutive_sum'] = temp_df['consecutive_sum'].fillna(0)
    flight_predictions = temp_df.groupby(['log_folder', 'log_name'])['consecutive_sum'].max() >= n_consecutive
    return flight_predictions

# --- DSG-DT Model Architecture ---
@register_keras_serializable()
class InverseGateLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs): super(InverseGateLayer, self).__init__(**kwargs)
    def call(self, inputs): return 1.0 - inputs
    def get_config(self): return super(InverseGateLayer, self).get_config()

def create_dsg_dt_model(n_steps, n_kinematic_features, n_context_features):
    kinematics_input = Input(shape=(n_steps, n_kinematic_features), name="kinematics_input")
    context_input = Input(shape=(n_context_features,), name="context_input")
    kinematics_features = GRU(32, name="dynamics_gru")(kinematics_input)
    kinematics_features = Dense(16, activation="relu", name="dynamics_dense")(kinematics_features)
    context_features = Dense(16, activation="relu", name="context_dense_1")(context_input)
    context_features = Dense(16, activation="relu", name="context_dense_2")(context_features)
    combined_for_gate = Concatenate()([kinematics_features, context_features])
    gate = Dense(1, activation='sigmoid', name='fusion_gate')(combined_for_gate)
    gated_kinematics = Multiply()([gate, kinematics_features])
    inverse_gate = InverseGateLayer()(gate)
    gated_context = Multiply()([inverse_gate, context_features])
    fused_features = Add()([gated_kinematics, gated_context])
    x = Dropout(0.4)(fused_features)
    x = Dense(16, activation='relu')(x)
    output = Dense(1, activation='sigmoid', name='output')(x)
    model = Model(inputs=[kinematics_input, context_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Main Experiment Functions ---
def run_dsg_dt_training(args):
    logger = ExperimentLogger()
    run_id = args.run_id or f"dsg_dt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_path = os.path.join('experiments', run_id)
    os.makedirs(run_path, exist_ok=True)
    print(f"=== Starting DSG-DT Training Run ID: {run_id} ===")

    kinematic_cols = ['r_zero', 'x_zero', 'y_zero', 'z_zero']
    context_cols = ['distance_trend', 'vel_x', 'vel_y', 'vel_z', 'accel_x', 'accel_y', 'accel_z']
    n_steps = 25
    
    print("Loading and preparing training and validation data...")
    full_labeled_df = extract_dataset(args.labeled_file, winsize=n_steps)
    train_df, validation_df = train_test_split(full_labeled_df, test_size=0.3, random_state=42, stratify=full_labeled_df['unsafe'])
    
    validation_path = os.path.join(run_path, 'validation_set.csv')
    validation_df.to_csv(validation_path, index=False)
    print(f"Validation set saved to {validation_path}")
    
    X_kinematics, X_context, k_scaler, c_scaler, final_context_cols = prepare_dsg_dt_data(train_df, kinematic_cols, context_cols)
    y_train = train_df['unsafe'].values
    
    dsg_dt_model = create_dsg_dt_model(n_steps, len(kinematic_cols), X_context.shape[1])
    dsg_dt_model.summary()
    
    print("Training the DSG-DT model...")
    dsg_dt_model.fit(
        [X_kinematics, X_context], y_train, 
        epochs=args.epochs, batch_size=256, validation_split=0.2, class_weight={0: 1., 1: 10.}, verbose=1
    )
    
    model_path = os.path.join(run_path, 'dsg_dt_model.keras')
    k_scaler_path = os.path.join(run_path, 'kinematic_scaler.joblib')
    c_scaler_path = os.path.join(run_path, 'context_scaler.joblib')
    
    dsg_dt_model.save(model_path)
    joblib.dump(k_scaler, k_scaler_path)
    joblib.dump(c_scaler, c_scaler_path)
    print(f"Model and scalers saved in {run_path}")

    log_details = {'epochs': args.epochs, 'kinematic_cols': kinematic_cols, 'context_cols': final_context_cols}
    logger.log(run_id, 'train', log_details, run_path)
    print(f"\n=== Finished DSG-DT Training Run ID: {run_id} ===")

def run_dsg_dt_evaluation(args):
    logger = ExperimentLogger()
    run_id = args.run_id
    run_path = os.path.join('experiments', run_id)
    if not os.path.isdir(run_path):
        print(f"Error: Run directory not found for run_id '{run_id}'.")
        return

    print(f"=== Evaluating DSG-DT from Run ID: {run_id} ===")
    
    model = tf.keras.models.load_model(os.path.join(run_path, 'dsg_dt_model.keras'))
    k_scaler = joblib.load(os.path.join(run_path, 'kinematic_scaler.joblib'))
    c_scaler = joblib.load(os.path.join(run_path, 'context_scaler.joblib'))

    kinematic_cols = ['r_zero', 'x_zero', 'y_zero', 'z_zero']
    context_cols = ['distance_trend', 'vel_x', 'vel_y', 'vel_z', 'accel_x', 'accel_y', 'accel_z']
    
    print("\n--- Tuning Threshold on Validation Set ---")
    validation_df = extract_dataset(os.path.join(run_path, 'validation_set.csv'))
    X_kin_val, X_con_val, _, _, _ = prepare_dsg_dt_data(validation_df, kinematic_cols, context_cols, k_scaler, c_scaler)
    y_val_labels = validation_df['unsafe'].values
    val_probabilities = model.predict([X_kin_val, X_con_val]).flatten()

    best_f1 = 0
    best_threshold = 0
    for threshold in np.arange(0.05, 1.0, 0.05):
        preds = (val_probabilities > threshold).astype(int)
        f1 = f1_score(y_val_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            
    print(f"\n--- Optimal Threshold Found: {best_threshold:.2f} (Validation F1-Score: {best_f1:.4f}) ---")
    
    print(f"\n\n=== Final Evaluation on Unseen Data ({args.eval_file}) ===")
    test_df = extract_dataset(args.eval_file)
    X_kin_test, X_con_test, _, _, _ = prepare_dsg_dt_data(test_df, kinematic_cols, context_cols, k_scaler, c_scaler)
    
    y_test_labels = test_df['unsafe'].values
    test_probabilities = model.predict([X_kin_test, X_con_test]).flatten()
    final_preds = (test_probabilities > best_threshold).astype(int)
    
    # --- UPDATED: Using print_stats for both window and flight performance ---
    window_metrics = print_stats("Per-Window Performance", y_test_labels, final_preds)

    results_df = test_df.copy()
    results_df['predicted_window'] = final_preds
    flight_labels = results_df.groupby(['log_folder', 'log_name'])['unsafe'].any()
    flight_predictions = get_consecutive_flight_predictions(results_df, args.consecutive_windows)
    flight_metrics = print_stats(f"Per-Flight Performance (Consecutive Windows: {args.consecutive_windows})", flight_labels, flight_predictions)

    log_details = {
        'eval_file': args.eval_file, 
        'best_threshold': best_threshold, 
        'window_metrics': window_metrics,
        'flight_metrics': flight_metrics
    }
    logger.log(run_id, 'evaluate', log_details, run_path)

def main():
    parser = argparse.ArgumentParser(description="Final DSG-DT Experiment Runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_train = subparsers.add_parser('train', help='Train the DSG-DT model.')
    parser_train.add_argument('--run-id', help='[Optional] Specify a custom run ID.')
    parser_train.add_argument('--labeled-file', default='datasets/train_dataset.csv', help="Dataset for training and creating validation split.")
    parser_train.add_argument('--epochs', type=int, default=50)
    parser_train.set_defaults(func=run_dsg_dt_training)

    parser_eval = subparsers.add_parser('evaluate', help='Evaluate a trained DSG-DT model.')
    parser_eval.add_argument('run_id', help='The Run ID of the trained model to evaluate.')
    parser_eval.add_argument('--eval-file', default='datasets/test1_dataset.csv')
    parser_eval.add_argument('--consecutive-windows', type=int, default=4, help='Number of consecutive positive windows to flag a flight.')
    parser_eval.set_defaults(func=run_dsg_dt_evaluation)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()