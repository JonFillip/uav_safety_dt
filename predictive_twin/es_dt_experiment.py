import argparse
import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D
from keras.saving import register_keras_serializable
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, classification_report
import joblib
from datetime import datetime
import mlflow
import mlflow.keras

# --- Data Loading and Helper Functions ---
def parse_string_to_list(s):
    cleaned_string = s.strip('[]')
    if not cleaned_string: return []
    return [float(item) for item in cleaned_string.split(',')]

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
    df.dropna(subset=['r_zero', 'x_zero', 'y_zero', 'z_zero'], inplace=True)
    df = df[df["r_zero"].apply(lambda x: len(x) == winsize)]
    df.reset_index(drop=True, inplace=True)
    return df

def prepare_data_and_scale(df, input_cols, scaler=None):
    """Stacks features and scales the data."""
    feature_array = np.dstack([np.array(df[col].to_list()) for col in input_cols])
    original_shape = feature_array.shape
    n_features = original_shape[2]
    reshaped_for_scaling = feature_array.reshape(-1, n_features)
    
    if scaler is None:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(reshaped_for_scaling)
    else:
        scaled_data = scaler.transform(reshaped_for_scaling)
        
    X_scaled = scaled_data.reshape(original_shape)
    return X_scaled, scaler

# --- Trajectory Transformer Architecture ---

@register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    """Injects positional information into the input sequence."""
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        # Standard sinusoidal positional encoding
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.pos_encoding = angle_rads[np.newaxis, ...]
        self.pos_encoding = tf.cast(self.pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
        
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
        
    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({'position': self.position, 'd_model': self.d_model})
        return config

def create_transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    """Creates a single block of a Transformer Encoder."""
    # Attention and Normalization
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = Dense(ff_dim, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x + res

def create_trajectory_transformer(n_steps, n_features, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, dropout=0.1):
    """Builds the complete Trajectory Transformer model."""
    inputs = Input(shape=(n_steps, n_features))
    x = PositionalEncoding(position=n_steps, d_model=n_features)(inputs)
    
    for _ in range(num_transformer_blocks):
        x = create_transformer_encoder_block(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_last")(x)
    x = Dropout(0.4)(x)
    x = Dense(20, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Main Experiment Functions ---

def run_transformer_training(args):
    run_name = f"transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment("Trajectory_Transformer_Experiments")

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"=== Starting Transformer Training Run ID: {run_id} ===")
        mlflow.log_param("epochs", args.epochs)
        
        input_cols = ['r_zero']
        n_steps = 25
        n_features = len(input_cols)
        
        train_df = extract_dataset(args.train_file, winsize=n_steps)
        X_train, scaler = prepare_data_and_scale(train_df, input_cols)
        y_train = train_df['unsafe'].values
        
        transformer_model = create_trajectory_transformer(n_steps, n_features)
        transformer_model.summary()
        
        mlflow_callback = mlflow.keras.MLflowCallback(run)
        transformer_model.fit(X_train, y_train, epochs=args.epochs, batch_size=256, validation_split=0.2, class_weight={0: 1., 1: 10.}, callbacks=[mlflow_callback], verbose=1)
        
        # --- NEW: Robust model weights saving logic ---
        # 1. Define paths for the weights and scaler
        weights_path = "transformer_weights.weights.h5"
        scaler_path = "transformer_scaler.joblib"
        
        # 2. Save only the model's weights
        transformer_model.save_weights(weights_path)
        joblib.dump(scaler, scaler_path)
        
        # 3. Log the weights file and scaler as artifacts
        mlflow.log_artifact(weights_path)
        mlflow.log_artifact(scaler_path)
        
        # 4. Clean up local files
        os.remove(weights_path)
        os.remove(scaler_path)
        # --- End of new logic ---
        
        print(f"\n=== Finished Transformer Training. Run ID: {run_id} ===")


def run_transformer_evaluation(args):
    print(f"=== Evaluating Transformer from Run ID: {args.run_id} ===")
    
    with mlflow.start_run(run_id=args.run_id):
        mlflow.log_param("evaluation_file", args.eval_file)
        
        # --- NEW: Robust model loading logic ---
        # 1. Download the weights and scaler artifacts
        weights_path = mlflow.artifacts.download_artifacts(run_id=args.run_id, artifact_path="transformer_weights.weights.h5")
        scaler_path = mlflow.artifacts.download_artifacts(run_id=args.run_id, artifact_path="transformer_scaler.joblib")
        
        # 2. Re-create the model architecture
        input_cols = ['r_zero']
        n_steps = 25
        n_features = len(input_cols)
        model = create_trajectory_transformer(n_steps, n_features)
        
        # 3. Load the trained weights into the fresh model
        model.load_weights(weights_path)
        scaler = joblib.load(scaler_path)
        # --- End of new logic ---
        
        test_df = extract_dataset(args.eval_file)
        X_test, _ = prepare_data_and_scale(test_df, input_cols, scaler=scaler)
        y_test_labels = test_df['unsafe'].values
        
        probabilities = model.predict(X_test).flatten()
        
        best_f1 = 0
        best_threshold = 0
        for threshold in np.arange(0.05, 1.0, 0.05):
            preds = (probabilities > threshold).astype(int)
            f1 = f1_score(y_test_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                
        print(f"\n--- Optimal Threshold: {best_threshold:.2f} (Window F1-Score: {best_f1:.4f}) ---")
        final_preds = (probabilities > best_threshold).astype(int)
        
        report = classification_report(y_test_labels, final_preds, target_names=['Safe', 'Unsafe'], output_dict=True)
        mlflow.log_metric("eval_f1_unsafe", report['Unsafe']['f1-score'])
        
        print("\n--- Per-Window Performance ---")
        print(classification_report(y_test_labels, final_preds, target_names=['Safe', 'Unsafe']))

def main():
    parser = argparse.ArgumentParser(description="Trajectory Transformer Experiment Runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_train = subparsers.add_parser('train', help='Train the Trajectory Transformer model.')
    parser_train.add_argument('--train-file', default='datasets/train_dataset.csv')
    parser_train.add_argument('--epochs', type=int, default=50)
    parser_train.set_defaults(func=run_transformer_training)

    parser_eval = subparsers.add_parser('evaluate', help='Evaluate a trained Transformer model.')
    parser_eval.add_argument('run_id', help='The MLflow Run ID of the trained model to evaluate.')
    parser_eval.add_argument('--eval-file', default='datasets/test1_dataset.csv')
    parser_eval.set_defaults(func=run_transformer_evaluation)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()