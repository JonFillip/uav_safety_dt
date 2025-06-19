#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import your model classes
from model import CNNModel
from temporal_fusion_transformer import TemporalFusionTransformer, prepare_tft_data

# --- Helper Functions ---
def print_stats(title, y_true, y_pred):
    """Prints a confusion matrix and key classification metrics."""
    print(f"\n--- {title} ---")
    # Ensure y_pred is boolean for confusion matrix
    y_pred_bool = np.array(y_pred) > 0.5 if y_pred.dtype != bool else y_pred
    cm = confusion_matrix(y_true, y_pred_bool)
    
    # Handle cases where cm might not have all 4 values (e.g., perfect prediction)
    if cm.size == 1:
        tn, fp, fn, tp = (cm[0][0], 0, 0, 0) if y_true.iloc[0] == 0 else (0, 0, 0, cm[0][0])
    else:
        tn, fp, fn, tp = cm.ravel()
    
    print(f"Confusion Matrix:\n{cm}")
    print(f"  - True Negatives (TN): {tn}")
    print(f"  - False Positives (FP): {fp}")
    print(f"  - False Negatives (FN): {fn}")
    print(f"  - True Positives (TP): {tp}")
    
    precision = precision_score(y_true, y_pred_bool, zero_division=0)
    recall = recall_score(y_true, y_pred_bool, zero_division=0)
    f1 = f1_score(y_true, y_pred_bool, zero_division=0)
    
    print(f"\n  - Precision: {precision:.3f}")
    print(f"  - Recall: {recall:.3f}")
    print(f"  - F1-Score: {f1:.3f}")
    return {"precision": precision, "recall": recall, "f1": f1}


# --- Command Functions ---

def train(args):
    print(f"=== Training Model: {args.model_type} ===")
    os.makedirs('models', exist_ok=True)
    
    model_wrapper = CNNModel()
    
    if args.model_type == 'autoencoder':
        train_df = model_wrapper.extract_dataset('datasets/train_dataset.csv')
        nominal_df = train_df[train_df["win_dist_0_10"] > 3.5]
        print(f"Training autoencoder on {len(nominal_df)} nominal data windows...")
        ae_model = model_wrapper.get_autoencoder_model(n_input_features=1)
        model_wrapper.fit(
            nominal_df,
            model=model_wrapper.get_best_autoencoder_model(1), # Use the new function here
            inputs=["r_zero"],
            outputs=["r_zero"],
            output_is_list=True,
            epochs=args.epochs
        )
        model_wrapper.save(args.output_path)
        print(f"Autoencoder model saved to {args.output_path}")

    elif args.model_type == 'classifier':

        training_file = 'datasets/test2_dataset.csv'
        print(f"Loading '{training_file}' for classifier training...")
        train_df = model_wrapper.extract_dataset(training_file)

        # 2. Balance the entire training dataset.
        print(f"Balancing training data for target '{args.target_label}'...")
        balanced_train_df = model_wrapper.balance(train_df, label=args.target_label)
        print(f"Original training size: {len(train_df)}, Balanced training size: {len(balanced_train_df)}")

        print(f"Training best classifier on {len(balanced_train_df)} labeled data windows...")
        num_features = 4
        input_cols = ['r_zero', 'x_zero', 'y_zero', 'z_zero']
        classifier_model = model_wrapper.get_best_classifier_model(num_features=num_features)
        
        # 3. Fit on the balanced dataframe.
        model_wrapper.fit(
            balanced_train_df,
            model=classifier_model, 
            inputs=input_cols,
            outputs=[args.target_label],
            epochs=args.epochs
        )
        model_wrapper.save(args.output_path)
        print(f"Best classifier model for target '{args.target_label}' saved to {args.output_path}")

    elif args.model_type == 'tft':
        print("Preparing data for TFT model...")
        train_df = model_wrapper.extract_dataset('datasets/train_dataset.csv')
        
        all_windows = np.array(train_df['r_zero'].to_list(), dtype=np.float32)
        past_history_len = 20
        future_target_len = 5
        
        # Ensure the model's window_size matches the input length
        tft_window_size = past_history_len 

        X = all_windows[:, :past_history_len]
        y = all_windows[:, past_history_len:] # This takes the last 5 steps

        X = np.expand_dims(X, axis=-1) # Shape -> (samples, 20, 1)
        y = np.expand_dims(y, axis=-1) # Shape -> (samples, 5, 1)
        
        print(f"Training TFT on {len(X)} sequences...")
        
        tft = TemporalFusionTransformer(window_size=tft_window_size, prediction_horizon=future_target_len)
        tft.build_model(num_features=1, hidden_layer_size=32, num_attention_heads=2)
        
        tft.fit(X, y, epochs=args.epochs)
        tft.save(args.output_path)
        print(f"TFT model weights saved to {args.output_path}")


def evaluate(args):
    print(f"=== Evaluating Model: {args.model_path} ===")
    print(f"--- Using evaluation file: {args.eval_file} ---")
    
    model_wrapper = CNNModel()
    test_df = model_wrapper.extract_dataset(args.eval_file)
    
    # --- Logic for Autoencoder ---
    if 'autoencoder' in args.model_path:
        model_wrapper.load(args.model_path)
        pred_df = model_wrapper.predict_encoder(test_df, inputs=["r_zero"])
        anomaly_score = pred_df['mean_loss_4']
        
        # --- FINAL FIX: Prioritize the command-line threshold ---
        if args.threshold is not None:
            threshold = args.threshold
            print(f"Using FIXED threshold from command line: {threshold}")
        else:
            # This is the demonstration mode if no threshold is passed
            threshold = np.percentile(anomaly_score.dropna(), 10) # Using 10th percentile for negative scores
            print(f"Warning: No threshold provided. Using DEMONSTRATION threshold (10th percentile): {threshold:.4f}")
        
        # --- FINAL FIX: Use '<' because more negative = bigger anomaly ---
        predicted_labels = anomaly_score < threshold
        
    # --- Logic for Classifier ---
    elif 'classifier' in args.model_path:
        model_wrapper.load(args.model_path)
        output_col = [f"{args.target_label}_pred"]
        input_cols = ['r_zero', 'x_zero', 'y_zero', 'z_zero']
        pred_df = model_wrapper.predict(test_df, inputs=input_cols, outputs=output_col)
        predicted_labels = pred_df[output_col[0]] > 0.5
        
    # --- Logic for TFT ---
    elif 'tft' in args.model_path:
        # --- START OF FIXES ---

        # FIX 1: Prepare evaluation data exactly like the training data
        print("Preparing data for TFT model evaluation...")
        test_df = test_df.reset_index(drop=True)
        all_windows = np.array(test_df['r_zero'].to_list(), dtype=np.float32)
        past_history_len = 20
        X_test = np.expand_dims(all_windows[:, :past_history_len], axis=-1)

        # FIX 2: Set window_size and prediction_horizon to match the trained model
        tft_window_size = 20
        prediction_horizon = 5
        tft = TemporalFusionTransformer(window_size=tft_window_size, prediction_horizon=prediction_horizon)
        
        # FIX 3: Pass hyperparameters to tft.load() to build the correct architecture
        print("Loading TFT model with correct architecture...")
        tft.load(
            args.model_path, 
            num_features=1,
            hidden_layer_size=32,  # This MUST match the value used in training
            num_attention_heads=2  # This MUST match the value used in training
        )

        # Now, proceed with prediction and evaluation
        anomaly_score = tft.predict_anomaly_score(X_test)
        
        aligned_scores = pd.Series([np.nan] * len(test_df), index=test_df.index)
        aligned_scores.iloc[:len(anomaly_score)] = anomaly_score
        anomaly_score = aligned_scores.ffill().bfill()
        
        if args.threshold is not None:
            threshold = args.threshold
            print(f"Using FIXED TFT threshold: {threshold}")
        else:
            default_threshold = np.percentile(anomaly_score.dropna(), 90)
            print(f"Warning: No threshold provided. Using DEMONSTRATION threshold (90th percentile): {default_threshold:.4f}")
            threshold = default_threshold
            
        # TFT anomaly scores are positive (uncertainty widths), so we use >
        predicted_labels = anomaly_score > threshold
    else:
        print("Unknown model type.")
        return
        
    # --- Final Statistical Analysis (Same for all models) ---
    test_df['predicted_positive'] = predicted_labels
    log_labels = test_df.groupby(['log_folder', 'log_name'])[[args.target_label]].any()
    log_predictions = test_df.groupby(['log_folder', 'log_name'])['predicted_positive'].any()
    comparison_df = log_labels.join(log_predictions)
    
    print_stats(
        f"Prediction for '{args.target_label}' (per-log)",
        comparison_df[args.target_label],
        comparison_df['predicted_positive']
    )

    
def tune(args):
    """New function to run hyperparameter tuning."""
    print(f"=== Tuning Hyperparameters for Model: {args.model_type} ===")
    model_wrapper = CNNModel()
    
    if args.model_type == 'classifier':
        labeled_df = model_wrapper.extract_dataset('datasets/test2_dataset.csv')
        train_df, _ = train_test_split(labeled_df, test_size=0.2, random_state=42, stratify=labeled_df[args.target_label])
        
        # Balance the training data for the classifier
        balanced_train_df = model_wrapper.balance(train_df, label=args.target_label)

        tuner = model_wrapper.search_hyperparameters(
            train_df=balanced_train_df,
            objective="val_accuracy",
            inputs=['r_zero', 'x_zero', 'y_zero', 'z_zero'],
            outputs=[args.target_label],
            trials=args.trials,
            epochs=20,
            project_name=f"tune_classifier_{args.target_label}"
        )

    elif args.model_type == 'autoencoder':
        train_data = model_wrapper.extract_dataset("datasets/train_dataset.csv")
        nominal_data = train_data[train_data["win_dist_0_10"] > 3.5].copy()

        tuner = model_wrapper.search_hyperparameters(
            train_df=nominal_data,
            objective="val_loss",
            inputs=['r_zero'],
            trials=args.trials,
            epochs=50,
            project_name="tune_autoencoder"
        )
    
    print("\n----------------------------------------------------")
    print("---           BEST HYPERPARAMETERS           ---")
    print("----------------------------------------------------")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    for param, value in best_hps.values.items():
        print(f"{param:<20}: {value}")
    print("----------------------------------------------------")


# --- Main Argument Parser ---
def main():
    parser = argparse.ArgumentParser(description="UAV Safety Analysis Runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # --- Train Command ---
    parser_train = subparsers.add_parser('train', help='Train a model')
    parser_train.add_argument('model_type', choices=['autoencoder', 'classifier', 'tft'], help='Type of model to train')
    parser_train.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser_train.add_argument('--output-path', '-o', required=True, help='Path to save the trained model')
    parser_train.add_argument('--target-label', choices=['unsafe', 'uncertain'], default='uncertain', help="Target label for supervised models ('classifier')")
    parser_train.set_defaults(func=train)

    # --- Evaluate Command ---
    parser_eval = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    parser_eval.add_argument('--eval-file', default='datasets/test2_dataset.csv', help="Path to the dataset for evaluation")
    parser_eval.add_argument('model_path', help='Path to the saved model file')
    parser_eval.add_argument('--target-label', choices=['unsafe', 'uncertain'], default='uncertain', help="Ground truth label to evaluate against")
    parser_eval.add_argument('--threshold', type=float, help="[Optional] Anomaly threshold for unsupervised models")
    parser_eval.set_defaults(func=evaluate)
    
    # --- Tune Command (New!) ---
    parser_tune = subparsers.add_parser('tune', help='Tune hyperparameters for a model')
    parser_tune.add_argument('model_type', choices=['classifier', 'autoencoder'], help='Type of model to tune') # Can be extended
    parser_tune.add_argument('--trials', type=int, default=10, help='Number of tuning trials to run')
    parser_tune.add_argument('--target-label', choices=['unsafe', 'uncertain'], default='uncertain', help="Target label for supervised models")
    parser_tune.set_defaults(func=tune)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()