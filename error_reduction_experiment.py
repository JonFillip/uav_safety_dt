import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from model import CNNModel

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

def train_multi_feature_autoencoder(args):
    """Trains an autoencoder on multiple input features."""
    print("--- Training Multi-Feature Autoencoder ---")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    model_wrapper = CNNModel()
    train_df = model_wrapper.extract_dataset('datasets/train_dataset.csv')
    nominal_df = train_df[train_df["win_dist_0_10"] > 3.5]
    
    # Define the four features we will use for training
    input_features = ['r_zero', 'x_zero', 'y_zero', 'z_zero']
    
    print(f"Training on {len(nominal_df)} windows with {len(input_features)} features...")

    # Build the model to accept 4 input channels
    # NOTE: This assumes your `get_best_autoencoder_model` is updated to handle `n_input_features`
    multi_feature_model = model_wrapper.get_best_autoencoder_model(n_input_features=len(input_features))
    
    model_wrapper.fit(
        nominal_df,
        model=multi_feature_model,
        inputs=input_features,
        outputs=input_features, # The model tries to reconstruct all inputs
        output_is_list=True,
        epochs=args.epochs
    )
    model_wrapper.save(args.output_path)
    print(f"\nMulti-feature autoencoder saved to {args.output_path}")

def evaluate_with_different_scores(args):
    """Evaluates the model using a user-specified anomaly score metric."""
    print("--- Evaluating with Different Anomaly Scores ---")
    
    model_wrapper = CNNModel()
    model_wrapper.load(args.model_path)
    
    test_df = model_wrapper.extract_dataset(args.eval_file)
    
    # Generate predictions using the same features the model was trained on
    input_features = ['r_zero', 'x_zero', 'y_zero', 'z_zero']
    pred_df = model_wrapper.predict_encoder(test_df, inputs=input_features)

    # Use the anomaly score specified by the user
    anomaly_score_col = args.score_metric
    print(f"\nUsing '{anomaly_score_col}' as the anomaly score.")
    pred_df['predicted_positive'] = pred_df[anomaly_score_col] < args.threshold

    # Aggregate results per log
    log_labels = pred_df.groupby(['log_folder', 'log_name'])[args.target_label].any()
    log_predictions = pred_df.groupby(['log_folder', 'log_name'])['predicted_positive'].any()
    
    print_stats(
        f"Prediction for '{args.target_label}' on {os.path.basename(args.eval_file)}",
        log_labels,
        log_predictions
    )

def main():
    parser = argparse.ArgumentParser(description="UAV Error Reduction Experiment Runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # --- Train Command for Multi-Feature Model ---
    parser_train = subparsers.add_parser('train', help='Train the multi-feature autoencoder')
    parser_train.add_argument('-o', '--output-path', required=True, help='Path to save the trained model')
    parser_train.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser_train.set_defaults(func=train_multi_feature_autoencoder)

    # --- Evaluate Command with Score Selection ---
    parser_eval = subparsers.add_parser('evaluate', help='Evaluate the model with different score metrics')
    parser_eval.add_argument('--model-path', required=True, help='Path to the saved multi-feature autoencoder')
    parser_eval.add_argument('--eval-file', default='datasets/test1_dataset.csv', help="Path to the dataset for evaluation")
    parser_eval.add_argument('--score-metric', choices=['mean_loss', 'mean_loss_4'], default='mean_loss_4', help='Which anomaly score to use for evaluation')
    parser_eval.add_argument('--threshold', type=float, required=True, help="Anomaly threshold (e.g., -0.15)")
    parser_eval.add_argument('--target-label', choices=['unsafe', 'uncertain'], default='unsafe', help="Ground truth label")
    parser_eval.set_defaults(func=evaluate_with_different_scores)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()