import os
import argparse
import pandas as pd
from sklearn.metrics import f1_score

from predictive_twin.online_learning_dt import DTModel

def print_f1_score(title, y_true, y_pred):
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"--- {title}: F1-Score = {f1:.3f} ---")
    return f1

def run_experiment(args):
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(args.base_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.adapted_model_path), exist_ok=True)
    
    # --- PHASE 1: TRAIN BASE MODEL ---
    print("\nPHASE 1: Training the Base Model...")
    base_model = DTModel()
    train_data = base_model.extract_and_process_data(args.base_train_file)
    nominal_data = train_data[train_data["win_dist_0_10"] > 3.5]
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Nominal data shape: {nominal_data.shape}")
    
    base_model.build_model(n_input_features=1)
    base_model.train(nominal_data, inputs=['r_zero'], epochs=args.epochs)
    base_model.save(args.base_model_path)
    print("Base Model training complete.")

    # --- PHASE 2: ADAPT MODEL (FINE-TUNING) ---
    print("\nPHASE 2: Adapting the Model with New Data...")
    adapted_model = DTModel()
    adapted_model.load(args.base_model_path) # Start with the trained base model
    
    adaptation_data = adapted_model.extract_and_process_data(args.adaptation_file)
    nominal_adaptation_data = adaptation_data[adaptation_data["win_dist_0_10"] > 3.5]

    print(f"Adaptation data shape: {adaptation_data.shape}")
    print(f"Nominal adaptation data shape: {nominal_adaptation_data.shape}")

    # Continue training (fine-tune) on the new data
    adapted_model.train(nominal_adaptation_data, inputs=['r_zero'], epochs=args.epochs)
    adapted_model.save(args.adapted_model_path)
    print("Model adaptation complete.")

    # --- PHASE 3: EVALUATE BOTH MODELS ---
    print("\nPHASE 3: Evaluating Models on Unseen Test Data...")
    eval_data = DTModel().extract_and_process_data(args.eval_file)
    
    print(f"Evaluation data shape: {eval_data.shape}")
    
    # Check if 'unsafe' column exists
    if 'unsafe' not in eval_data.columns:
        print("WARNING: 'unsafe' column not found in evaluation data. Creating dummy labels.")
        eval_data['unsafe'] = False  # Create dummy labels for testing
    
    # Evaluate Base Model
    print("\nEvaluating BASE model...")
    base_pred_df = base_model.get_anomaly_scores(eval_data.copy(), inputs=['r_zero'])
    base_pred_df['predicted_positive'] = base_pred_df['mean_loss_4'] < args.threshold
    
    if 'log_folder' not in base_pred_df.columns or 'log_name' not in base_pred_df.columns:
        print("WARNING: 'log_folder' or 'log_name' columns not found. Using index-based grouping.")
        base_pred_df['log_folder'] = 'default'
        base_pred_df['log_name'] = base_pred_df.index
    
    log_labels = base_pred_df.groupby(['log_folder', 'log_name'])['unsafe'].any()
    log_predictions_base = base_pred_df.groupby(['log_folder', 'log_name'])['predicted_positive'].any()
    f1_base = print_f1_score("Base Model Performance", log_labels, log_predictions_base)

    # Evaluate Adapted Model
    print("\nEvaluating ADAPTED model...")
    adapted_pred_df = adapted_model.get_anomaly_scores(eval_data.copy(), inputs=['r_zero'])
    adapted_pred_df['predicted_positive'] = adapted_pred_df['mean_loss_4'] < args.threshold
    
    # Ensure same grouping columns
    if 'log_folder' not in adapted_pred_df.columns or 'log_name' not in adapted_pred_df.columns:
        adapted_pred_df['log_folder'] = 'default'
        adapted_pred_df['log_name'] = adapted_pred_df.index
    
    log_predictions_adapted = adapted_pred_df.groupby(['log_folder', 'log_name'])['predicted_positive'].any()
    f1_adapted = print_f1_score("Adapted Model Performance", log_labels, log_predictions_adapted)

    # --- FINAL CONCLUSION ---
    print("\n--- EXPERIMENT CONCLUSION ---")
    improvement = ((f1_adapted - f1_base) / f1_base) * 100 if f1_base > 0 else float('inf')
    print(f"Base Model F1-Score: {f1_base:.3f}")
    print(f"Adapted Model (Digital Twin) F1-Score: {f1_adapted:.3f}")
    if f1_adapted > f1_base:
        print(f"The adapted model showed an improvement of {improvement:.2f}%")
    elif f1_adapted == f1_base:
        print("The adapted model performed equally to the base model.")
    else:
        decline = ((f1_base - f1_adapted) / f1_base) * 100
        print(f"The adapted model showed a decline of {decline:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Digital Twin Online Learning Experiment")
    parser.add_argument('--base-train-file', default='datasets/train_dataset.csv')
    parser.add_argument('--adaptation-file', default='datasets/test2_dataset.csv')
    parser.add_argument('--eval-file', default='datasets/test1_dataset.csv')
    parser.add_argument('--base-model-path', default='models/dt_base_model.keras')
    parser.add_argument('--adapted-model-path', default='models/dt_adapted_model.keras')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--threshold', type=float, default=-0.15, help="Optimal threshold found from previous validation.")
    
    args = parser.parse_args()
    
    # Check if input files exist
    for file_path in [args.base_train_file, args.adaptation_file, args.eval_file]:
        if not os.path.exists(file_path):
            print(f"ERROR: Input file not found: {file_path}")
            return
    
    run_experiment(args)

if __name__ == "__main__":
    main()