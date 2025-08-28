#!/bin/bash

# -----------------------------------------------------------------------------
# UAV Safety Experiment Runner
# -----------------------------------------------------------------------------
# This script automates the process of running the two-stage DT experiment
# multiple times to account for model initialization randomness.
# -----------------------------------------------------------------------------

# --- Configuration ---
# Set the parameters for the batch of experiments you want to run.

# Total number of times to run the experiment
NUM_RUNS=30

# A descriptive base name for this batch of runs.
# The script will append a number to this for each run (e.g., causal_lstm_run_1)
BASE_RUN_ID="active_casual_run"

# The feature mode for the detector. Options: 'causal', 'proactive', 'active'
MODE="active"

# The type of predictor model. Options: 'lstm', 'tft'
PREDICTOR="tft"

# Number of training epochs for both predictor and detector
EPOCHS=50

# The final, unseen dataset to use for evaluation
EVAL_FILE="datasets/test1_dataset.csv"

# The number of consecutive unsafe windows required to flag a flight as unsafe
CONSECUTIVE_WINDOWS=4


# --- Main Loop ---
echo "Starting $NUM_RUNS experiment runs with base ID '$BASE_RUN_ID'..."

for i in $(seq 1 $NUM_RUNS)
do
    # Create a unique run ID for this specific iteration
    RUN_ID="${BASE_RUN_ID}_${i}"

    echo "----------------------------------------------------"
    echo "--- RUN $i/$NUM_RUNS: Starting (ID: $RUN_ID) ---"
    echo "----------------------------------------------------"

    # Step 1: Train the model
    echo "--- Training model for run $RUN_ID ---"
    python predictive_twin/advanced_dt_experiment.py train \
        --run-id "$RUN_ID" \
        --mode "$MODE" \
        --predictor-type "$PREDICTOR" \
        --epochs "$EPOCHS"

    # Check if training was successful before proceeding
    if [ $? -ne 0 ]; then
        echo "!!! Training failed for run $RUN_ID. Skipping evaluation. !!!"
        continue
    fi

    # Step 2: Evaluate the model
    echo "--- Evaluating model for run $RUN_ID ---"
    python predictive_twin/advanced_dt_experiment.py evaluate "$RUN_ID" \
        --mode "$MODE" \
        --eval-file "$EVAL_FILE" \
        --consecutive-windows "$CONSECUTIVE_WINDOWS"

    if [ $? -ne 0 ]; then
        echo "!!! Evaluation failed for run $RUN_ID. !!!"
    fi

    echo "--- RUN $i/$NUM_RUNS: Finished ---"
done

echo "----------------------------------------------------"
echo "All $NUM_RUNS experiment runs completed."
echo "----------------------------------------------------"
