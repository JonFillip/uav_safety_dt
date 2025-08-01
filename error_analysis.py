import pandas as pd
from model import CNNModel
from superialist.data_analysis import stats

print("--- Starting Error Analysis: Safety vs. Uncertainty ---")

# --- Analysis for test1_dataset.csv ---
print("\n\n=== Analysis for test1_dataset.csv ===")
model_wrapper_1 = CNNModel()
test1_data = model_wrapper_1.extract_dataset('datasets/test1_dataset.csv')

# Use the stats function to compare the 'unsafe' and 'uncertain' labels
# We are treating 'uncertain' as the "prediction" and 'unsafe' as the "ground truth"
stats(
    data=test1_data,
    flag="uncertain",  # The column to treat as the prediction
    label="unsafe",    # The column to treat as the ground truth
    levels=["log"]     # We want the analysis on a per-flight-log basis
)


# --- Analysis for test2_dataset.csv ---
print("\n\n=== Analysis for test2_dataset.csv ===")
model_wrapper_2 = CNNModel()
test2_data = model_wrapper_2.extract_dataset('datasets/test2_dataset.csv')

stats(
    data=test2_data,
    flag="uncertain",
    label="unsafe",
    levels=["log"]
)

print("\n--- Analysis Complete ---")