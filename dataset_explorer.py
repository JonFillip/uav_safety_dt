import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

def explore_dataset(filepath):
    """Explore a single dataset and return insights"""
    print(f"\n===== Exploring {os.path.basename(filepath)} =====")
    
    try:
        # Load the dataset
        df = pd.read_csv(filepath, skipinitialspace=True)
        
        # Basic information
        print(f"Shape: {df.shape}")
        print("\nColumns and data types:")
        for col, dtype in df.dtypes.items():
            print(f"  - {col}: {dtype}")
        
        # Sample data
        print("\nSample data (first 5 rows):")
        print(df.head())
        
        # Basic statistics for numeric columns
        print("\nNumeric column statistics:")
        print(df.describe())
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print("\nMissing values:")
            print(missing[missing > 0])
        else:
            print("\nNo missing values found.")
        
        # Create correlation heatmap for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 10))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title(f'Correlation Matrix - {os.path.basename(filepath)}')
            plt.tight_layout()
            
            output_file = f"correlation_{os.path.basename(filepath).split('.')[0]}.png"
            plt.savefig(output_file)
            print(f"\nCorrelation matrix saved to {output_file}")
        
        # Check for list-like columns (comma-separated values)
        list_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if sample and isinstance(sample, str) and ',' in sample:
                    list_cols.append(col)
        
        if list_cols:
            print("\nDetected list-like columns (comma-separated values):")
            for col in list_cols:
                print(f"  - {col}")
        
        return df
    
    except Exception as e:
        print(f"Error exploring {filepath}: {e}")
        return None

def explore_all_datasets():
    """Explore all CSV datasets in the datasets directory"""
    # List available datasets
    dataset_files = [f for f in os.listdir('datasets') if f.endswith('.csv') and not os.path.isdir(os.path.join('datasets', f))]
    print(f"Found {len(dataset_files)} datasets: {dataset_files}")
    
    results = {}
    for dataset in dataset_files:
        results[dataset] = explore_dataset(os.path.join('datasets', dataset))
    
    return results

if __name__ == "__main__":
    print("UAV Dataset Explorer")
    print("===================")
    explore_all_datasets()
