#!/usr/bin/env python3
"""
UAV Test Dataset Comparator

This script compares test datasets to identify differences and potential safety issues.
It works without relying on pandas or other complex libraries that might have version compatibility issues.
"""

import os
import csv
import sys
import argparse
from collections import defaultdict

def load_dataset_stats(filepath):
    """Load a dataset and compute basic statistics"""
    print(f"Loading {os.path.basename(filepath)}...")
    
    stats = {
        'name': os.path.basename(filepath),
        'row_count': 0,
        'columns': [],
        'numeric_stats': {},
        'categorical_counts': {},
        'safety_metrics': {}
    }
    
    try:
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            stats['columns'] = header
            
            # Initialize data structures
            for col in header:
                stats['numeric_stats'][col] = {
                    'is_numeric': True,
                    'min': None,
                    'max': None,
                    'sum': 0,
                    'count': 0
                }
                stats['categorical_counts'][col] = defaultdict(int)
            
            # Process each row
            for row in reader:
                stats['row_count'] += 1
                
                # Process each column
                for i, value in enumerate(row):
                    if i < len(header):  # Ensure we don't go out of bounds
                        col_name = header[i]
                        
                        # Try to convert to float for numeric analysis
                        try:
                            float_val = float(value)
                            col_stats = stats['numeric_stats'][col_name]
                            
                            if col_stats['min'] is None or float_val < col_stats['min']:
                                col_stats['min'] = float_val
                            
                            if col_stats['max'] is None or float_val > col_stats['max']:
                                col_stats['max'] = float_val
                            
                            col_stats['sum'] += float_val
                            col_stats['count'] += 1
                        except (ValueError, TypeError):
                            # Mark as non-numeric
                            stats['numeric_stats'][col_name]['is_numeric'] = False
                            # Count categorical values
                            stats['categorical_counts'][col_name][value] += 1
            
            # Calculate averages for numeric columns
            for col, col_stats in stats['numeric_stats'].items():
                if col_stats['is_numeric'] and col_stats['count'] > 0:
                    col_stats['avg'] = col_stats['sum'] / col_stats['count']
                else:
                    # Remove numeric stats for non-numeric columns
                    stats['numeric_stats'][col] = {'is_numeric': False}
            
            # Extract safety metrics
            safety_columns = [col for col in header if 'obstacle' in col.lower() or 'distance' in col.lower() or 'risky' in col.lower() or 'unsafe' in col.lower()]
            
            for col in safety_columns:
                if stats['numeric_stats'][col].get('is_numeric', False):
                    stats['safety_metrics'][col] = {
                        'min': stats['numeric_stats'][col]['min'],
                        'max': stats['numeric_stats'][col]['max'],
                        'avg': stats['numeric_stats'][col].get('avg', 0)
                    }
                elif col in stats['categorical_counts']:
                    # For boolean/categorical safety columns
                    stats['safety_metrics'][col] = dict(stats['categorical_counts'][col])
            
            return stats
            
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def compare_datasets(dataset1_stats, dataset2_stats):
    """Compare two datasets and print the differences"""
    print("\n===== Dataset Comparison =====")
    print(f"Comparing {dataset1_stats['name']} and {dataset2_stats['name']}")
    
    # Compare basic metrics
    print("\nBasic Metrics:")
    print(f"  Rows: {dataset1_stats['row_count']} vs {dataset2_stats['row_count']}")
    
    # Compare columns
    common_columns = set(dataset1_stats['columns']).intersection(set(dataset2_stats['columns']))
    only_in_1 = set(dataset1_stats['columns']).difference(set(dataset2_stats['columns']))
    only_in_2 = set(dataset2_stats['columns']).difference(set(dataset1_stats['columns']))
    
    print(f"\nColumns:")
    print(f"  Common: {len(common_columns)}")
    if only_in_1:
        print(f"  Only in {dataset1_stats['name']}: {', '.join(only_in_1)}")
    if only_in_2:
        print(f"  Only in {dataset2_stats['name']}: {', '.join(only_in_2)}")
    
    # Compare safety metrics
    print("\nSafety Metrics Comparison:")
    all_safety_cols = set(dataset1_stats['safety_metrics'].keys()).union(set(dataset2_stats['safety_metrics'].keys()))
    
    for col in sorted(all_safety_cols):
        print(f"\n  {col}:")
        
        if col in dataset1_stats['safety_metrics'] and col in dataset2_stats['safety_metrics']:
            # Both datasets have this safety metric
            metric1 = dataset1_stats['safety_metrics'][col]
            metric2 = dataset2_stats['safety_metrics'][col]
            
            if isinstance(metric1, dict) and 'min' in metric1:
                # Numeric comparison
                print(f"    {dataset1_stats['name']}: min={metric1['min']:.4f}, max={metric1['max']:.4f}, avg={metric1['avg']:.4f}")
                print(f"    {dataset2_stats['name']}: min={metric2['min']:.4f}, max={metric2['max']:.4f}, avg={metric2['avg']:.4f}")
                
                # Calculate differences
                min_diff = ((metric2['min'] - metric1['min']) / max(abs(metric1['min']), 1)) * 100 if metric1['min'] != 0 else float('inf')
                max_diff = ((metric2['max'] - metric1['max']) / max(abs(metric1['max']), 1)) * 100 if metric1['max'] != 0 else float('inf')
                avg_diff = ((metric2['avg'] - metric1['avg']) / max(abs(metric1['avg']), 1)) * 100 if metric1['avg'] != 0 else float('inf')
                
                print(f"    Difference: min={min_diff:.2f}%, max={max_diff:.2f}%, avg={avg_diff:.2f}%")
            else:
                # Categorical comparison
                for value in sorted(set(metric1.keys()).union(set(metric2.keys()))):
                    count1 = metric1.get(value, 0)
                    count2 = metric2.get(value, 0)
                    pct1 = (count1 / dataset1_stats['row_count']) * 100 if dataset1_stats['row_count'] > 0 else 0
                    pct2 = (count2 / dataset2_stats['row_count']) * 100 if dataset2_stats['row_count'] > 0 else 0
                    
                    print(f"    {value}: {count1} ({pct1:.2f}%) vs {count2} ({pct2:.2f}%)")
        else:
            # Only one dataset has this safety metric
            if col in dataset1_stats['safety_metrics']:
                print(f"    Only in {dataset1_stats['name']}")
            else:
                print(f"    Only in {dataset2_stats['name']}")

def main():
    parser = argparse.ArgumentParser(
        description="UAV Test Dataset Comparator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare test1_dataset.csv and test2_dataset.csv
  python compare_test_datasets.py --datasets test1_dataset.csv test2_dataset.csv
  
  # Compare test1_labels.csv and test2_labels.csv
  python compare_test_datasets.py --datasets test1_labels.csv test2_labels.csv
"""
    )
    
    parser.add_argument(
        "--datasets", 
        nargs=2,
        required=True,
        help="Specify two datasets to compare"
    )
    
    args = parser.parse_args()
    
    # Validate datasets
    dataset_files = []
    for dataset in args.datasets:
        dataset_path = os.path.join('datasets', dataset)
        if os.path.exists(dataset_path):
            dataset_files.append(dataset_path)
        else:
            print(f"Error: Dataset '{dataset}' not found in the datasets directory.")
            return
    
    if len(dataset_files) != 2:
        print("Error: Please specify exactly two valid datasets to compare.")
        return
    
    # Load and compare datasets
    dataset1_stats = load_dataset_stats(dataset_files[0])
    dataset2_stats = load_dataset_stats(dataset_files[1])
    
    if dataset1_stats and dataset2_stats:
        compare_datasets(dataset1_stats, dataset2_stats)
    
    print("\nComparison complete.")

if __name__ == "__main__":
    main()
