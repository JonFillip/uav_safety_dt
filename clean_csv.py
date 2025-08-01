import csv

def clean_csv_file(input_path, output_path, expected_columns):
    """
    Reads a malformed CSV, truncates rows with too many columns,
    and writes the cleaned data to a new file.
    """
    print(f"--- Cleaning {input_path} ---")
    with open(input_path, 'r', newline='') as infile, \
        open(output_path, 'w', newline='') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Read the header and write it to the new file
        header = next(reader)
        writer.writerow(header)
        
        # Process the rest of the rows
        for i, row in enumerate(reader):
            # If a row has more columns than expected, truncate it
            if len(row) > expected_columns:
                print(f"Fixing row {i+2}: Found {len(row)} columns, expected {expected_columns}. Truncating.")
                row = row[:expected_columns]
            
            writer.writerow(row)
            
    print(f"\n--- Cleaned data saved to {output_path} ---")

if __name__ == '__main__':
    # Based on the error, the file should have 14 columns
    clean_csv_file('datasets/train_dataset.csv', 'datasets/train_dataset_cleaned.csv', 14)