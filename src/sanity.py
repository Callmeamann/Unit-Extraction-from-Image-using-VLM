import pandas as pd
import argparse
import os
from src.constants import entity_unit_map
from src.utils import extract_value_and_unit

# Check if the file exists and is a valid CSV file
def check_file(filename):
    if not filename.lower().endswith('.csv'):
        raise ValueError("Only CSV files are allowed.")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Filepath: {filename} is invalid or not found.")

# Sanity check function to compare test and output CSVs
def sanity_check(test_filename, output_filename):
    check_file(test_filename)
    check_file(output_filename)
    
    try:
        test_df = pd.read_csv(test_filename)
        output_df = pd.read_csv(output_filename)
    except Exception as e:
        raise ValueError(f"Error reading the CSV files: {e}")
    
    # Ensure the 'index' column is in both files and the 'prediction' column is in the output
    if 'index' not in test_df.columns:
        raise ValueError("Test CSV file must contain the 'index' column.")
    
    if 'index' not in output_df.columns or 'prediction' not in output_df.columns:
        raise ValueError("Output CSV file must contain 'index' and 'prediction' columns.")
    
    # Check for missing indices in the output compared to the test data
    missing_index = set(test_df['index']).difference(set(output_df['index']))
    if len(missing_index) != 0:
        print(f"Missing indices in output file: {missing_index}")
        
    # Check for extra indices in the output that aren't in the test data
    extra_index = set(output_df['index']).difference(set(test_df['index']))
    if len(extra_index) != 0:
        print(f"Extra indices in output file: {extra_index}")
    
    # Apply string parsing function from the utils module on the prediction column
    output_df.apply(lambda x: extract_value_and_unit(x['prediction'], x['index']), axis=1)
    
    print(f"Parsing successful for file: {output_filename}")

# Main function to parse arguments and run the sanity check
if __name__ == "__main__":
    # Usage example: python sanity.py --test_filename dataset/test.csv --output_filename test_out.csv
    
    parser = argparse.ArgumentParser(description="Run sanity check on a CSV file.")
    parser.add_argument("--test_filename", type=str, required=True, help="The test CSV file name.")
    parser.add_argument("--output_filename", type=str, required=True, help="The output CSV file name to check.")
    args = parser.parse_args()

    try:
        sanity_check(args.test_filename, args.output_filename)
    except Exception as e:
        print('Error:', e)
