import pandas as pd
import os
import sys

try:
    # Define the working directory
    working_dir = r'c:\Users\bhoom\Downloads\ml1'
    os.chdir(working_dir)
    print(f"Working directory: {working_dir}\n", flush=True)
    sys.stdout.flush()

    # Find all CSV files in the directory
    csv_files = sorted([f for f in os.listdir('.') if f.endswith('.csv')])
    print(f"Found {len(csv_files)} CSV files:", flush=True)
    for csv_file in csv_files:
        print(f"  - {csv_file}", flush=True)
    sys.stdout.flush()

    # Load all CSV files
    dataframes = {}
    for csv_file in csv_files:
        print(f"\nLoading {csv_file}...", flush=True)
        df = pd.read_csv(csv_file)
        dataframes[csv_file] = df
        print(f"  Rows: {len(df)}, Columns: {len(df.columns)}", flush=True)
        print(f"  Columns: {list(df.columns)}", flush=True)
    sys.stdout.flush()

    # Concatenate all DataFrames vertically
    print("\n" + "="*60, flush=True)
    print("Merging CSV files...", flush=True)
    print("="*60, flush=True)
    
    combined_df = pd.concat(list(dataframes.values()), ignore_index=True)
    print(f"✓ Successfully combined {len(dataframes)} files", flush=True)
    print(f"  Total rows: {len(combined_df)}", flush=True)
    print(f"  Total columns: {len(combined_df.columns)}", flush=True)
    sys.stdout.flush()

    # Save the combined DataFrame to a new CSV file
    print("\n" + "="*60, flush=True)
    print("Saving combined file...", flush=True)
    print("="*60, flush=True)

    output_file = 'motor_data_combined.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"✓ Combined CSV saved as: {output_file}", flush=True)
    print(f"  File location: {os.path.abspath(output_file)}", flush=True)
    print(f"  File size: {os.path.getsize(output_file) / 1024:.2f} KB", flush=True)
    sys.stdout.flush()

    print("\nMerge completed successfully!", flush=True)

except Exception as e:
    print(f"ERROR: {str(e)}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
