import pandas as pd
import os
import re

# Define input file path
input_file_path = "cleaned_fema_filtered.csv"  # Update with actual file path

# Define output directory
output_dir = "split_by_incidentType"
os.makedirs(output_dir, exist_ok=True)

# Define chunk size for reading large datasets
chunk_size = 50000  # Adjust based on available memory

# Function to create safe filenames
def sanitize_filename(name):
    """Replace spaces and special characters to create a safe filename."""
    return re.sub(r'[^a-zA-Z0-9]', '_', name) + ".csv"

# Dictionary to track open file handles
file_handles = {}

# Read dataset in chunks and process
with pd.read_csv(input_file_path, chunksize=chunk_size, dtype=str) as reader:
    for chunk in reader:
        # Ensure incidentType column exists
        if "incidentType" not in chunk.columns:
            raise ValueError("Column 'incidentType' not found in dataset.")

        # Process each unique incidentType in the chunk
        for incident_type, subset in chunk.groupby("incidentType"):
            # Generate a safe filename based on incidentType
            filename = sanitize_filename(incident_type)
            file_path = os.path.join(output_dir, filename)

            # Append the subset to the corresponding file (write header only for first write)
            subset.to_csv(file_path, mode="a", header=not os.path.exists(file_path), index=False)

print(f"Splitting complete! Files are saved in '{output_dir}' directory.")
