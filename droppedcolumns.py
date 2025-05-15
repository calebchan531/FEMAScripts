import pandas as pd

# File paths
input_file_path = "Cleaned IHPVR Disaster Summaries.csv"  # Update this with actual file path
output_file_path = "cleaned_fema_filtered.csv"

# Define chunk size for large file processing
chunk_size = 100000  # Adjust based on system capabilities

# Columns to filter out "Unknown" values
filter_columns = ["ihpEligible", "applicantAge", "ownRent"]

# Open a new file for writing filtered data
with pd.read_csv(input_file_path, chunksize=chunk_size) as reader:
    for i, chunk in enumerate(reader):
        # Remove rows where any of the filter_columns have "Unknown"
        filtered_chunk = chunk[~chunk[filter_columns].isin(["Unknown"]).any(axis=1)]

        # Append to output file (write header only for the first chunk)
        filtered_chunk.to_csv(output_file_path, mode='a', header=(i == 0), index=False)

# Display first few rows of cleaned dataset
import ace_tools as tools
tools.display_dataframe_to_user(name="Filtered FEMA Dataset", dataframe=pd.read_csv(output_file_path, nrows=100))

print(f"Download your cleaned dataset here: {output_file_path}")
