import pandas as pd

# File paths
input_file_path = "Cleaned IHPVR Disaster Summaries.csv"  # Update with actual file path
output_file_path = "cleaned_fema_dataset.csv"

# Define chunk size (adjustable based on dataset size)
chunk_size = 100000  # Process 100,000 rows at a time

# List of categorical columns with 'Unknown' values
categorical_cols = ["applicantAge", "homeOwnersInsurance", "floodInsurance", "renterDamageLevel"]

# List of numerical columns where '0' might indicate missing values
numerical_cols = ["floodDamageAmount", "foundationDamageAmount", "roofDamageAmount"]

# First pass: Calculate the mode for categorical columns
categorical_modes = {}

for chunk in pd.read_csv(input_file_path, chunksize=chunk_size):
    for col in categorical_cols:
        if col in chunk.columns:
            mode_value = chunk[col][chunk[col] != "Unknown"].mode()
            if not mode_value.empty:
                categorical_modes[col] = mode_value[0]  # Store mode

# Second pass: Process dataset in chunks and apply transformations
with pd.read_csv(input_file_path, chunksize=chunk_size) as reader:
    for i, chunk in enumerate(reader):
        # Replace 'Unknown' with precomputed most frequent category
        for col in categorical_cols:
            if col in chunk.columns and col in categorical_modes:
                chunk[col] = chunk[col].replace("Unknown", categorical_modes[col])

        # Replace 0 values in numerical columns with median values (computed per chunk)
        for col in numerical_cols:
            if col in chunk.columns:
                median_value = chunk[col].median()
                chunk[col] = chunk[col].replace(0, median_value)

        # Save processed chunk (append after the first write)
        chunk.to_csv(output_file_path, mode='a', header=(i == 0), index=False)

# Display the first few rows of the cleaned dataset
import ace_tools as tools
tools.display_dataframe_to_user(name="Chunk Processed FEMA Dataset", dataframe=pd.read_csv(output_file_path, nrows=100))

print(f"Download your cleaned dataset here: {output_file_path}")
