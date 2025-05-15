import pandas as pd
import numpy as np

# Suppress dtype warnings
pd.options.mode.chained_assignment = None  

# List of 9 input files
input_files = [
    "Fire.csv",
    "Flood.csv",
    "Hurricane.csv",
    "Mud_Landslide.csv",
    "Other.csv",
    "Severe_Ice_Storm.csv",
    "Severe_Storm.csv",
    "Tornado.csv",
    "Typhoon.csv"
]

# Corresponding output file names
output_files = [
    "Fire_encoded.csv",
    "Flood_encoded.csv",
    "Hurricane_encoded.csv",
    "Mud_Landslide_encoded.csv",
    "Other_encoded.csv",
    "Severe_Ice_Storm_encoded.csv",
    "Severe_Storm_encoded.csv",
    "Tornado_encoded.csv",
    "Typhoon_encoded.csv"
]

chunk_size = 100000  # Adjust chunk size as needed

for input_path, output_path in zip(input_files, output_files):
    processed_chunks = []

    # Create a separate encoding map for each file
    encoding_maps = {col: {} for col in ['residenceType', 'damageCity', 'county', 'applicantAge', 'ownRent', 'haStatus']}

    for chunk in pd.read_csv(input_path, chunksize=chunk_size, dtype=str, low_memory=False):
        # Drop 'highWaterLocation' column if it exists
        chunk.drop(columns=['highWaterLocation'], errors='ignore', inplace=True)

        # Handle missing (NaN) values by replacing them with '?'
        chunk.fillna('?', inplace=True)  # Use '?' as WEKA's representation for missing values

        # Encode categorical columns
        for col in encoding_maps.keys():
            if col in chunk.columns:
                unique_values = chunk[col].unique()
                # Filter out the '?' placeholder before encoding
                valid_values = [val for val in unique_values if val != '?']
                for val in valid_values:
                    if val not in encoding_maps[col]:
                        encoding_maps[col][val] = len(encoding_maps[col])
                # Map the values, and set '?' as the value for missing data
                chunk[col] = chunk[col].map(encoding_maps[col]).fillna('?')

        processed_chunks.append(chunk)

    # Concatenate the processed chunks
    processed_df = pd.concat(processed_chunks, ignore_index=True)

    # Double-check and handle any remaining NaN values
    processed_df = processed_df.replace(np.nan, '?')  # Ensure any NaN values are replaced by '?'

    # Check for any remaining NaN values before saving
    if processed_df.isnull().values.any():
        print(f"Warning: Found null values in {input_path}. Replacing with '?'")
        processed_df = processed_df.fillna('?')  # Replace any remaining NaN values with '?'

    # Ensure all columns are properly encoded for WEKA
    processed_df.to_csv(output_path, index=False)

    # Save encoding mappings for this file
    mapping_df = []
    for col, mapping in encoding_maps.items():
        for string_val, int_val in mapping.items():
            mapping_df.append([col, string_val, int_val])

    mapping_df = pd.DataFrame(mapping_df, columns=['Column', 'Original_Value', 'Encoded_Value'])
    mapping_file = output_path.replace("_encoded.csv", "_encoding.csv")
    mapping_df.to_csv(mapping_file, index=False)

    print(f"Processed: {input_path} → {output_path}")
    print(f"Encoding mapping saved to: {mapping_file}")

print("All files processed successfully.")
