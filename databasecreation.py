import pandas as pd
import numpy as np
import time
import gc  # For garbage collection

start_time = time.time()

print("Loading the disaster declarations dataset...")
disaster_declarations = pd.read_csv('DisasterDeclarationsSummaries.csv')

# Select only the columns we need from disaster declarations
disaster_columns = disaster_declarations[['disasterNumber', 'declarationType', 'declarationTitle']].copy()
disaster_columns['disasterNumber'] = disaster_columns['disasterNumber'].astype(str)

# Free up memory
del disaster_declarations
gc.collect()  # Force garbage collection
print(f"Disaster declarations processed: {len(disaster_columns)} rows")

# Convert to dictionary for faster lookups
disaster_type_dict = dict(zip(disaster_columns['disasterNumber'], disaster_columns['declarationType']))
disaster_title_dict = dict(zip(disaster_columns['disasterNumber'], disaster_columns['declarationTitle']))

# Free up more memory
del disaster_columns
gc.collect()
print("Lookup dictionaries created for quick joining")

# Process the large file in chunks
chunk_size = 100000  # Reduced chunk size for better progress visibility
output_file = 'ihp_vr_enriched.csv'
first_chunk = True
total_rows = 0
duplicate_count = 0
chunk_count = 0

print(f"Processing IHP-VR dataset in chunks of {chunk_size} rows...")

for chunk in pd.read_csv('IndividualsAndHouseholdsProgramValidRegistrations.csv', 
                         chunksize=chunk_size, 
                         low_memory=False):
    
    chunk_start_time = time.time()
    chunk_count += 1
    print(f"Processing chunk #{chunk_count}...")
    
    # Track original chunk size
    original_size = len(chunk)
    
    # Remove duplicates in the chunk
    chunk.drop_duplicates(inplace=True)
    current_chunk_duplicates = original_size - len(chunk)
    duplicate_count += current_chunk_duplicates
    
    if current_chunk_duplicates > 0:
        print(f"  Removed {current_chunk_duplicates} duplicates in this chunk")
    
    # Ensure disasterNumber is a string
    chunk['disasterNumber'] = chunk['disasterNumber'].astype(str)
    
    # Add new columns with default values
    chunk['declarationType'] = 'Unknown Type'
    chunk['declarationTitle'] = 'No Title Available'
    
    # Fill empty values in all columns with type-appropriate values
    print("  Filling missing values in all columns...")
    
    # Process in batches to show progress
    for col in chunk.columns:
        # Skip the newly added columns as we'll handle those separately
        if col in ['declarationType', 'declarationTitle']:
            continue
            
        # Determine the data type and fill accordingly
        if pd.api.types.is_numeric_dtype(chunk[col]):
            # For numeric columns, use 0 or -1 depending on if it might be an ID
            if col.lower().endswith('id') or 'number' in col.lower():
                chunk[col].fillna(-1, inplace=True)
            else:
                chunk[col].fillna(0, inplace=True)
                
        elif pd.api.types.is_datetime64_dtype(chunk[col]):
            # For date columns, use a default date (1900-01-01)
            chunk[col].fillna(pd.Timestamp('1900-01-01'), inplace=True)
            
        else:
            # For string/object columns
            if col.lower().endswith('date') or 'date' in col.lower():
                # Date-like string columns
                chunk[col].fillna('1900-01-01', inplace=True)
            elif col.lower().endswith('flag') or 'flag' in col.lower():
                # Flag columns
                chunk[col].fillna('N', inplace=True)
            elif 'state' in col.lower():
                # State columns
                chunk[col].fillna('NA', inplace=True)
            elif 'zip' in col.lower():
                # Zip code columns
                chunk[col].fillna('00000', inplace=True)
            else:
                # General string columns
                chunk[col].fillna('Unknown', inplace=True)
    
    # Use vectorized operations for joining
    # Get unique disaster numbers in this chunk for faster processing
    unique_disaster_nums = chunk['disasterNumber'].unique()
    print(f"  Processing {len(unique_disaster_nums)} unique disaster numbers")
    
    # Process in small batches to show progress
    batch_size = max(1, len(unique_disaster_nums) // 10)
    for i in range(0, len(unique_disaster_nums), batch_size):
        batch = unique_disaster_nums[i:i+batch_size]
        for disaster_num in batch:
            # Update all matching rows at once
            mask = chunk['disasterNumber'] == disaster_num
            if disaster_num in disaster_type_dict:
                chunk.loc[mask, 'declarationType'] = disaster_type_dict[disaster_num]
            if disaster_num in disaster_title_dict:
                chunk.loc[mask, 'declarationTitle'] = disaster_title_dict[disaster_num]
        
        if i % (batch_size * 5) == 0 and i > 0:
            print(f"    Processed {i}/{len(unique_disaster_nums)} disaster numbers...")
    
    # Write to CSV
    if first_chunk:
        chunk.to_csv(output_file, index=False)
        first_chunk = False
    else:
        chunk.to_csv(output_file, mode='a', header=False, index=False)
    
    total_rows += len(chunk)
    chunk_time = time.time() - chunk_start_time
    print(f"Chunk #{chunk_count} completed in {chunk_time:.2f} seconds")
    print(f"Total progress: {total_rows:,} rows processed, {duplicate_count:,} duplicates removed")
    print(f"Elapsed time: {(time.time() - start_time)/60:.2f} minutes")
    
    # Force garbage collection to free memory
    del chunk
    gc.collect()

print(f"\nJoin completed successfully. New dataset saved as '{output_file}'")
print(f"Total rows in final dataset: {total_rows:,}")
print(f"Total duplicates removed: {duplicate_count:,}")
print(f"Total processing time: {(time.time() - start_time)/60:.2f} minutes")