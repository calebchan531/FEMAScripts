import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Force immediate printing of messages
def print_status(message):
    print(message, flush=True)

# Define a safe chunk reader function
def read_csv_safely(file_path, max_rows=None):
    """Read a CSV file safely, even if it's very large"""
    print_status(f"Attempting to read {file_path}")
    
    # First try - direct read with row limit
    if max_rows:
        try:
            print_status(f"Trying direct read with {max_rows} row limit...")
            df = pd.read_csv(file_path, nrows=max_rows)
            print_status(f"Success! Read {len(df)} rows directly")
            return df
        except Exception as e:
            print_status(f"Direct read failed: {e}")
    
    # Second try - chunk reading
    try:
        print_status("Trying chunk reading approach...")
        chunks = []
        chunk_size = 5000  # Use a small chunk size
        total_rows = 0
        
        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
            print_status(f"Reading chunk {i+1} with {len(chunk)} rows")
            chunks.append(chunk)
            total_rows += len(chunk)
            
            # Break if we've read enough rows or chunks
            if max_rows and total_rows >= max_rows:
                print_status(f"Reached {max_rows} row limit")
                break
            
            if i >= 9:  # Limit to ~10 chunks
                print_status("Reached chunk limit")
                break
        
        if chunks:
            df = pd.concat(chunks)
            print_status(f"Success! Read {len(df)} rows in {len(chunks)} chunks")
            return df
        else:
            print_status("No data was read in chunks")
    except Exception as e:
        print_status(f"Chunk reading failed: {e}")
    
    # Last try - manual line reading
    try:
        print_status("Trying manual line reading approach...")
        with open(file_path, 'r') as f:
            # Read header
            header = f.readline().strip()
            print_status("Read header: " + header[:50] + "...")
            
            # Prepare for sampling
            line_count = 0
            sample_lines = [header]
            sample_limit = max_rows or 50000
            
            # Read a sample of lines
            for i, line in enumerate(f):
                if i % 1000 == 0:
                    print_status(f"Scanned {i} lines...")
                
                if i < sample_limit and line.strip():
                    sample_lines.append(line)
                    line_count += 1
                
                if line_count >= sample_limit:
                    break
            
            # Save to temporary file and read
            temp_file = 'temp_sample.csv'
            with open(temp_file, 'w') as out_f:
                out_f.writelines(sample_lines)
            
            print_status(f"Created sample with {len(sample_lines)} lines")
            df = pd.read_csv(temp_file)
            print_status(f"Success! Read {len(df)} rows from sample file")
            
            # Clean up temp file
            try:
                os.remove(temp_file)
            except:
                pass
                
            return df
    except Exception as e:
        print_status(f"Manual reading failed: {e}")
        raise RuntimeError(f"All methods failed to read {file_path}")

def main():
    # Create output directory
    output_dir = 'fire_viz_output'
    os.makedirs(output_dir, exist_ok=True)
    print_status(f"Created output directory: {output_dir}")
    
    # Try to read the CSV file safely
    try:
        print_status("\n=== READING DATA ===")
        df = read_csv_safely('Fire.csv', max_rows=50000)
        print_status(f"Data summary: {len(df)} rows, {len(df.columns)} columns")
        
        # Print a few column names to verify
        print_status(f"Sample columns: {', '.join(df.columns[:5])}...")
    except Exception as e:
        print_status(f"Failed to read data: {e}")
        return
    
    # Process the data
    print_status("\n=== PROCESSING DATA ===")
    
    # Convert age categories to numeric values
    if 'applicantAge' in df.columns:
        print_status("Converting age categories to numeric values")
        age_mapping = {
            '19-34': 27,
            '35-49': 42,
            '50-64': 57,
            '65+': 75
        }
        df['applicantAgeNumeric'] = df['applicantAge'].map(age_mapping)
        print_status(f"Age conversion complete. Null values: {df['applicantAgeNumeric'].isna().sum()}")
    else:
        print_status("Warning: 'applicantAge' column not found")
    
    # Convert ownRent to binary
    if 'ownRent' in df.columns:
        print_status("Converting ownership status to binary")
        df['ownRentNumeric'] = df['ownRent'].apply(lambda x: 1 if x == 'Owner' else 0)
        print_status("Ownership conversion complete")
    else:
        print_status("Warning: 'ownRent' column not found")
    
    # Create correlation matrix
    print_status("\n=== CREATING VISUALIZATIONS ===")
    print_status("1. Creating Pearson correlation matrix")
    
    # Define columns for correlation
    columns = [col for col in ['applicantAgeNumeric', 'occupantsUnderTwo', 'grossIncome', 'ownRentNumeric'] 
               if col in df.columns]
    
    if len(columns) < 2:
        print_status("Not enough columns for correlation matrix")
    else:
        # Set up nice labels
        labels = {
            'applicantAgeNumeric': 'Applicant Age',
            'occupantsUnderTwo': 'Occupants Under Two',
            'grossIncome': 'Gross Income',
            'ownRentNumeric': 'Own/Rent'
        }
        
        # Create a dataframe with the selected columns
        corr_df = df[columns].copy()
        # Rename columns for better display
        corr_df.columns = [labels.get(col, col) for col in columns]
        
        # Calculate correlation
        corr_matrix = corr_df.corr(method='pearson')
        print_status("Correlation matrix calculated:")
        print(corr_matrix)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create a custom colormap
        colors = ["#4575b4", "#91bfdb", "#e0f3f8", "#ffffbf", "#fee090", "#fc8d59", "#d73027"]
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=100)
        
        # Plot heatmap
        sns.heatmap(
            corr_matrix, 
            annot=True,
            cmap=cmap,
            vmin=-1, 
            vmax=1, 
            center=0,
            square=True, 
            linewidths=.5,
            cbar_kws={"shrink": .8},
            fmt=".2f"
        )
        
        plt.title('Pearson Correlation Matrix', fontsize=16, pad=20)
        plt.tight_layout()
        
        # Save the plot
        output_file = os.path.join(output_dir, 'pearson_correlation.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print_status(f"Saved correlation matrix to {output_file}")
        plt.close()
    
    # Create age distribution plot
    if 'applicantAge' in df.columns:
        print_status("2. Creating age distribution plot")
        plt.figure(figsize=(10, 6))
        
        # Count age groups
        age_counts = df['applicantAge'].value_counts().sort_index()
        print_status(f"Age counts: {dict(age_counts)}")
        
        # Define custom order if possible
        try:
            custom_order = ['19-34', '35-49', '50-64', '65+']
            ordered_counts = age_counts.reindex(custom_order)
            
            # Create bar plot
            sns.barplot(x=ordered_counts.index, y=ordered_counts.values, palette='viridis')
            
            plt.title('Distribution of Applicant Age Groups', fontsize=16)
            plt.xlabel('Age Group')
            plt.ylabel('Count')
            plt.tight_layout()
            
            # Save the plot
            output_file = os.path.join(output_dir, 'age_distribution.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print_status(f"Saved age distribution to {output_file}")
            plt.close()
        except Exception as e:
            print_status(f"Error creating age plot: {e}")
    
    # Create ownership distribution plot
    if 'ownRent' in df.columns:
        print_status("3. Creating ownership distribution plot")
        try:
            plt.figure(figsize=(8, 6))
            
            # Count ownership status
            ownership_counts = df['ownRent'].value_counts()
            print_status(f"Ownership counts: {dict(ownership_counts)}")
            
            # Create pie chart
            plt.pie(
                ownership_counts.values, 
                labels=ownership_counts.index, 
                autopct='%1.1f%%',
                startangle=90,
                colors=sns.color_palette('Set2')
            )
            plt.axis('equal')
            plt.title('Distribution of Ownership Status', fontsize=16)
            plt.tight_layout()
            
            # Save the plot
            output_file = os.path.join(output_dir, 'ownership_distribution.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print_status(f"Saved ownership distribution to {output_file}")
            plt.close()
        except Exception as e:
            print_status(f"Error creating ownership plot: {e}")
    
    # Create income distribution plot
    if 'grossIncome' in df.columns:
        print_status("4. Creating income distribution plot")
        try:
            plt.figure(figsize=(10, 6))
            
            # Count income categories
            income_counts = df['grossIncome'].value_counts().sort_index()
            print_status(f"Income category counts: {dict(income_counts)}")
            
            # Create bar plot
            sns.barplot(x=income_counts.index, y=income_counts.values, palette='Blues_r')
            
            plt.title('Distribution of Gross Income Categories', fontsize=16)
            plt.xlabel('Income Category')
            plt.ylabel('Count')
            plt.tight_layout()
            
            # Save the plot
            output_file = os.path.join(output_dir, 'income_distribution.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print_status(f"Saved income distribution to {output_file}")
            plt.close()
        except Exception as e:
            print_status(f"Error creating income plot: {e}")
    
    # Created stacked chart of ownership by age
    if 'applicantAge' in df.columns and 'ownRent' in df.columns:
        print_status("5. Creating ownership by age group plot")
        try:
            plt.figure(figsize=(10, 6))
            
            # Create crosstab
            ct = pd.crosstab(df['applicantAge'], df['ownRent'])
            print_status("Ownership by age crosstab:")
            print(ct)
            
            # Try to sort by age order
            try:
                custom_order = ['19-34', '35-49', '50-64', '65+']
                ct = ct.reindex(custom_order)
            except:
                pass
            
            # Create stacked bar plot
            ct_pct = ct.div(ct.sum(axis=1), axis=0)
            ct_pct.plot(kind='bar', stacked=True, colormap='Set2')
            
            plt.title('Ownership Status by Age Group', fontsize=16)
            plt.xlabel('Age Group')
            plt.ylabel('Percentage')
            plt.legend(title='Status')
            plt.tight_layout()
            
            # Save the plot
            output_file = os.path.join(output_dir, 'ownership_by_age.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print_status(f"Saved ownership by age to {output_file}")
            plt.close()
        except Exception as e:
            print_status(f"Error creating ownership by age plot: {e}")
    
    print_status("\n=== VISUALIZATION COMPLETE ===")
    print_status(f"All visualizations saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" FIRE DATA VISUALIZER ".center(60, "*"))
    print("="*60 + "\n")
    
    try:
        main()
        print("\n" + "="*60)
        print(" PROCESSING COMPLETE ".center(60, "*"))
        print("="*60 + "\n")
    except Exception as e:
        print("\nERROR: An unexpected error occurred:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        print("\nStack trace:")
        print(traceback.format_exc())
        print("\nPlease check the error message above.")