import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os

# Set up the plotting style
plt.style.use('ggplot')
sns.set(font_scale=1.1)

def read_csv_in_chunks(file_path, chunk_size=10000):
    """Read a large CSV file in chunks"""
    chunks = pd.read_csv(file_path, chunksize=chunk_size)
    return pd.concat(chunks)

def preprocess_data(df):
    """Preprocess the data for correlation analysis"""
    # Create a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Convert age categories to numeric values (midpoint of range)
    age_mapping = {
        '19-34': 27,
        '35-49': 42,
        '50-64': 57,
        '65+': 75
    }
    processed_df['applicantAgeNumeric'] = processed_df['applicantAge'].map(age_mapping)
    
    # Convert ownRent to binary (1 = Owner, 0 = Renter)
    processed_df['ownRentNumeric'] = processed_df['ownRent'].apply(lambda x: 1 if x == 'Owner' else 0)
    
    return processed_df

def calculate_correlation_matrix(df, columns):
    """Calculate the Pearson correlation matrix"""
    return df[columns].corr(method='pearson')

def create_correlation_heatmap(corr_matrix, output_dir, filename_base):
    """Create and save a correlation heatmap"""
    plt.figure(figsize=(10, 8))
    
    # Create a custom colormap (blue to white to red)
    colors = ["#4575b4", "#91bfdb", "#e0f3f8", "#ffffbf", "#fee090", "#fc8d59", "#d73027"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=100)
    
    # Create the heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    heatmap = sns.heatmap(
        corr_matrix, 
        annot=True,
        cmap=cmap,
        vmin=-1, 
        vmax=1, 
        center=0,
        square=True, 
        linewidths=.5,
        cbar_kws={"shrink": .8},
        fmt=".2f",
        mask=mask
    )
    
    # Add title and labels
    plt.title('Pearson Correlation Matrix', fontsize=16, pad=20)
    plt.tight_layout()
    
    # Save as PNG
    png_path = os.path.join(output_dir, f"{filename_base}.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    # Save as PDF
    pdf_path = os.path.join(output_dir, f"{filename_base}.pdf")
    plt.savefig(pdf_path, bbox_inches='tight')
    
    plt.close()
    
    return png_path, pdf_path

def main():
    # Set up output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the data
    print("Reading and processing the CSV file...")
    df = read_csv_in_chunks('Fire.csv')
    
    # Preprocess the data
    print("Preprocessing data...")
    processed_df = preprocess_data(df)
    
    # Calculate required correlation matrix
    print("Calculating basic correlation matrix...")
    basic_cols = ['applicantAgeNumeric', 'occupantsUnderTwo', 'grossIncome', 'ownRentNumeric']
    basic_labels = ['Applicant Age', 'Occupants Under Two', 'Gross Income', 'Own/Rent']
    
    # Rename columns for better display
    corr_df = processed_df[basic_cols].copy()
    corr_df.columns = basic_labels
    
    basic_corr = corr_df.corr(method='pearson')
    
    # Create and save the correlation heatmap
    print("Creating correlation heatmap...")
    png_path, pdf_path = create_correlation_heatmap(
        basic_corr, 
        output_dir, 
        'pearson_correlation_matrix'
    )
    print(f"Saved heatmap to {png_path} and {pdf_path}")
    
    # Calculate extended correlation matrix with additional variables
    print("Calculating extended correlation matrix...")
    extended_cols = [
        'applicantAgeNumeric', 'occupantsUnderTwo', 'grossIncome', 'ownRentNumeric',
        'ihpAmount', 'haAmount', 'onaAmount', 'personalPropertyAmount', 'rentalAssistanceAmount'
    ]
    extended_labels = [
        'Applicant Age', 'Occupants Under Two', 'Gross Income', 'Own/Rent',
        'IHP Amount', 'HA Amount', 'ONA Amount', 'Personal Property', 'Rental Assistance'
    ]
    
    # Select only numeric columns that exist
    available_cols = [col for col in extended_cols if col in processed_df.columns]
    available_labels = [extended_labels[extended_cols.index(col)] for col in available_cols]
    
    if len(available_cols) > 4:  # Only create extended matrix if we have additional columns
        ext_corr_df = processed_df[available_cols].copy()
        ext_corr_df.columns = available_labels
        
        extended_corr = ext_corr_df.corr(method='pearson')
        
        # Create and save the extended correlation heatmap
        print("Creating extended correlation heatmap...")
        ext_png_path, ext_pdf_path = create_correlation_heatmap(
            extended_corr, 
            output_dir, 
            'extended_pearson_correlation_matrix'
        )
        print(f"Saved extended heatmap to {ext_png_path} and {ext_pdf_path}")
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()