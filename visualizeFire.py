import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def main():
    print("Starting Error Bars Visualization Script")
    
    # Create output directory
    output_dir = 'error_bars_output'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Read the data (in chunks if file is large)
    print("Reading Fire.csv file...")
    try:
        # Try reading the full file first
        df = pd.read_csv('Fire.csv')
        print(f"Successfully read full dataset: {len(df)} rows")
    except:
        print("File too large for direct reading, trying with chunks...")
        try:
            # Read in chunks
            chunks = []
            for chunk in pd.read_csv('Fire.csv', chunksize=10000):
                chunks.append(chunk)
                print(f"Read chunk of {len(chunk)} rows")
                if len(chunks) * 10000 >= 50000:  # Limit to ~50,000 rows for faster processing
                    break
            df = pd.concat(chunks)
            print(f"Successfully read dataset in chunks: {len(df)} rows")
        except Exception as e:
            print(f"Error reading file in chunks: {e}")
            print("Trying alternative approach...")
            
            # If all else fails, read first N lines directly
            try:
                with open('Fire.csv', 'r') as f:
                    header = f.readline()
                    lines = [header]
                    for i, line in enumerate(f):
                        if i < 50000:  # Read up to 50,000 lines
                            lines.append(line)
                        else:
                            break
                
                with open('temp_sample.csv', 'w') as f:
                    f.writelines(lines)
                
                df = pd.read_csv('temp_sample.csv')
                print(f"Successfully read sample dataset: {len(df)} rows")
            except Exception as e:
                print(f"Failed to read data: {e}")
                return
    
    # Preprocess data
    print("Preprocessing data...")
    
    # Convert age categories to numeric
    age_mapping = {
        '19-34': 27,
        '35-49': 42,
        '50-64': 57,
        '65+': 75
    }
    
    if 'applicantAge' in df.columns:
        df['applicantAgeNumeric'] = df['applicantAge'].map(age_mapping)
    
    # Convert ownership status to numeric
    if 'ownRent' in df.columns:
        df['ownRentNumeric'] = df['ownRent'].apply(lambda x: 1 if x == 'Owner' else 0)
    
    # Define columns to analyze
    print("Identifying columns for analysis...")
    primary_columns = [
        'applicantAgeNumeric', 
        'occupantsUnderTwo', 
        'grossIncome', 
        'ownRentNumeric'
    ]
    
    # Add additional numeric columns that might be useful
    additional_columns = [
        'ihpAmount', 
        'haAmount', 
        'onaAmount', 
        'personalPropertyAmount', 
        'rentalAssistanceAmount'
    ]
    
    # Verify which columns actually exist in the dataframe
    valid_columns = [col for col in primary_columns + additional_columns if col in df.columns]
    print(f"Found {len(valid_columns)} valid columns for analysis: {valid_columns}")
    
    # Generate column labels for plotting
    column_labels = {
        'applicantAgeNumeric': 'Applicant Age',
        'occupantsUnderTwo': 'Occupants Under Two',
        'grossIncome': 'Gross Income',
        'ownRentNumeric': 'Own/Rent (1=Owner)',
        'ihpAmount': 'IHP Amount',
        'haAmount': 'HA Amount',
        'onaAmount': 'ONA Amount',
        'personalPropertyAmount': 'Personal Property',
        'rentalAssistanceAmount': 'Rental Assistance'
    }
    
    # Calculate means and standard errors for each column
    print("Calculating means and standard errors...")
    means = []
    std_errs = []
    rel_errors = []
    labels = []
    
    for col in valid_columns:
        # Get values, excluding NaN and null
        values = df[col].dropna().values
        
        if len(values) > 0:
            mean = np.mean(values)
            # Standard error of the mean
            std_err = np.std(values) / np.sqrt(len(values))
            # Relative error (coefficient of variation)
            rel_error = np.std(values) / mean if mean != 0 else 0
            
            means.append(mean)
            std_errs.append(std_err)
            rel_errors.append(rel_error)
            labels.append(column_labels.get(col, col))
            
            print(f"{column_labels.get(col, col)}:")
            print(f"  Mean: {mean}")
            print(f"  Standard Error: {std_err}")
            print(f"  Relative Error: {rel_error:.2%}")
    
    # Create error bar plot
    print("\nCreating error bar plots...")
    
    # Plot means with error bars
    plt.figure(figsize=(12, 8))
    x_pos = np.arange(len(labels))
    
    # Calculate confidence interval (95%)
    ci_low = [m - 1.96 * se for m, se in zip(means, std_errs)]
    ci_high = [m + 1.96 * se for m, se in zip(means, std_errs)]
    
    # Create bar plot
    bars = plt.bar(x_pos, means, align='center', alpha=0.7, color='skyblue', capsize=10)
    plt.errorbar(x_pos, means, yerr=np.array([means - np.array(ci_low), np.array(ci_high) - means]), fmt='none', ecolor='black', capsize=5)
    
    # Add labels
    plt.xlabel('Variables')
    plt.ylabel('Mean Value')
    plt.title('Mean Values with 95% Confidence Interval Error Bars', fontsize=14)
    plt.xticks(x_pos, labels, rotation=45, ha='right')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'mean_values_with_error_bars.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'mean_values_with_error_bars.pdf'), bbox_inches='tight')
    print(f"Saved mean values plot to {output_dir}/mean_values_with_error_bars.png")
    
    plt.close()
    
    # Plot relative errors
    plt.figure(figsize=(12, 8))
    
    # Convert to percentage for better readability
    rel_errors_pct = [re * 100 for re in rel_errors]
    
    # Create bar plot for relative errors
    plt.bar(x_pos, rel_errors_pct, align='center', alpha=0.7, color='lightcoral')
    
    # Add labels
    plt.xlabel('Variables')
    plt.ylabel('Relative Error (%)')
    plt.title('Relative Error by Variable (Standard Deviation / Mean)', fontsize=14)
    plt.xticks(x_pos, labels, rotation=45, ha='right')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'relative_errors.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'relative_errors.pdf'), bbox_inches='tight')
    print(f"Saved relative errors plot to {output_dir}/relative_errors.png")
    
    plt.close()
    
    # Create a combined visualization
    plt.figure(figsize=(15, 10))
    
    # Set up subplots
    plt.subplot(2, 1, 1)
    plt.bar(x_pos, means, align='center', alpha=0.7, color='skyblue', capsize=10)
    plt.errorbar(x_pos, means, yerr=np.array([means - np.array(ci_low), np.array(ci_high) - means]), fmt='none', ecolor='black', capsize=5)
    plt.title('Mean Values with 95% Confidence Interval Error Bars', fontsize=14)
    plt.xticks(x_pos, labels, rotation=45, ha='right')
    plt.ylabel('Mean Value')
    
    plt.subplot(2, 1, 2)
    plt.bar(x_pos, rel_errors_pct, align='center', alpha=0.7, color='lightcoral')
    plt.title('Relative Error by Variable (Standard Deviation / Mean)', fontsize=14)
    plt.xticks(x_pos, labels, rotation=45, ha='right')
    plt.ylabel('Relative Error (%)')
    
    plt.tight_layout()
    
    # Save combined figure
    plt.savefig(os.path.join(output_dir, 'combined_error_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'combined_error_analysis.pdf'), bbox_inches='tight')
    print(f"Saved combined plot to {output_dir}/combined_error_analysis.png")
    
    # Create a column-by-column error distribution visualization
    print("\nCreating individual column error distributions...")
    
    for i, col in enumerate(valid_columns):
        values = df[col].dropna().values
        
        if len(values) > 0:
            plt.figure(figsize=(10, 6))
            
            # Create histogram with mean and error bars
            plt.hist(values, bins=30, alpha=0.7, color='skyblue')
            
            # Add mean line
            plt.axvline(means[i], color='red', linestyle='--', linewidth=2, label=f'Mean: {means[i]:.3f}')
            
            # Add confidence interval
            plt.axvline(ci_low[i], color='black', linestyle=':', linewidth=1.5, label=f'95% CI: [{ci_low[i]:.3f}, {ci_high[i]:.3f}]')
            plt.axvline(ci_high[i], color='black', linestyle=':', linewidth=1.5)
            
            # Add labels
            plt.title(f'Distribution of {labels[i]} with Error Ranges', fontsize=14)
            plt.xlabel(labels[i])
            plt.ylabel('Frequency')
            plt.legend()
            plt.tight_layout()
            
            # Save figure
            filename = col.replace('/', '_').lower()
            plt.savefig(os.path.join(output_dir, f'{filename}_distribution.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(output_dir, f'{filename}_distribution.pdf'), bbox_inches='tight')
            print(f"Saved {labels[i]} distribution to {output_dir}/{filename}_distribution.png")
            
            plt.close()
    
    print("\nAll visualizations completed successfully!")
    print(f"Output files are located in: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    print("Error Bars Visualization Script")
    print("===============================")
    try:
        main()
        print("\nScript completed successfully.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        print(traceback.format_exc())
        print("\nScript terminated with errors.")
        sys.exit(1)