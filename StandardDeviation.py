import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['declarationDate'], low_memory=False)
    return df

# Compute and plot standard deviation
def standard_deviation_graph(file_path, window=30):
    df = load_data(file_path)

    # Ensure necessary columns exist
    if 'ihpAmount' not in df.columns or 'declarationDate' not in df.columns:
        raise ValueError("Columns 'ihpAmount' or 'declarationDate' not found in the CSV file.")

    # Drop NaN values
    df = df.dropna(subset=['ihpAmount', 'declarationDate'])

    # Sort by declarationDate
    df = df.sort_values(by='declarationDate')

    # Compute rolling standard deviation
    df['rolling_std'] = df['ihpAmount'].rolling(window=window, min_periods=1).std()

    # Plot the standard deviation over time
    plt.figure(figsize=(10, 5))
    plt.plot(df['declarationDate'], df['rolling_std'], color='purple', linewidth=2, label=f'Rolling Std Dev (window={window})')
    plt.xlabel('Declaration Date')
    plt.ylabel('Standard Deviation of ihpAmount')
    plt.title('Standard Deviation of ihpAmount Over Time')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

# 
standard_deviation_graph('data_03102025.csv', window=30)
