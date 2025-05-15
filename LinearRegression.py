import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the CSV file
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['declarationDate'], low_memory=False)  # Parse declarationDate as datetime, avoid dtype warning
    return df

# Perform linear regression
def linear_regression_graph(file_path):
    df = load_data(file_path)
    
    # Ensure necessary columns exist
    if 'ihpAmount' not in df.columns or 'declarationDate' not in df.columns:
        raise ValueError("Columns 'ihpAmount' or 'declarationDate' not found in the CSV file.")
    
    # Prepare the data
    df = df.dropna(subset=['ihpAmount', 'declarationDate'])  # Drop NaN values
    df = df.sort_values(by='declarationDate')  # Sort by declarationDate to ensure correct order
    
    # Normalize dates by converting to float years
    min_date = df['declarationDate'].min()
    df['normalizedDate'] = (df['declarationDate'] - min_date).dt.days / 365.25  # Convert to year difference
    
    X = df['normalizedDate'].values.reshape(-1, 1)
    y = df['ihpAmount'].values.reshape(-1, 1)
    
    # Fit the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predictions
    y_pred = model.predict(X)
    
    # Plot the data
    plt.scatter(df['declarationDate'], y, color='blue', s=10, label='Actual Data')  # Reduce dot size with s=10
    plt.plot(df['declarationDate'], y_pred, color='red', linewidth=2, label='Regression Line')
    plt.xlabel('Declaration Date')
    plt.ylabel('ihpAmount')
    plt.title('Linear Regression on ihpAmount by Declaration Date')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

# 
linear_regression_graph('Typhoon.csv')