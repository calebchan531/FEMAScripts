import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os

def visualize_data(df, target_column='ihpAmount'):
    # Display basic info about the dataset
    print(df.info())
    print(df.describe())

    # Histogram of target column (ihpAmount)
    plt.figure(figsize=(8, 6))
    sns.histplot(df[target_column], kde=True, color='blue')
    plt.title(f"Distribution of {target_column}")
    plt.xlabel(target_column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(10, 6))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # Pairplot for selected numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        sns.pairplot(df[numeric_cols], height=2.5)
        plt.tight_layout()
        plt.show()

    # Boxplot to detect outliers in the target column
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[target_column], color='green')
    plt.title(f"Boxplot of {target_column}")
    plt.xlabel(target_column)
    plt.tight_layout()
    plt.show()

def absolute_accuracy(csv_path, target_column='ihpAmount', chunksize=10000):
    chunk_iter = pd.read_csv(csv_path, chunksize=chunksize)

    # Initialize variables to accumulate results
    acc_zeroR_list = []
    acc_tree_list = []
    acc_forest_list = []
    y_test_all = []
    y_pred_zeroR_all = []
    y_pred_tree_all = []
    y_pred_forest_all = []
    all_data = []  # To store all chunks for visualization at the end

    for chunk in chunk_iter:
        df = chunk.dropna()
        if target_column not in df.columns:
            print(f"Target column '{target_column}' not found in {csv_path}.")
            return

        # Store the chunk for visualization later
        all_data.append(df)

        X = pd.get_dummies(df.drop(columns=[target_column]))
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # ZeroR
        y_pred_zeroR = np.full_like(y_test, y_train.mean(), dtype=np.float64)
        mae_zeroR = mean_absolute_error(y_test, y_pred_zeroR)
        
        # Random Tree
        tree = DecisionTreeRegressor(random_state=42)
        tree.fit(X_train, y_train)
        y_pred_tree = tree.predict(X_test)
        mae_tree = mean_absolute_error(y_test, y_pred_tree)

        # Random Forest
        forest = RandomForestRegressor(n_estimators=100, random_state=42)
        forest.fit(X_train, y_train)
        y_pred_forest = forest.predict(X_test)
        mae_forest = mean_absolute_error(y_test, y_pred_forest)

        # Absolute scale: compare to mean of true values
        mean_abs_y = np.mean(np.abs(y_test))
        acc_zeroR = 100 * (1 - mae_zeroR / mean_abs_y)
        acc_tree = 100 * (1 - mae_tree / mean_abs_y)
        acc_forest = 100 * (1 - mae_forest / mean_abs_y)

        # Store the results for later aggregation
        acc_zeroR_list.append(abs(acc_zeroR))
        acc_tree_list.append(abs(acc_tree))
        acc_forest_list.append(abs(acc_forest))
        
        y_test_all.extend(y_test)
        y_pred_zeroR_all.extend(y_pred_zeroR)
        y_pred_tree_all.extend(y_pred_tree)
        y_pred_forest_all.extend(y_pred_forest)

    # After processing all chunks, visualize the data
    full_df = pd.concat(all_data, ignore_index=True)
    visualize_data(full_df, target_column)

    # Calculate overall accuracies after processing all chunks
    overall_acc_zeroR = np.mean(acc_zeroR_list)
    overall_acc_tree = np.mean(acc_tree_list)
    overall_acc_forest = np.mean(acc_forest_list)

    # Plot Model Comparison
    models = ['ZeroR', 'Random Tree', 'Random Forest']
    accuracies = [overall_acc_zeroR, overall_acc_tree, overall_acc_forest]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, accuracies, color=['gray', 'skyblue', 'green'])
    plt.title(f"Model Accuracy Comparison for {os.path.basename(csv_path)}")
    plt.ylabel("Accuracy (%)")
    plt.ylim(min(0, min(accuracies) - 5), 100)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom')

    # Save the figure as a PDF in the current directory
    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_name = f"model_accuracy_comparison_{os.path.basename(csv_path).replace('.csv', '')}.pdf"
    #file_path = os.path.join(current_directory, file_name)
    plt.tight_layout()
    plt.savefig(file_path, format="pdf")

    # Show the plot
    plt.show()

# Run the function with your desired CSV file
absolute_accuracy("other.csv")  # Change "other.csv" to your new file name
 