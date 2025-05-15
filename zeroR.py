import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os

def absolute_accuracy(csv_path, target_column='ihpAmount'):
    df = pd.read_csv(csv_path).dropna()
    if target_column not in df.columns:
        print(f"Target column '{target_column}' not found.")
        return

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

    acc_zeroR = abs(acc_zeroR)
    
    # Plot
    models = ['ZeroR', 'Random Tree', 'Random Forest']
    accuracies = [acc_zeroR, acc_tree, acc_forest]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, accuracies, color=['gray', 'skyblue', 'green'])
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy (%)")
    plt.ylim(min(0, min(accuracies) - 5), 100)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom')

    # Save the figure as a PDF in the current directory
    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_directory, "model_accuracy_comparison.pdf")
    plt.tight_layout()
    plt.savefig(file_path, format="pdf")

    # Show the plot
    plt.show()

# Run the function
absolute_accuracy("Other.csv")
