import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ✅ STEP 1: Load Dataset
file_path = "Fire.csv"  # Update with actual dataset path
df = pd.read_csv(file_path)

# ✅ STEP 2: Check Class Distribution
target_variable = "ihpEligible"  # Replace with your actual target variable
class_counts = df[target_variable].value_counts()
total_samples = len(df)

print("📊 **Class Distribution:**")
for class_label, count in class_counts.items():
    print(f"  {class_label}: {count} samples ({(count/total_samples)*100:.2f}%)")

# ✅ STEP 3: Calculate ZeroR Expected Accuracy
most_frequent_class = class_counts.idxmax()
zero_r_accuracy = class_counts.max() / total_samples

print(f"\n⚡ **ZeroR Expected Accuracy:** {zero_r_accuracy:.4f} (Always predicts '{most_frequent_class}')")

# ✅ STEP 4: Train & Evaluate ZeroR Model
X = df.drop(columns=[target_variable])  # Features
y = df[target_variable]  # Target variable

# Convert categorical features if needed
X = pd.get_dummies(X, drop_first=True)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train ZeroR (Dummy Classifier)
zero_r_model = DummyClassifier(strategy="most_frequent")
zero_r_model.fit(X_train, y_train)
y_pred = zero_r_model.predict(X_test)

# Compute ZeroR Accuracy on Test Set
zero_r_test_accuracy = accuracy_score(y_test, y_pred)

print(f"\n🧐 **ZeroR Actual Accuracy on Test Data:** {zero_r_test_accuracy:.4f}")

# ✅ STEP 5: Visualize Class Distribution
plt.figure(figsize=(8, 5))
plt.bar(class_counts.index.astype(str), class_counts.values, color="skyblue")
plt.xlabel("Class Labels")
plt.ylabel("Number of Samples")
plt.title("Class Distribution in Dataset")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save the class distribution plot
plot_path = "class_distribution.png"
plt.savefig(plot_path)
plt.show()

print(f"\n📥 **Download the Class Distribution Graph Here:** {plot_path}")
