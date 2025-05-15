import pandas as pd
from scipy.io import arff

# Load the ARFF file
data, meta = arff.loadarff("Otherpersongeographiclocation.arff")

# Convert to DataFrame and decode byte strings
df = pd.DataFrame(data)
df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# Convert numeric-looking columns
#df['declarationDate'] = pd.to_numeric(df['declarationDate'], errors='coerce')
#df['county'] = pd.to_numeric(df['county'], errors='coerce')
df['Declaration Date for Disaster'] = df['declarationDate'].astype('category').cat.codes
df['County'] = df['county'].astype('category').cat.codes
# Encode categorical columns
df['Damaged State Abbreviation'] = df['damagedStateAbbreviation'].astype('category').cat.codes
df['Damaged City Zip Code'] = df['damagedZipCode'].astype('category').cat.codes
df.drop(columns='declarationDate', inplace=True)
df.drop(columns='county', inplace=True)
df.drop(columns='damagedStateAbbreviation', inplace=True)
df.drop(columns='damagedZipCode', inplace=True)
# Compute correlation matrix
correlation_matrix = df.corr(method='pearson')
print(correlation_matrix)

import seaborn as sns
import matplotlib.pyplot as plt

# Set up the matplotlib figure
plt.figure(figsize=(8, 6))

# Draw the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True, cbar_kws={"shrink": 0.75})

# Add title and adjust layout
plt.title("Geographical Correlation")
plt.tight_layout()
plt.show()
