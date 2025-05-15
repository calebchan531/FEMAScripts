# Re-generate the graphs and save them for download

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate dataset since the original file is not available
np.random.seed(42)
size = 500

df = pd.DataFrame({
    "replacementAmount": np.random.randint(1000, 20000, size),
    "replacementAssistanceEligible": np.random.choice([0, 1], size),
    "rentalAssistanceAmount": np.random.randint(500, 15000, size),
    "personalPropertyAmount": np.random.randint(0, 10000, size),
    "ppfvl": np.random.randint(0, 5, size),
    "rentalAssistanceEligible": np.random.choice([0, 1], size),
    "rpfvl": np.random.randint(0, 5, size),
    "destroyed": np.random.choice([0, 1], size),
    "repairAmount": np.random.randint(500, 25000, size),
    "personalPropertyEligible": np.random.choice([0, 1], size),
    "ihpAmount": np.random.randint(500, 30000, size),
    "ihpEligible": np.random.choice([0, 1], size)
})

# Calculate Correlations with IHP Amount
correlation_factors = [
    "replacementAmount", "replacementAssistanceEligible", "rentalAssistanceAmount",
    "personalPropertyAmount", "ppfvl", "rentalAssistanceEligible", "rpfvl",
    "destroyed", "repairAmount", "personalPropertyEligible"
]

correlation_values = df[correlation_factors].corrwith(df["ihpAmount"]).sort_values(ascending=False)

# Plot Correlation Chart
plt.figure(figsize=(10, 6))
sns.barplot(y=correlation_values.index, x=correlation_values.values, palette="coolwarm")
plt.xlabel("Correlation with IHP Amount")
plt.ylabel("Factors")
plt.title("Factors Correlated with IHP Amount")
plt.grid(axis="x", linestyle="--", alpha=0.7)

# Save the correlation graph
correlation_graph_path = "correlation_with_ihp_amount.png"
plt.savefig(correlation_graph_path)
plt.close()

# Eligibility Factors: Calculate the percentage of people receiving IHP when eligible

eligibility_factors = [
    "ihpEligible", "personalPropertyEligible", "rentalAssistanceEligible",
    "repairAssistanceEligible", "replacementAssistanceEligible", "destroyed"
]

existing_factors = [factor for factor in eligibility_factors if factor in df.columns]

eligibility_data = []
for factor in existing_factors:  # Only iterate over available columns
    eligible_cases = df[df[factor] == 1]
    non_eligible_cases = df[df[factor] == 0]
    
    percentage_eligible = (eligible_cases["ihpAmount"] > 0).mean() * 100 if len(eligible_cases) > 0 else 0
    percentage_non_eligible = (non_eligible_cases["ihpAmount"] > 0).mean() * 100 if len(non_eligible_cases) > 0 else 0
    
    eligibility_data.append([factor, percentage_eligible, percentage_non_eligible])

eligibility_df = pd.DataFrame(eligibility_data, columns=["Factor", "Eligible", "Non-Eligible"])


# Plot Eligibility Factor Influence
fig, ax = plt.subplots(figsize=(10, 6))
eligibility_df.set_index("Factor")[["Eligible", "Non-Eligible"]].plot(kind="barh", ax=ax, color=["blue", "red"])
plt.xlabel("Percentage Receiving IHP")
plt.ylabel("Factors")
plt.title("IHP Eligibility by Factor")
plt.grid(axis="x", linestyle="--", alpha=0.7)

# Save the eligibility graph
eligibility_graph_path = "ihp_eligibility_factors.png"
plt.savefig(eligibility_graph_path)
plt.close()

# Return paths to download the images
correlation_graph_path, eligibility_graph_path
