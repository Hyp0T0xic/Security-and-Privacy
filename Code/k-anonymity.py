import pandas as pd

# Load anonymized dataset
df = pd.read_csv("anonymised_dataX.csv")

# Define quasi-identifiers
quasi_identifiers = ['age_group', 'zip', 'sex', 'education']

# Group by quasi-identifiers and count
group_counts = df.groupby(quasi_identifiers).size().reset_index(name='count')

# Filter high-risk records (k = 1)
high_risk = group_counts[group_counts['count'] == 1]

print(f"Minimum k-anonymity: {group_counts['count'].min()}")
print(f"Number of unique records (k=1): {high_risk.shape[0]}")
print("\nHigh-risk combinations (k=1):")
print(high_risk.to_string(index=False))
