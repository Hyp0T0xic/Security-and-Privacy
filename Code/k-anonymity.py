import pandas as pd

# Load anonymized dataset
df = pd.read_csv("anonymised_dataX.csv")

# Define quasi-identifiers
quasi_identifiers = ['age_group', 'zip', 'sex', 'education']

# Group and calculate k-anonymity metrics
k_counts = df.groupby(quasi_identifiers).size().reset_index(name='count')
k_min = k_counts['count'].min()
k1_records = k_counts[k_counts['count'] == 1].shape[0]
k_distribution = k_counts['count'].value_counts().sort_index()

print(f"Minimum k-anonymity: {k_min}")
print(f"Number of unique records (k=1): {k1_records}\n")
print("Distribution of k-values:")
print(k_distribution.to_string())

# Optional: Print high-risk combinations
print("\nHigh-risk combinations (k=1):")
print(k_counts[k_counts['count'] == 1].to_string(index=False))