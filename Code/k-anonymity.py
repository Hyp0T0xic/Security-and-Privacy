import pandas as pd

# Load anonymized dataset
df = pd.read_csv("anonymised_dataX.csv")

# Define quasi-identifiers
quasi_identifiers = ['age_group', 'zip', 'sex', 'education']

# Group by quasi-identifiers and count
group_counts = df.groupby(quasi_identifiers).size().reset_index(name='count')

# Metrics
k_min = group_counts['count'].min()
unique_records = group_counts[group_counts['count'] == 1].shape[0]
risk_distribution = group_counts['count'].value_counts().sort_index()

print(f"Minimum k-anonymity: {k_min}")
print(f"Number of unique records (k=1): {unique_records}")
print("\nDistribution of k-values:")
print(risk_distribution)