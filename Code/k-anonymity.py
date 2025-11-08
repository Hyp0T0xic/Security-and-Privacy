import pandas as pd
import numpy as np

# Load anonymized dataset
df = pd.read_csv("anonymised_dataX.csv")

# Define quasi-identifiers and sensitive attribute
quasi_identifiers = ['age_group', 'zip', 'sex', 'education']
sensitive_attr = 'party'

# Group by quasi-identifiers
grouped = df.groupby(quasi_identifiers)
k_counts = grouped.size().reset_index(name='count')

# 1. Minimum k-anonymity
k_min = k_counts['count'].min()

# 2. Count of k=1 records
k1_records = k_counts[k_counts['count'] == 1].shape[0]

# 3. Risk distribution
k_distribution = k_counts['count'].value_counts().sort_index()

# 4. Average individual risk
individual_risks = 1 / k_counts['count']
avg_individual_risk = individual_risks.mean()

# 5. l-diversity for sensitive attribute
l_diversity = grouped[sensitive_attr].nunique().reset_index(name='l_diversity')
l_min = l_diversity['l_diversity'].min()
l_violations = l_diversity[l_diversity['l_diversity'] < 2]  # threshold l=2

# Print results
print(f"Minimum k-anonymity: {k_min}")
print(f"Number of unique records (k=1): {k1_records}\n")
print("Distribution of k-values:")
print(k_distribution.to_string())
print(f"\nAverage individual risk: {avg_individual_risk:.4f}\n")
print(f"Minimum l-diversity for 'party': {l_min}")
print(f"Number of equivalence classes failing l-diversity (l < 2): {l_violations.shape[0]}")
print("\nEquivalence classes failing l-diversity:")
print(l_violations.to_string(index=False))