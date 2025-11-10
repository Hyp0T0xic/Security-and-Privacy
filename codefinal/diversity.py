import pandas as pd
import numpy as np

# === Load anonymized dataset ===
df = pd.read_csv(r"C:/Users/andre/Downloads/Group goopers Dataset F-20251106/suppressed_dataF.csv")

# === Define quasi-identifiers and sensitive attribute ===
quasi_identifiers = ['age_group', 'sex', 'marital_status', 'evote']
sensitive_attr = 'party'

# === Group by quasi-identifiers ===
grouped = df.groupby(quasi_identifiers)
k_counts = grouped.size().reset_index(name='k')  # renamed to 'k' for clarity

# === Calculate basic k-anonymity metrics ===
k_min = k_counts['k'].min()
k1_records = k_counts[k_counts['k'] == 1].shape[0]
k_distribution = k_counts['k'].value_counts().sort_index()
individual_risks = 1 / k_counts['k']
avg_individual_risk = individual_risks.mean()

# === Compute l-diversity ===
l_diversity = grouped[sensitive_attr].nunique().reset_index(name='l_diversity')

# === Detect "homogeneous" groups (dominant party ≥ 80%) ===
party_counts = df.groupby(quasi_identifiers + [sensitive_attr]).size().reset_index(name='party_count')
group_totals = df.groupby(quasi_identifiers).size().reset_index(name='k')

# Compute ratio within each group
dominant_ratio = pd.merge(party_counts, group_totals, on=quasi_identifiers)
dominant_ratio['party_ratio'] = dominant_ratio['party_count'] / dominant_ratio['k']

# Keep only groups where the dominant party ≥ 80%
homogeneous_groups = (
    dominant_ratio[dominant_ratio['party_ratio'] >= 0.8]
    .sort_values('party_ratio', ascending=False)
)

# === Print summary ===
print(f"Minimum k-anonymity: {k_min}")
print(f"Number of unique records (k=1): {k1_records}")
print(f"Average individual risk: {avg_individual_risk:.4f}\n")

print("=== Groups with ≥80% same party ===")
print(f"Total groups: {homogeneous_groups.shape[0]}\n")

print(
    homogeneous_groups[
        quasi_identifiers + [sensitive_attr, 'party_count', 'k', 'party_ratio']
    ].to_string(index=False)
)
