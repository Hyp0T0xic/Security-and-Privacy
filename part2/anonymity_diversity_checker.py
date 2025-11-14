import pandas as pd
import numpy as np

# === Load dataset ===
df = pd.read_excel(r"C:\\Users\\andre\\Downloads\\invading privacy\\anonymised_dataL.xlsx")

# === Define quasi-identifiers and sensitive attribute ===
quasi_identifiers = ['age_a', 'sex', 'maritalstatus_a', 'last_voted', 'citizenship_a', 'zip_a']
#sensitive_attr = 'party'

# === Group by quasi-identifiers ===
grouped = df.groupby(quasi_identifiers)
k_counts = grouped.size().reset_index(name='k')  # "k" = number of records in each equivalence class

# === Compute basic k-anonymity metrics ===
k_min = k_counts['k'].min()
k1_records = k_counts[k_counts['k'] == 1].shape[0]
k_distribution = k_counts['k'].value_counts().sort_index()

# Individual risk per group = 1/k
individual_risks = 1 / k_counts['k']
avg_individual_risk = individual_risks.mean()
'''
# === Compute l-diversity ===
l_diversity = grouped[sensitive_attr].nunique().reset_index(name='l_diversity')

# === Detect "homogeneous" groups (dominant sensitive value ≥ 80%) ===
party_counts = df.groupby(quasi_identifiers + [sensitive_attr]).size().reset_index(name='party_count')
group_totals = df.groupby(quasi_identifiers).size().reset_index(name='k')

# Merge and calculate dominant party ratio
dominant_ratio = pd.merge(party_counts, group_totals, on=quasi_identifiers)
dominant_ratio['party_ratio'] = dominant_ratio['party_count'] / dominant_ratio['k']

# Filter groups where one party dominates ≥ 80%
homogeneous_groups = (
    dominant_ratio[dominant_ratio['party_ratio'] >= 0.8]
    .sort_values('party_ratio', ascending=False)
)
'''
# =========================================================
# === Print summary ===
# =========================================================
print("=== K-Anonymity Overview ===")
print(f"Minimum k-anonymity: {k_min}")
print(f"Number of unique records (k=1): {k1_records}")
print(f"Average individual risk: {avg_individual_risk:.4f}\n")

print("=== Distribution of k-values ===")
print(k_distribution.to_string())

# =========================================================
# === Print high-risk combinations (k=1 to k=8) ===
# =========================================================
for k_val in range(1, 9):
    subset = k_counts[k_counts['k'] == k_val]
    if not subset.empty:
        print(f"\nCombinations with k={k_val}:")
        print(subset.to_string(index=False))

# =========================================================
# === Print homogeneous groups (≥80% same party) ===
# =========================================================
'''
print("\n=== Groups with ≥80% same party ===")
print(f"Total groups: {homogeneous_groups.shape[0]}\n")

if not homogeneous_groups.empty:
    print(
        homogeneous_groups[
            quasi_identifiers + [sensitive_attr, 'party_count', 'k', 'party_ratio']
        ].to_string(index=False)
    )
else:
    print("No homogeneous groups found.")
'''