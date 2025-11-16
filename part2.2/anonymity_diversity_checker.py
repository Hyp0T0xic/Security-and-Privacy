import pandas as pd

# Load dataset
df = pd.read_excel(r"C:\Users\Gamer\Downloads\anonymised_dataL.xlsx")


# Quasi-identifiers (use age_a as is)
quasi_identifiers = ['sex', 'maritalstatus_a', 'last_voted', 'citizenship_a', 'age_a']
sensitive_attr = 'party'

# Count total per group
group_totals = df.groupby(quasi_identifiers).size().reset_index(name='k')

# Count per party within group
party_counts = df.groupby(quasi_identifiers + [sensitive_attr]).size().reset_index(name='party_count')

# Merge totals and compute party ratio
dominant_ratio = pd.merge(party_counts, group_totals, on=quasi_identifiers)
dominant_ratio['party_ratio'] = dominant_ratio['party_count'] / dominant_ratio['k']

# Keep groups with ≥80% dominant party
homogeneous_groups = dominant_ratio[dominant_ratio['party_ratio'] >= 0.8]

# Print results
print("\n=== Groups with ≥80% Dominant Party ===")
if homogeneous_groups.empty:
    print("No homogeneous groups found.")
else:
    print(homogeneous_groups[quasi_identifiers + [sensitive_attr, 'party_count', 'k', 'party_ratio']]
          .sort_values('party_ratio', ascending=False)
          .to_string(index=False))
