import pandas as pd

# Load anonymised dataset
df = pd.read_csv("anonymised_dataF_balanced_final.csv")

# Define quasi-identifiers
quasi_identifiers = ['age_group', 'zip_region', 'sex', 'education']

# --- k-Anonymity ---
group_sizes = df.groupby(quasi_identifiers).size()
k_anonymity = group_sizes.min()
print(f"k-anonymity: {k_anonymity}")

# --- l-Diversity for 'party' ---
l_diversity = df.groupby(quasi_identifiers)['party'].nunique().min()
print(f"l-diversity: {l_diversity}")

# --- Global Risk ---
total_records = len(df)
unique_records = (group_sizes == 1).sum()
global_risk = (unique_records / total_records) * 100
print(f"Global risk: {global_risk:.2f}% of records are unique based on quasi-identifiers")