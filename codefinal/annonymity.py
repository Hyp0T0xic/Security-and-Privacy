import pandas as pd

# Load anonymized dataset
df = pd.read_csv(f"C:/Users/andre/Downloads/Group goopers Dataset F-20251106/suppressed_dataF.csv")
#Group goopers Dataset F-20251106/anonymised_dataF_v2.csv
# Define quasi-identifiers
quasi_identifiers = ['age_group', 'sex', 'marital_status', 'evote']

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
print("\nMedium-risk combinations (k=2):")
print(k_counts[k_counts['count'] == 2].to_string(index=False))

print("\nMedium-risk combinations (k=3):")
print(k_counts[k_counts['count'] == 3].to_string(index=False))

print("\nMedium-risk combinations (k=4):")
print(k_counts[k_counts['count'] == 4].to_string(index=False))

print("\nMedium-risk combinations (k=5):")
print(k_counts[k_counts['count'] == 5].to_string(index=False))

print("\nMedium-risk combinations (k=6):")
print(k_counts[k_counts['count'] == 6].to_string(index=False))

print("\nMedium-risk combinations (k=7):")
print(k_counts[k_counts['count'] == 7].to_string(index=False))

print("\nMedium-risk combinations (k=8):")
print(k_counts[k_counts['count'] == 8].to_string(index=False))