import pandas as pd
import numpy as np

# Load original dataset
df = pd.read_excel("Group goopers Dataset F-20251103\private_dataF.xlsx", engine="openpyxl")

# --- GENERALISATION ---

# Convert DOB to age (assuming survey date is July 1, 2025)
survey_date = pd.to_datetime("2025-07-01")
df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
df['age'] = (survey_date - df['dob']).dt.days // 365

# Age groups: fewer but meaningful
age_bins = [0, 30, 60, 100]
age_labels = ['<30', '30-60', '60+']
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

# ZIP regions: merge into two broader regions
zip_map = {2100: '21-22xx', 2200: '21-22xx', 2300: '23-24xx', 2400: '23-24xx'}
df['zip_region'] = df['zip'].map(zip_map)

# Education: group into Low, Medium, High
education_map = {
    'Primary education': 'Low',
    'Upper secondary education': 'Low',
    'Vocational Education and Training (VET)': 'Medium',
    'Short cycle higher education': 'Medium',
    "Vocational bachelors educations": 'Medium',
    "Bachelors programmes": 'High',
    "Masters programmes": 'High',
    "Qualifying educational programmes": 'Medium'
}
df['education_group'] = df['education'].map(education_map)

# Marital status: generalise into Single, Married/Separated, Other
marital_map = {
    'Never married': 'Single',
    'Married/separated': 'Married/Separated',
    'Divorced': 'Other',
    'Widowed': 'Other'
}
df['marital_group'] = df['marital_status'].map(marital_map)

# --- SUPPRESSION ---

# Define quasi-identifiers
quasi_identifiers = ['age_group', 'zip_region', 'sex', 'education_group']

# Identify unique combinations
group_sizes = df.groupby(quasi_identifiers).size()
unique_combos = group_sizes[group_sizes == 1].reset_index()

# Mark rows for suppression
df['suppress'] = df[quasi_identifiers].merge(unique_combos, on=quasi_identifiers, how='left', indicator=True)['_merge'] == 'both'

# Suppress only 'party' (keep evote and marital_group)
df.loc[df['suppress'], 'party'] = np.nan

# Drop intermediate columns
df.drop(columns=['dob', 'age', 'suppress'], inplace=True)

# Save anonymised dataset
df.to_csv("anonymised_dataF_balanced_final.csv", index=False)

# --- DISCLOSURE RISK METRICS ---

# k-anonymity
group_sizes = df.groupby(quasi_identifiers).size()
k_anonymity = group_sizes.min()

# l-diversity for 'party'
l_diversity = df.groupby(quasi_identifiers)['party'].nunique().min()

# Global risk
total_records = len(df)
unique_records = (group_sizes == 1).sum()
global_risk = (unique_records / total_records) * 100

# Print results
print(f"k-anonymity: {k_anonymity}")
print(f"l-diversity: {l_diversity}")
print(f"Global risk: {global_risk:.2f}% of records are unique based on quasi-identifiers")