import pandas as pd
import numpy as np
from datetime import datetime

# Load raw dataset
raw_df = pd.read_excel("Group goopers Dataset F-20251103\private_dataF.xlsx", engine="openpyxl")

# Drop citizenship
if 'citizenship' in raw_df.columns:
    raw_df.drop(columns=['citizenship'], inplace=True)

# Generalize ZIP codes into two regions
def generalize_zip(zip_code):
    zip_str = str(zip_code)
    if zip_str.startswith(('21', '22')):
        return '21-22xx'
    else:
        return '23-24xx'

raw_df['zip'] = raw_df['zip'].apply(generalize_zip)

# Convert DOB to age groups
def dob_to_age_group(dob):
    today = datetime(2025, 11, 7)
    age = (today - pd.to_datetime(dob)).days // 365
    if age <= 30:
        return "18-30"
    elif age <= 45:
        return "31-45"
    elif age <= 60:
        return "46-60"
    else:
        return "61+"

raw_df['age_group'] = raw_df['dob'].apply(dob_to_age_group)
raw_df.drop(columns=['dob'], inplace=True)

# Apply new education grouping
education_map = {
    'Primary education': 'Lower Education',
    'Upper secondary education': 'Lower Education',
    'Vocational Education and Training (VET)': 'Lower Education',
    'Short cycle higher education': 'Mid Education',
    'Vocational bachelors education': 'Mid Education',
    'Bachelors programmes': 'Mid Education',
    'Masters programmes': 'Higher Education',
    'PhD programmes': 'Higher Education',
    'Education': 'Unclassified',
    'Not stated': 'Unclassified'
}
raw_df['education'] = raw_df['education'].map(education_map)

# Suppress marital status for k=1
quasi_identifiers = ['age_group', 'zip', 'sex', 'education']
group_counts = raw_df.groupby(quasi_identifiers).size().reset_index(name='count')
raw_df = raw_df.merge(group_counts, on=quasi_identifiers, how='left')
raw_df.loc[raw_df['count'] == 1, 'marital_status'] = np.nan
raw_df.drop(columns=['count'], inplace=True)

# Save anonymized dataset
raw_df.to_csv("anonymised_dataX.csv", index=False)
print(f"Anonymization complete. Rows retained: {len(raw_df)}")