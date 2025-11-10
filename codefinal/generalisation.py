import pandas as pd
import numpy as np
from datetime import datetime
import random

# === Load raw dataset ===
raw_df = pd.read_excel(r"C:\\Users\\andre\\Downloads\\Group goopers Dataset F-20251106\\private_dataF.xlsx")

# === Drop direct identifiers ===
cols_to_drop = ['citizenship', 'name', 'zip']
raw_df = raw_df.drop(columns=[c for c in cols_to_drop if c in raw_df.columns])

# === Convert DOB to age groups ===
# === Convert DOB to age groups ===
def dob_to_age_group(dob):
    today = datetime(2025, 11, 7)
    try:
        age = (today - pd.to_datetime(dob)).days // 365
    except Exception:
        return np.nan
    if age <= 30:
        return "18-30"
    elif age <= 50:
        return "31-50"
    else:
        return "51+"

raw_df['age_group'] = raw_df['dob'].apply(dob_to_age_group)
raw_df.drop(columns=['dob'], inplace=True, errors='ignore')

# === Simplify education levels ===
education_map = {
    'Primary education': 'Lower Education',
    'Upper secondary education': 'Lower Education',
    'Vocational Education and Training (VET)': 'Higher Education',
    'Short cycle higher education': 'Lower Education',
    'Vocational bachelors educations': 'Higher Education',
    'Bachelors programmes': 'Higher Education',
    'Masters programmes': 'Higher Education',
    'PhD programmes': 'Higher Education',
    'Education': 'Lower Education',
    'Not stated': 'Lower Education'
}
raw_df['education'] = raw_df['education'].map(education_map).fillna('Lower Education')

# === Simplify marital status to married/not married ===
marital_map = {
    'Married': 'Married',
    'Married/separated': 'Married',
    'Never married': 'Not married',
    'Divorced': 'Not married',
    'Widowed': 'Not married'
}
raw_df['marital_status'] = raw_df['marital_status'].map(marital_map).fillna('Not married')

# === Replace 'Invalid vote' with random Red or Green ===
np.random.seed(69)  # For reproducibility
raw_df.loc[raw_df['party'] == 'Invalid vote', 'party'] = np.random.choice(['Red', 'Green'], 
                                                                          size=(raw_df['party'] == 'Invalid vote').sum())

# === Save anonymised dataset ===
output_path = "anonymised_dataF.csv"
raw_df.to_csv(output_path, index=False)

print(f"Anonymisation complete. Rows retained: {len(raw_df)}")
print(f"Saved to: {output_path}")
