import pandas as pd
import numpy as np
from datetime import datetime

# Load the raw dataset
raw_df = pd.read_excel("Group goopers Dataset F-20251103\private_dataF.xlsx", engine="openpyxl")

# -----------------------------
# Anonymisation Strategy
# -----------------------------

# 1. Generalize ZIP codes to broader regions (e.g., 21xx, 22xx)
def generalize_zip(zip_code):
    return str(zip_code)[:2] + "xx"
raw_df['zip'] = raw_df['zip'].apply(generalize_zip)

# 2. Convert DOB to age groups
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
raw_df.drop(columns=['dob'], inplace=True)  # Drop original DOB

# 3. Group rare education levels into broader categories
education_map = {
    'PhD': 'Higher Education',
    "Master's": 'Higher Education',
    "Bachelor's": 'Higher Education',
    "Vocational bachelor's": 'Higher Education',
    'Short cycle higher education': 'Mid Education',
    'VET': 'Mid Education',
    'Upper secondary': 'Lower Education',
    'Primary education': 'Lower Education',
    'Not stated': 'Unknown'
}
raw_df['education'] = raw_df['education'].map(education_map)

# 4. Suppress marital status for high-risk records
quasi_identifiers = ['age_group', 'zip', 'sex', 'education']
group_counts = raw_df.groupby(quasi_identifiers).size().reset_index(name='count')
raw_df = raw_df.merge(group_counts, on=quasi_identifiers)

raw_df.loc[raw_df['count'] == 1, 'marital_status'] = np.nan
raw_df.drop(columns=['count'], inplace=True)

# 5. Keep evote intact for utility (no changes needed)

# Save anonymised dataset
raw_df.to_csv("anonymised_dataX.csv", index=False)
print("Anonymisation complete. Output saved to anonymised_dataX.csv")
