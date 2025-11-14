import pandas as pd
import numpy as np
from datetime import datetime

# === Load dataset ===
raw_df = pd.read_excel(r"C:\Users\andre\Downloads\invading privacy\public_data_registerL.xlsx")

# =========================================================
# 0. CONFIG
# =========================================================

EU_COUNTRIES = [
    "Austria",
    "Belgium",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czech Republic",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Ireland",
    "Italy",
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Netherlands",
    "Poland",
    "Portugal",
    "Romania",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden"
]


np.random.seed(42)  # reproducibility

# =========================================================
# 1. AGE ANONYMISATION (DOB → noisy grouped age)
# =========================================================

def dob_to_age_group(dob):
    today = datetime(2025, 11, 7)
    try:
        age = (today - pd.to_datetime(dob)).days // 365
    except Exception:
        return np.nan
    if age < 30:
        return "<30"
    elif age < 50:
        return "30-49"
    elif age < 65:
        return "50-64"
    else:
        return "65+"

raw_df['age_a'] = raw_df['dob'].apply(dob_to_age_group)
raw_df.drop(columns=['dob'], inplace=True, errors='ignore')


# =========================================================
# 2. CITIZENSHIP (EU / non EU)
# =========================================================

def simplify_citizenship(c):
    if pd.isna(c):
        return "Other"
    c = str(c).strip()
    if c in EU_COUNTRIES:
        return "EU"
    else:
        return "non EU"

raw_df["citizenship_a"] = raw_df["citizenship"].apply(simplify_citizenship)
raw_df.drop(columns=["citizenship"], inplace=True, errors="ignore")

# =========================================================
# 3. MARITAL STATUS (Married / Single)
# =========================================================

def simplify_marital(m):
    if pd.isna(m):
        return "Single"
    m = str(m).lower()
    if "married" in m:
        return "Married"
    else:
        return "Single"

raw_df["maritalstatus_a"] = raw_df["marital_status"].apply(simplify_marital)
raw_df.drop(columns=["marital_status"], inplace=True, errors="ignore")

# =========================================================
# 4. ZIP CODE (Generalisation + Suppression)
# =========================================================

def generalize_zip(zip_code):
    try:
        zip_str = str(int(zip_code))
        return zip_str[:2] + "xx"
    except:
        return "*"

raw_df["zip_a"] = raw_df["zip"].apply(generalize_zip)
raw_df.drop(columns=["zip"], inplace=True, errors="ignore")

# =========================================================
# 5. PARTY + EVOTE (Preserve)
# =========================================================


# =========================================================
# 7. SUPPRESSION: Replace ZIPs for uniques (k=1)
# =========================================================

# Define quasi-identifiers for uniqueness detection
quasi_identifiers = ["sex", "age_a", "citizenship_a", "maritalstatus_a"]

# Compute equivalence class sizes
eq_sizes = raw_df.groupby(quasi_identifiers).transform("size")

# Suppress ZIPs for unique combinations (k=1)
raw_df.loc[eq_sizes == 1, "zip_a"] = "*"

# =========================================================
# 8. FINALISE + SAVE
# =========================================================

cols = ["sex", "last_voted", "age_a", "citizenship_a", "maritalstatus_a", "zip_a", "name"]
raw_df = raw_df[cols]

output_path = r"C:\Users\andre\Downloads\invading privacy/anonymised_dataL_predicted.xlsx"
raw_df.to_excel(output_path, index=False)

print(f"✅ Anonymisation complete. Rows retained: {len(raw_df)}")
print(f"📁 Saved to: {output_path}")
