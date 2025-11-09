import pandas as pd
import numpy as np
from datetime import datetime

INPUT = "private_dataF.xlsx"
OUTPUT = "anonymised_dataF_mitigated.csv"
SURVEY_DATE = datetime(2025, 11, 7)
K = 3
RANDOM_SEED = 69
np.random.seed(RANDOM_SEED)

# --- load and basic cleaning (same as your outline) ---
df = pd.read_excel(INPUT)
for c in ['citizenship','name','zip']:
    if c in df.columns:
        df.drop(columns=c, inplace=True, errors=True)

def dob_to_age_group(dob):
    dob_parsed = pd.to_datetime(dob, errors='coerce')
    if pd.isnull(dob_parsed):
        return np.nan
    age = int((SURVEY_DATE - dob_parsed).days // 365.25)
    if age <= 30:
        return "18-30"
    elif age < 50:
        return "31-50"
    else:
        return "51+"

df['age_group'] = df.get('dob').apply(dob_to_age_group)
df.drop(columns=['dob'], inplace=True, errors=True)

# education and marital maps (as before)
education_map = {
    "Primary education": "Lower education",
    "Upper secondary education": "Lower education",
    "Vocational Education and Training (VET)": "Lower education",
    "Short cycle higher education": "Lower education",
    "Vocational bachelors educations": "Lower education",
    "Bachelors programmes": "Higher education",
    "Masters programmes": "Higher education",
    "PhD programmes": "Higher education",
    "Not stated": "Lower education",
    "Education": "Lower education"
}
df['education'] = df['education'].map(education_map).fillna("Lower education")

marital_map = {
    "Married": "Married",
    "Married/separated": "Married",
    "Separated": "Married",
    "Never married": "Not married",
    "Divorced": "Not married",
    "Widowed": "Not married"
}
df['marital_status'] = df['marital_status'].map(marital_map).fillna("Not married")

# fix invalid votes (keep your seed)
invalid_mask = df['party'] == 'Invalid vote'
df.loc[invalid_mask, 'party'] = np.random.choice(['Red','Green'], size=invalid_mask.sum())

# ---------------- Risk helper ----------------
def risk_metrics(df, qis, k=K):
    # compute group sizes for given QIs (treat NaN as its own value)
    freq = df.groupby(qis, dropna=False).size().reset_index(name='count')
    merged = df.merge(freq, on=qis, how='left')
    total = len(df)
    unique = (merged['count'] == 1).sum()
    small = (merged['count'] < k).sum()
    avg_risk = (1 / merged['count']).mean()
    return {
        'total': total,
        'unique': int(unique),
        'unique_pct': unique/total*100,
        'small': int(small),
        'small_pct': small/total*100,
        'avg_risk_pct': avg_risk*100,
        'freq_table': freq.sort_values('count')
    }

# ---------------- Baseline risk (including evote as QI) ----------------
qis_attack = ['sex','age_group','marital_status','education','evote']
baseline = risk_metrics(df, qis_attack, k=K)
print("BASELINE (attack includes evote):")
print(f" Total: {baseline['total']}; Unique: {baseline['unique']} ({baseline['unique_pct']:.2f}%)")
print(f" <{K}: {baseline['small']} ({baseline['small_pct']:.2f}%) ; Avg risk: {baseline['avg_risk_pct']:.2f}%")

# ---------------- Suppress evote for small cells (correct order) ----------------
# compute counts per full QI including evote
freq = df.groupby(qis_attack, dropna=False).size().reset_index(name='count')
df = df.merge(freq, on=qis_attack, how='left')

# suppress evote where count < K
mask_small = df['count'] < K
n_suppressed = mask_small.sum()
df.loc[mask_small, 'evote'] = np.nan
df.drop(columns=['count'], inplace=True, errors=True)

print(f"Suppressed evote for {n_suppressed} records (groups with size < {K}).")
# ---------------- Recompute risk AFTER suppression ----------------
# 1) attack scenario: attacker still uses evote as QI (but evote now NaN for suppressed rows)
after_attack = risk_metrics(df, qis_attack, k=K)
print("\nAFTER SUPPRESSION (attack includes evote):")
print(f" Unique: {after_attack['unique']} ({after_attack['unique_pct']:.2f}%) ; <{K}: {after_attack['small']} ({after_attack['small_pct']:.2f}%) ; Avg risk: {after_attack['avg_risk_pct']:.2f}%")

# 2) agency release scenario: QIs the agency will publish (evote excluded where NaN) --
#    here evaluate QIs without evote to show protection level for released microdata
qis_release = ['sex','age_group','marital_status','education']
release_risk = risk_metrics(df, qis_release, k=K)
print("\nAFTER SUPPRESSION (agency release QIs, excluding evote):")
print(f" Unique: {release_risk['unique']} ({release_risk['unique_pct']:.2f}%) ; <{K}: {release_risk['small']} ({release_risk['small_pct']:.2f}%) ; Avg risk: {release_risk['avg_risk_pct']:.2f}%")

# ---------------- If still high: show quick automatic coarsening step suggestion ---------------
if release_risk['small_pct'] > 10 or release_risk['avg_risk_pct'] > 15:
    print("\nNOTE: risk is still high. Consider one or more of:")
    print(" - Coarsen age (e.g. 18-49 / 50+)")
    print(" - Collapse education to two levels (Lower/Higher) or single level")
    print(" - Increase k to 8 or 10 for suppression")
    print(" - Apply small swap (3-5%) of evote values within large groups")
    # (you can automate one step below if you want)



# ---------------- save final file ----------------
df.to_csv(OUTPUT, index=False)
print(f"\nSaved output to {OUTPUT}")
