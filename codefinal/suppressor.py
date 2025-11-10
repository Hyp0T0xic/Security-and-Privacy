import pandas as pd
import numpy as np

# === Load anonymized dataset ===
df = pd.read_csv(r"C:\Users\andre\Downloads\Group goopers Dataset F-20251106\generalised_dataF.csv")

# === Define quasi-identifiers ===
quasi_identifiers = ['age_group', 'sex', 'marital_status', 'evote']

# === Identify high-risk (k=1) combinations ===
k_counts = df.groupby(quasi_identifiers).size().reset_index(name='count')
high_risk_combos = k_counts[k_counts['count'] == 1][quasi_identifiers]

print(f"Found {len(high_risk_combos)} high-risk (k=1) combinations to suppress evote for.")

# === Blank evote for those combinations ===
for _, combo in high_risk_combos.iterrows():
    mask = (
        (df['age_group'] == combo['age_group']) &
        (df['sex'] == combo['sex']) &
        (df['marital_status'] == combo['marital_status']) &
        (df['evote'] == combo['evote'])
    )
    df.loc[mask, 'evote'] = np.nan  # blank it

# === Save updated dataset ===
output_path = r"C:\Users\andre\Downloads\Group goopers Dataset F-20251106\suppressed_dataF.csv"
df.to_csv(output_path, index=False)

print(f"Suppression complete. Updated dataset saved to: {output_path}")
