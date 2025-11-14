import pandas as pd

# === Load dataset ===
file_path = r"C:\Users\andre\Downloads\invading privacy\public_data_registerL_with_age.xlsx"
df = pd.read_excel(file_path)

# === Generalize ZIP codes (e.g. 2100 → 21xx) ===
def generalize_zip(zip_code):
    try:
        zip_str = str(int(zip_code))
        return zip_str[:2] + "xx"
    except:
        return zip_code  # leave unchanged if invalid or missing

df["zip_a"] = df["zip"].apply(generalize_zip)

# === Move 'zip_a' to 7th column (G column, index 6 since 0-based) ===
cols = list(df.columns)
if "zip_a" in cols:
    cols.insert(6, cols.pop(cols.index("zip_a")))  # move to index 6
    df = df[cols]

# === Save updated file ===
output_path = r"C:\Users\andre\Downloads\invading privacy\public_data_registerL_with_zipA.xlsx"
df.to_excel(output_path, index=False)

print(f"✅ Added 'zip_a' in column G with generalized ZIPs. Saved to: {output_path}")
