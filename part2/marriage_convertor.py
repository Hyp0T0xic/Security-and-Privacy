import pandas as pd

# === Load dataset ===
file_path = r"C:\Users\andre\Downloads\invading privacy\public_data_registerL_with_zipA.xlsx"
df = pd.read_excel(file_path)

# === Simplify marital status ===
def simplify_marital(status):
    if pd.isna(status):
        return "Single"
    s = str(status).strip().lower()
    if s in ["married/separated", "never married"]:
        return "Married"
    elif s in ["widowed", "divorced"]:
        return "Single"
    else:
        return "Single"  # default fallback

df["marital_status"] = df["marital_status"].apply(simplify_marital)

# === Save updated file ===
output_path = r"C:\Users\andre\Downloads\invading privacy\public_data_registerL_marital_fixed.xlsx"
df.to_excel(output_path, index=False)

print(f"✅ Marital statuses converted and saved to: {output_path}")
