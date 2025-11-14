import pandas as pd

# === Load dataset ===
file_path = r"C:\Users\andre\Downloads\invading privacy\public_data_registerL.xlsx"
df = pd.read_excel(file_path)

# === Remove rows where last_voted == 2 ===
df_cleaned = df[df["last_voted"] != 2]

# === Save cleaned file ===
output_path = r"C:\Users\andre\Downloads\invading privacy\public_data_registerL_cleaned.xlsx"
df_cleaned.to_excel(output_path, index=False)

print(f"✅ Cleaned file saved to: {output_path}")
print(f"🧹 Removed {len(df) - len(df_cleaned)} rows where last_voted == 2.")
