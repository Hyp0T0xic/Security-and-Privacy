import pandas as pd
from datetime import datetime

# === Load dataset ===
file_path = r"C:\Users\andre\Downloads\invading privacy\public_data_registerL_cleaned.xlsx"
df = pd.read_excel(file_path)

# === Convert DOB to Age ===
def dob_to_age(dob):
    try:
        dob = pd.to_datetime(dob, dayfirst=True, errors='coerce')
        if pd.isna(dob):
            return None
        today = datetime.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return age
    except Exception:
        return None

df["age"] = df["dob"].apply(dob_to_age)

# === Save updated file ===
output_path = r"C:\Users\andre\Downloads\invading privacy\public_data_registerL_with_age.xlsx"
df.to_excel(output_path, index=False)

print(f"✅ Added 'age' column based on 'dob'. Saved to: {output_path}")
