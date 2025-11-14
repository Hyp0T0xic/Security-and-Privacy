import pandas as pd

# === Load dataset ===
file_path = r"C:\Users\andre\Downloads\invading privacy\public_data_registerL_marital_fixed.xlsx"
df = pd.read_excel(file_path)

# === Define EU countries ===
eu_countries = {
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
    "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary",
    "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta",
    "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia",
    "Spain", "Sweden"
}

# === Convert citizenship to EU / non-EU ===
def classify_citizenship(country):
    if pd.isna(country):
        return "non-EU"
    c = str(country).strip()
    return "EU" if c in eu_countries else "non-EU"

df["citizenship"] = df["citizenship"].apply(classify_citizenship)

# === Save the updated file ===
output_path = r"C:\Users\andre\Downloads\invading privacy\public_data_registerL_citizenship_fixed.xlsx"
df.to_excel(output_path, index=False)

print(f"✅ Citizenship column classified and saved to: {output_path}")
