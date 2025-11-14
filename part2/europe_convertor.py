import pandas as pd

# === Load dataset ===
file_path = r"C:\Users\andre\Downloads\invading privacy\public_data_registerL_marital_fixed.xlsx"
df = pd.read_excel(file_path)

# === Define European countries ===
european_countries = {
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
    "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary",
    "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta",
    "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia",
    "Spain", "Sweden", "Norway", "Switzerland", "United Kingdom",
    "Bosnia and Herzegovina", "Iceland", "Russia", "Turkey", "Ukraine"
}

# === Convert citizenship to Europe / Non-Europe ===
def classify_region(country):
    if pd.isna(country):
        return "EU"
    c = str(country).strip()
    return "EU" if c in european_countries else "non EU"

df["citizenship"] = df["citizenship"].apply(classify_region)

# === Save the updated file ===
output_path = r"C:\Users\andre\Downloads\invading privacy\public_data_registerL_region_fixed.xlsx"
df.to_excel(output_path, index=False)

print(f"✅ Citizenship column classified as Europe/Non-Europe and saved to: {output_path}")
