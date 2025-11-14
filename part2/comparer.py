import pandas as pd

# === Load predicted and actual anonymised datasets ===
predicted_path = r"C:\Users\andre\Downloads\invading privacy\anonymised_dataL_predicted.csv"
actual_path = r"C:\Users\andre\Downloads\invading privacy\anonymised_dataL.csv"

predicted_df = pd.read_csv(predicted_path)
actual_df = pd.read_csv(actual_path)

# === Define the columns to match ===
quasi_identifiers = ["sex", "evote", "party", "age_a", "edu_a", "citizenship_a", "maritalstatus_a", "zip_a"]

# === Check which predicted rows exist in the actual anonymised dataset ===
predicted_df["match"] = predicted_df.apply(
    lambda row: ((actual_df[quasi_identifiers] == row[quasi_identifiers]).all(axis=1)).any(),
    axis=1
)

# === Summary ===
num_matches = predicted_df["match"].sum()
num_total = len(predicted_df)
print(f"✅ {num_matches}/{num_total} predicted rows match actual anonymised data ({num_matches/num_total:.2%})")

# === Optional: Save matched rows ===
predicted_df.to_csv(r"C:\Users\andre\Downloads\invading privacy\predicted_with_matches.csv", index=False)
