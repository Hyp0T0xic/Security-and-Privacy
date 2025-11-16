import pandas as pd
import os

# === Load datasets ===
anonymised_path = r"C:\\Users\\Gamer\\Downloads\\anonymised_dataL.csv"
public_data_path = r"C:\\Users\\Gamer\\Downloads\\public_data_registerL_citizenship_fixed.xlsx"

anon_df = pd.read_csv(anonymised_path)
pub_df = pd.read_excel(public_data_path)

quasi_identifiers = ["sex", "last_voted", "citizenship_a", "maritalstatus_a", "zip_a"]

# === Check column consistency ===
print("\n--- Checking Columns ---")
print("Anonymised columns:", anon_df.columns.tolist())
print("Public columns:", pub_df.columns.tolist())

# === Step 1: Find pure groups in anonymised data ===
pure_groups = (
    anon_df.groupby(quasi_identifiers)
    .agg(
        party_count=("party", lambda x: x.nunique()),
        unique_party=("party", lambda x: list(x.unique())[0] if x.nunique()==1 else None),
        k=("party", "count")
    )
    .reset_index()
)

print("\n--- Pure Groups Found in Anonymised Data ---")
print("Total QI groups:", len(anon_df.groupby(quasi_identifiers)))
print("Pure groups (unique party):", len(pure_groups[pure_groups['party_count'] == 1]))

print(pure_groups[pure_groups["party_count"] == 1])

# Keep only pure groups
pure_groups = pure_groups[pure_groups["party_count"] == 1]

# === Step 2: Group public dataset ===
pub_grouped = (
    pub_df.groupby(quasi_identifiers)["name"]
    .apply(list)
    .reset_index()
)

print("\n--- Public Groups ---")
print(pub_grouped)

# === Step 3: Merge ===
merged = pure_groups.merge(pub_grouped, on=quasi_identifiers, how="left")

print("\n--- After Merging Pure Groups with Public Data ---")
print(merged)

# Drop groups with no public matches
merged = merged.dropna(subset=["name"])

print("\n--- Final Groups with Public Matches ---")
print(merged)

# === Final output ===
if merged.empty:
    print("\nNO MATCHING PURE GROUPS FOUND.")
else:
    for _, row in merged.iterrows():
        print("\n=== Pure Group (Unique Party) ===")
        for col in quasi_identifiers:
            print(f"{col}: {row[col]}")
        print(f"Unique party in anonymised data: {row['unique_party']}")
        print(f"k (anonymised count): {row['k']}")
        print("Possible public matches:")
        for n in row["name"]:
            print(" -", n)
