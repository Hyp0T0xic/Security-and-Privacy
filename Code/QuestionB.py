import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# Load your data
df = pd.read_excel("Group goopers Dataset F-20251103\private_dataF.xlsx")  # or .csv

# Clean and prepare
df["party"] = df["party"].str.strip().str.title()  # Normalize party labels
df["age"] = pd.to_datetime("2025-07-01") - pd.to_datetime(df["dob"], dayfirst=True)
df["age"] = df["age"].dt.days // 365
df["age_bin"] = pd.cut(df["age"], bins=[18, 30, 45, 65, 120], labels=["18–29", "30–44", "45–64", "65+"])

# Attributes to test
attributes = ["sex", "zip", "education", "marital_status", "citizenship", "age_bin"]

for attr in attributes:
    # Drop missing
    sub = df[[attr, "party"]].dropna()
    
    # Contingency table
    table = pd.crosstab(sub[attr], sub["party"])
    print(f"\n{attr} × party\n", table)
    
    # Chi-square test
    chi2, p, dof, expected = chi2_contingency(table)
    print(f"Chi² = {chi2:.2f}, p = {p:.4f}, dof = {dof}")
    
    # Plot
    table_pct = table.div(table.sum(axis=1), axis=0)
    table_pct.plot(kind="bar", stacked=True)
    plt.title(f"{attr} × Party")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.show()