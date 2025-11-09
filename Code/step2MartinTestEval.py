# =============================================================================
# Script B: Risk & Utility Metrics -> risk_tables.xlsx, aux_chisq_summary.csv
# =============================================================================
# What it does:
#   - Loads anonymised_dataF.csv
#   - Computes k-anonymity metrics:
#       * k_min, #k=1, risk distribution, avg individual risk (record-weighted)
#   - Computes l-diversity on 'party' (flags classes with l < 2)
#   - Two views:
#       * Release-QI:  sex + age_group + education + marital_status_collapsed
#       * Linkage-risk: same QIs + evote
#   - Exports tables to Excel; optional chi-square (evote vs demographics)
# =============================================================================

import pandas as pd
import numpy as np

IN_CSV = "Code/anonymised_dataF_sup.csv"
OUT_XLSX = "risk_tables.xlsx"
OUT_CHISQ = "aux_chisq_summary.csv"

# ------------------------- Load ------------------------------------------------
df = pd.read_csv(IN_CSV)

# ------------------------- Validate columns -----------------------------------
required = {"sex", "age_group", "education", "party"}
missing = required - set(df.columns)
if missing:
    print(f"[WARN] Missing columns: {missing}. Proceeding with available fields.")

has_evote = "evote" in df.columns
has_mstatc = "marital_status_collapsed" in df.columns

# Define QIs
QIs_release = ["sex", "age_group", "education"] + (["marital_status_collapsed"] if has_mstatc else [])
QIs_linkage = QIs_release + (["evote"] if has_evote else [])

def k_metrics(data: pd.DataFrame, qi_cols, label: str):
    if not qi_cols:
        raise ValueError("No QIs provided.")

    # Equivalence classes
    grp = data.groupby(qi_cols, dropna=False)
    k_counts = grp.size().reset_index(name="k")

    # Core k-metrics
    k_min = int(k_counts["k"].min())
    k1_classes = int((k_counts["k"] == 1).sum())
    k_dist = k_counts["k"].value_counts().sort_index()

    # Record-weighted average individual risk: mean(1/k_row)
    # (merge class sizes back to each row)
    data_tmp = data.merge(k_counts, on=qi_cols, how="left")
    avg_indiv_risk = float((1.0 / data_tmp["k"]).mean())

    # l-diversity on 'party'
    if "party" in data.columns:
        l_table = grp["party"].nunique(dropna=False).reset_index(name="l")
        l_min = int(l_table["l"].min())
        l_viol = l_table[l_table["l"] < 2].copy()
        l_viol_count = int(len(l_viol))
    else:
        l_table = pd.DataFrame()
        l_min = np.nan
        l_viol = pd.DataFrame()
        l_viol_count = 0

    # Small class summaries
    small_classes = int(((k_counts["k"] == 1) | (k_counts["k"] == 2)).sum())
    small_records = int(
        (k_counts.loc[k_counts["k"] == 1, "k"].sum())
        + (k_counts.loc[k_counts["k"] == 2, "k"].sum())
    )

    # Print quick summary
    print(f"\n=== {label} ===")
    print(f"QIs: {qi_cols}")
    print(f"k_min: {k_min} | #k=1 classes: {k1_classes}")
    print("Risk distribution (#classes by k):")
    print(k_dist.to_string())
    print(f"Avg individual risk (record-weighted): {avg_indiv_risk:.4f}")
    if len(l_table):
        print(f"l_min (party): {l_min} | # l-diversity violations (l<2): {l_viol_count}")
    print(f"#classes with k<3: {small_classes} | #records in k<3: {small_records}")

    # Top risky patterns
    risky = k_counts.sort_values("k").head(20)

    return {
        "k_counts": k_counts,
        "k_min": k_min,
        "k1_classes": k1_classes,
        "k_dist": k_dist,
        "avg_indiv_risk": avg_indiv_risk,
        "l_table": l_table,
        "l_viol": l_viol,
        "risky": risky
    }

# ------------------------- Run metrics ----------------------------------------
res_release = k_metrics(df, QIs_release, label="Release-QI (Married vs Not married)")
res_linkage = k_metrics(df, QIs_linkage, label="Linkage-risk (Release-QI + evote)") if has_evote else None

# ------------------------- Optional chi-square checks --------------------------
chisq_rows = []
try:
    from scipy.stats import chi2_contingency
    def chisq_pair(a, b, data):
        tab = pd.crosstab(data[a], data[b])
        chi2, p, dof, _ = chi2_contingency(tab)
        return {"var1": a, "var2": b, "chi2": chi2, "dof": dof, "p": p}

    demo_vars = ["sex", "age_group", "education"]
    if has_mstatc:
        demo_vars.append("marital_status_collapsed")

    if has_evote:
        for v in demo_vars:
            chisq_rows.append(chisq_pair(v, "evote", df))
    # You can add more checks here (e.g., party vs demographics) if needed.

except Exception as e:
    print(f"[INFO] SciPy not available or chi-square failed: {e}")

# ------------------------- Export tables --------------------------------------
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
    # Release view sheets
    res_release["k_counts"].to_excel(xw, sheet_name="k_counts_release", index=False)
    res_release["risky"].to_excel(xw, sheet_name="risky_release", index=False)
    if len(res_release["l_table"]):
        res_release["l_table"].to_excel(xw, sheet_name="l_diversity_release", index=False)
        res_release["l_viol"].to_excel(xw, sheet_name="l_violations_release", index=False)
    # Linkage view sheets
    if res_linkage is not None:
        res_linkage["k_counts"].to_excel(xw, sheet_name="k_counts_linkage", index=False)
        res_linkage["risky"].to_excel(xw, sheet_name="risky_linkage", index=False)

# χ² summary (if any)
if len(chisq_rows):
    pd.DataFrame(chisq_rows).to_csv(OUT_CHISQ, index=False)
    print(f"[OK] Wrote chi-square summary: {OUT_CHISQ}")

print(f"[OK] Wrote risk tables to: {OUT_XLSX}")