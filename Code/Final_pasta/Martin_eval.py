# =============================================================================
# Short Evaluation: k-metrics (PUBLIC QIs), l-diversity, χ² + Cramér's V
# =============================================================================
import pandas as pd
import numpy as np
from pathlib import Path

IN_CSV   = "anonymised_dataF_sup2222.csv"   # <-- your anonymised file
OUT_XLSX = "risk_utility_report.xlsx"   # Excel with tables
K_FLOOR  = 3

# Try SciPy; skip χ² if not available
try:
    from scipy.stats import chi2_contingency
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ----------------------- Config: PUBLIC QIs -----------------------
QIS_PUBLIC = ["sex", "age_group", "marital_status", "evote"]  # education EXCLUDED
SENSITIVE  = "party"

# ----------------------- Load ------------------------------------
df = pd.read_csv(IN_CSV) if IN_CSV.lower().endswith(".csv") else pd.read_excel(IN_CSV)
if "evote" in df.columns:
    ev = pd.to_numeric(df["evote"], errors="coerce")
    ev = ev.where(ev.isin([0,1]), np.nan).astype("Int64")
    df["evote"] = ev

# ----------------------- k & l metrics ----------------------------
def k_metrics(data: pd.DataFrame, qi_cols, sensitive=SENSITIVE, k_floor=K_FLOOR):
    grp = data.groupby(qi_cols, dropna=False)
    k_counts = grp.size().reset_index(name="k")
    k_min = int(k_counts["k"].min()) if len(k_counts) else 0
    k1_classes = int((k_counts["k"] == 1).sum())
    k_dist = k_counts["k"].value_counts().sort_index()

    tmp = data.merge(k_counts, on=qi_cols, how="left")
    avg_indiv_risk = float((1.0 / tmp["k"]).mean()) if len(tmp) else float("nan")
    rec_in_small = int((tmp["k"] < k_floor).sum())

    if sensitive in data.columns:
        l_tab = grp[sensitive].nunique(dropna=False).reset_index(name="l")
        l_min = int(l_tab["l"].min()) if len(l_tab) else 0
        l_viol = int((l_tab["l"] < 2).sum())
    else:
        l_tab = pd.DataFrame()
        l_min, l_viol = 0, 0

    risky = k_counts.sort_values("k").head(25)
    metrics = {
        "n_records": int(len(data)),
        "min_k": k_min,
        "k1_classes": k1_classes,
        "avg_indiv_risk": avg_indiv_risk,
        "records_in_k<{}".format(k_floor): rec_in_small,
        "l_min": l_min,
        "l_violations": l_viol,
        "risk_dist_classes": k_dist.to_dict()
    }
    return k_counts, risky, l_tab, metrics

# ----------------------- χ² & Cramér’s V -------------------------
def cramers_v(chi2, n, r, c):
    denom = n * (min(r-1, c-1))
    if denom <= 0: return np.nan
    return float(np.sqrt(chi2 / denom))

def chisq_cramer(data: pd.DataFrame, pairs):
    rows = []
    if not SCIPY_OK: return rows
    for A, B in pairs:
        if A not in data.columns or B not in data.columns: continue
        sub = data[[A,B]].dropna()
        if sub.empty: continue
        tab = pd.crosstab(sub[A], sub[B])
        if tab.shape[0] < 2 or tab.shape[1] < 2: continue
        chi2, p, dof, _ = chi2_contingency(tab)
        n = int(tab.values.sum())
        V = cramers_v(chi2, n, tab.shape[0], tab.shape[1])
        rows.append({"var1":A,"var2":B,"n":n,"rows":tab.shape[0],"cols":tab.shape[1],
                     "chi2":float(chi2),"dof":int(dof),"p_value":float(p),"cramers_v":V})
    return rows

demo_vars = [v for v in ["sex","age_group","marital_status","education"] if v in df.columns]
pairs = []
if "evote" in df.columns:
    for v in demo_vars: pairs.append((v,"evote"))     # (C) channel vs demos
for v in demo_vars: pairs.append((v,"party"))         # (A/B) party vs demos
chi_rows = chisq_cramer(df, pairs)

# ----------------------- Run metrics -----------------------------
kc, risky, ltab, m = k_metrics(df, QIS_PUBLIC, SENSITIVE, K_FLOOR)

# ----------------------- Console summary -------------------------
print("\n=== PUBLIC RISK (education excluded) ===")
print(f"n_records: {m['n_records']}")
print(f"min_k: {m['min_k']}")
print(f"#k=1 classes: {m['k1_classes']}")
print(f"avg individual risk: {m['avg_indiv_risk']:.4f}")
print(f"records in k<{K_FLOOR}: {m['records_in_k<{}'.format(K_FLOOR)]}")
print(f"l_min(party): {m['l_min']} | l_violations: {m['l_violations']}")
print("risk_dist_classes:", m["risk_dist_classes"])

if SCIPY_OK and chi_rows:
    print("\n=== χ² & Cramér’s V (anonymised) ===")
    df_ch = pd.DataFrame(chi_rows).sort_values("cramers_v", ascending=False)
    print(df_ch.to_string(index=False, max_rows=12))
else:
    print("\n[INFO] SciPy not available or no valid χ² tables; skipping χ².")

# ----------------------- Excel export ----------------------------
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
    kc.to_excel(xw, "k_counts_public", index=False)
    risky.to_excel(xw, "risky_public", index=False)
    if len(ltab): ltab.to_excel(xw, "l_diversity_public", index=False)
    pd.DataFrame([m]).to_excel(xw, "metrics_public", index=False)
    if chi_rows:
        pd.DataFrame(chi_rows).to_excel(xw, "chisq_cramer", index=False)
    # quick audit
    pd.DataFrame({"column": sorted(df.columns)}).to_excel(xw, "columns_audit", index=False)

print(f"\n[OK] Wrote Excel report: {Path(OUT_XLSX).resolve()}")
print("[DONE]")