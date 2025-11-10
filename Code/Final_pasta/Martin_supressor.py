import pandas as pd
import numpy as np
from datetime import datetime

# ---------------- Config ----------------
INPUT = "Group goopers Dataset F-20251103\private_dataF.xlsx"         # <-- set your private survey path
OUTPUT = "anonymised_dataF_sup2222.csv"  # <-- output CSV (evote + education published)
SURVEY_DATE = datetime(2025, 11, 7)
K = 3
RANDOM_SEED = 69
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)

# PUBLIC QIs for k-anonymity guarantee (education EXCLUDED)
PUB  = ['sex', 'age_group', 'marital_status', 'evote']
BASE = ['sex', 'age_group', 'marital_status']

# ---------------- Load & basic cleaning ----------------
df = pd.read_excel(INPUT)

# Drop direct identifiers (ZIP not published nor used)
for c in ['name', 'citizenship', 'zip']:
    if c in df.columns:
        df.drop(columns=c, inplace=True, errors='ignore')

# ------------- Age grouping from dob (3 fixed bands) -----
def dob_to_age_group(dob):
    dob_parsed = pd.to_datetime(dob, errors='coerce', dayfirst=True)
    if pd.isnull(dob_parsed):
        return np.nan
    age = int((SURVEY_DATE - dob_parsed).days // 365.25)
    return "18-30" if age <= 30 else "31-50" if age <= 50 else "51+"

if 'dob' in df.columns:
    df['age_group'] = df['dob'].apply(dob_to_age_group)
    df.drop(columns=['dob'], inplace=True, errors='ignore')
elif 'age_group' not in df.columns:
    raise ValueError("Neither 'dob' nor 'age_group' present.")

# ------------- Collapse education & marital --------------
education_map = {
    "Primary education": "Lower education",
    "Upper secondary education": "Lower education",
    "Vocational Education and Training (VET)": "Lower education",
    "Short cycle higher education": "Higher education",
    "Vocational bachelors educations": "Higher education",
    "Bachelors programmes": "Higher education",
    "Masters programmes": "Higher education",
    "PhD programmes": "Higher education",
    "Not stated": "Lower education",
    "Education": "Lower education"
}
df['education'] = df.get('education', "Lower education")
df['education'] = df['education'].map(education_map).fillna("Lower education")

marital_map = {
    "Married": "Married",
    "Married/separated": "Married",
    "Separated": "Married",
    "Never married": "Not married",
    "Divorced": "Not married",
    "Widowed": "Not married"
}
df['marital_status'] = df.get('marital_status', "Not married")
df['marital_status'] = df['marital_status'].map(marital_map).fillna("Not married")

# Sensitive attribute cleanup
if 'party' not in df.columns:
    raise ValueError("Missing 'party' in input data.")
# Optional: keep 'Invalid vote' as missing so we don't invent signal
df.loc[df['party'].astype(str).str.strip().eq('Invalid vote'), 'party'] = np.nan

# Coerce evote to {0,1,NaN}
if 'evote' not in df.columns:
    raise ValueError("Missing 'evote' in input data.")
df['evote'] = pd.to_numeric(df['evote'], errors='coerce')
df.loc[~df['evote'].isin([0, 1]), 'evote'] = np.nan
df['evote'] = df['evote'].astype('Int64')

# ---------------- Initial k-suppression on evote ----------
# (Only PUBLIC QIs used; education excluded)
freq = df.groupby(PUB, dropna=False).size().reset_index(name='count')
df = df.merge(freq, on=PUB, how='left')
mask_small = df['count'] < K
df.loc[mask_small, 'evote'] = pd.NA
init_suppressed = int(mask_small.sum())
df.drop(columns=['count'], inplace=True, errors=True)
print(f"Suppressed evote for {init_suppressed} records (groups with size < {K}).")

# ---------------- PRAM (small noise to QIs) ---------------
def pram_2cat(series, cats, p):
    """Flip between two categories with probability p (keeps NaN)."""
    s = series.copy()
    m = s.notna()
    s = s.astype(object)
    idx = s[m].index
    u = rng.random(len(idx))
    for i, ix in enumerate(idx):
        val = s.at[ix]
        if val not in cats:
            continue
        if u[i] < p:
            s.at[ix] = cats[1] if val == cats[0] else cats[0]
    return s

def pram_age(series, p):
    """Adjacent PRAM for ordered age bands (3 fixed bands)."""
    order = ['18-30', '31-50', '51+']
    s = series.copy().astype(object)
    m = s.notna()
    idx = s[m].index
    u = rng.random(len(idx))
    for i, ix in enumerate(idx):
        val = s.at[ix]
        if val not in order:
            continue
        r = u[i]
        if val == '18-30':
            if r < p: s.at[ix] = '31-50'
        elif val == '31-50':
            if r < p/2: s.at[ix] = '18-30'
            elif r < p: s.at[ix] = '51+'
        elif val == '51+':
            if r < p: s.at[ix] = '31-50'
    return s

# Apply gentle PRAM (education is published but NOT used in k calcs)
df['sex'] = pram_2cat(df['sex'], ['Female', 'Male'], p=0.01)
df['age_group'] = pram_age(df['age_group'], p=0.02)  # 3 bands unchanged
df['education'] = pram_2cat(df['education'], ['Lower education', 'Higher education'], p=0.03)
df['marital_status'] = pram_2cat(df['marital_status'], ['Married', 'Not married'], p=0.01)
ev_mask = df['evote'].notna()
df.loc[ev_mask, 'evote'] = pram_2cat(df.loc[ev_mask, 'evote'], [0, 1], p=0.03)

# -------- Enforce k≥K on PUBLIC QIs (no age change) -------
def enforce_k_public(df_in, K, max_rounds=4):
    """
    Ensure k≥K on PUBLIC QIs by:
      1) Moving any small PUBLIC class (BASE+evote) into (evote=NaN) within same BASE cluster
      2) Topping up that NaN bucket to K by suppressing a few more evote within same BASE cluster
      3) If still failing, locally coarsen marital_status to 'Any' (age bands unchanged)
      4) As last resort, drop failing PUBLIC classes
    """
    dfw = df_in.copy()

    for _ in range(max_rounds):
        freq_pub = dfw.groupby(PUB, dropna=False).size().reset_index(name='k_pub')
        small_pub = freq_pub[freq_pub['k_pub'] < K]
        if small_pub.empty:
            return dfw, []

        # Step 1: Move small 0/1 classes to NaN & top up NaN bucket to K
        for _, row in small_pub.iterrows():
            sx, ag, ms, ev = row['sex'], row['age_group'], row['marital_status'], row['evote']
            base_mask = (dfw['sex'].eq(sx)) & (dfw['age_group'].eq(ag)) & (dfw['marital_status'].eq(ms))

            if pd.notna(ev):
                dfw.loc[base_mask & dfw['evote'].eq(ev), 'evote'] = pd.NA

            cluster_idx = dfw.index[base_mask]
            nan_count = dfw.loc[cluster_idx, 'evote'].isna().sum()
            need = K - nan_count
            if need > 0:
                cand_idx = dfw.index[base_mask & dfw['evote'].notna()]
                if len(cand_idx):
                    to_sup = list(cand_idx[:need])
                    dfw.loc[to_sup, 'evote'] = pd.NA

        # Step 2: If still failing, coarsen marital to 'Any' in those BASE clusters (age unchanged)
        freq_pub2 = dfw.groupby(PUB, dropna=False).size().reset_index(name='k_after')
        still_small = freq_pub2[freq_pub2['k_after'] < K]
        if still_small.empty:
            return dfw, []

        changed = False
        for _, row in still_small.iterrows():
            sx, ag, ms, ev = row['sex'], row['age_group'], row['marital_status'], row['evote']
            base_mask = (dfw['sex'].eq(sx)) & (dfw['age_group'].eq(ag)) & (dfw['marital_status'].eq(ms))
            if ms != 'Any':
                dfw.loc[base_mask, 'marital_status'] = 'Any'
                changed = True

        if not changed:
            break  # nothing else we can do in this loop

    # Step 3: Last resort—drop failing PUBLIC classes
    freq_final = dfw.groupby(PUB, dropna=False).size().reset_index(name='k_pub')
    failing = freq_final[freq_final['k_pub'] < K]
    drop_index = pd.Index([])
    if not failing.empty:
        for _, row in failing.iterrows():
            m = (dfw['sex'].eq(row['sex']) &
                 dfw['age_group'].eq(row['age_group']) &
                 dfw['marital_status'].eq(row['marital_status']) &
                 ((dfw['evote'].isna() & pd.isna(row['evote'])) | (dfw['evote'] == row['evote'])))
            drop_index = drop_index.union(dfw.index[m])
        dfw = dfw.drop(index=drop_index)

    return dfw, list(drop_index)

df, dropped = enforce_k_public(df, K=K, max_rounds=4)
if dropped:
    print(f"[INFO] Dropped {len(dropped)} record(s) to satisfy k≥{K} on PUBLIC QIs (age bands unchanged).")

# --------- REPAIR l-diversity on 'party' without suppression ---------
# Strategy:
# 1) Pairwise SWAPS between RED-only and GREEN-only PUBLIC classes (preserve marginals)
# 2) Minimal FLIPS for any remaining single-party classes

def compute_party_counts_by_pub(data):
    """Return counts of party within each PUBLIC class (exclude NaN)."""
    vc = data.groupby(PUB)['party'].value_counts(dropna=True).unstack(fill_value=0)
    # Ensure both columns exist
    for p in ['Red','Green']:
        if p not in vc.columns: vc[p] = 0
    vc = vc[['Red','Green']]
    return vc

def sample_index_in_class(data, key, party_value=None):
    """Pick a row index from a PUBLIC class; optionally restricted to a party value."""
    m = (data['sex'].eq(key[0]) &
         data['age_group'].eq(key[1]) &
         data['marital_status'].eq(key[2]) &
         ((data['evote'].isna() & pd.isna(key[3])) | (data['evote'] == key[3])))
    if party_value is not None:
        m = m & (data['party'] == party_value)
    idx = data.index[m]
    if len(idx) == 0: return None
    return rng.choice(idx)

def repair_party_l_diversity(df_in, max_swaps=1000, max_flips=1000):
    dfw = df_in.copy()

    # 1) Swaps between RED-only and GREEN-only classes
    vc = compute_party_counts_by_pub(dfw)
    red_only = vc[(vc['Red'] > 0) & (vc['Green'] == 0)].copy()
    green_only = vc[(vc['Green'] > 0) & (vc['Red'] == 0)].copy()

    n_swaps = 0
    # Align pair counts
    n_pairs = min(len(red_only), len(green_only), max_swaps)
    if n_pairs > 0:
        red_keys = list(red_only.index)[:n_pairs]
        green_keys = list(green_only.index)[:n_pairs]
        for rk, gk in zip(red_keys, green_keys):
            # Pick a Red in rk and a Green in gk
            idx_r = sample_index_in_class(dfw, rk, 'Red')
            idx_g = sample_index_in_class(dfw, gk, 'Green')
            if idx_r is None or idx_g is None: 
                continue
            # Swap party labels
            dfw.at[idx_r, 'party'], dfw.at[idx_g, 'party'] = dfw.at[idx_g, 'party'], dfw.at[idx_r, 'party']
            n_swaps += 1

    # 2) Minimal flips for remaining l=1 classes
    # Recompute after swaps
    vc2 = compute_party_counts_by_pub(dfw)
    red_only2 = vc2[(vc2['Red'] > 0) & (vc2['Green'] == 0)]
    green_only2 = vc2[(vc2['Green'] > 0) & (vc2['Red'] == 0)]

    n_flips = 0
    # Flip one record to create the missing party in each remaining class (bounded)
    for key in list(red_only2.index)[:max_flips]:
        idx = sample_index_in_class(dfw, key, 'Red')
        if idx is not None:
            dfw.at[idx, 'party'] = 'Green'
            n_flips += 1
    for key in list(green_only2.index)[:max_flips - n_flips]:
        idx = sample_index_in_class(dfw, key, 'Green')
        if idx is not None:
            dfw.at[idx, 'party'] = 'Red'
            n_flips += 1

    return dfw, n_swaps, n_flips

df, party_swaps, party_flips = repair_party_l_diversity(df, max_swaps=1000, max_flips=1000)
print(f"Party l-diversity repair: swaps={party_swaps}, flips={party_flips}")

# ---------------- Metrics (PUBLIC only; education EXCLUDED) -----------
def risk_metrics(df, qis, k=K):
    freq = df.groupby(qis, dropna=False).size().reset_index(name='count')
    merged = df.merge(freq, on=qis, how='left')
    total = len(df)
    unique = int((merged['count'] == 1).sum())
    small = int((merged['count'] < k).sum())
    avg_risk = float((1.0 / merged['count']).mean()) if len(merged) else float('nan')
    # l-diversity on 'party' (exclude NaN from counting distincts)
    l_table = df.groupby(qis)['party'].nunique(dropna=True).reset_index(name='l')
    l_viol = int((l_table['l'] < 2).sum())
    k_dist = freq['count'].value_counts().sort_index().to_dict()
    return {
        'total': total,
        'min_k': int(freq['count'].min()),
        'unique': unique,
        'unique_pct': unique / total * 100,
        'small': small,
        'small_pct': small / total * 100,
        'avg_risk_pct': avg_risk * 100,
        'l_violations': l_viol,
        'risk_by_k': k_dist
    }

print("\n=== Evaluation Metrics (PUBLIC QIs: sex, age_group, marital_status, evote) ===")
m_pub = risk_metrics(df, PUB, k=K)
for k_, v_ in m_pub.items():
    if k_ != 'risk_by_k': print(f"{k_}: {v_}")
print("risk_by_k:", m_pub['risk_by_k'])

# ---------------- Save final file -------------------------
df.to_csv(OUTPUT, index=False)
print(f"\nSaved output to {OUTPUT}")