import pandas as pd
import numpy as np

# -----------------------------
# Configuration
# -----------------------------
INPUT_PATH  = r"C:\Users\andre\Downloads\Group goopers Dataset F-20251106\suppressed_dataF.csv"
OUTPUT_PATH = r"C:\Users\andre\Downloads\Group goopers Dataset F-20251106\pram_dataF.csv"

quasi_identifiers = ['age_group', 'sex', 'marital_status', 'evote']
dominance_threshold = 0.80
flip_frac_low, flip_frac_high = 0.20, 0.40
opposite_party = {'Green': 'Red', 'Red': 'Green'}

seed = 69
rng = np.random.default_rng(seed)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(INPUT_PATH)

# Normalize party strings to avoid trailing spaces/case issues
if df['party'].dtype == object:
    df['party'] = df['party'].astype(str).str.strip()

# (Optional) Normalize evote dtype so group filters match consistently
# If evote sometimes comes as "1"/"0" strings, this coerces to numeric.
if df['evote'].dtype == object:
    df['evote'] = pd.to_numeric(df['evote'], errors='coerce')

# Validate required cols
required = set(quasi_identifiers + ['party'])
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# -----------------------------
# Compute per-group party counts
# -----------------------------
party_counts = (
    df.groupby(quasi_identifiers + ['party'], dropna=True)
      .size().reset_index(name='party_count')
)
group_totals = (
    df.groupby(quasi_identifiers, dropna=True)
      .size().reset_index(name='k')
)

stats = party_counts.merge(group_totals, on=quasi_identifiers, how='left')
stats['party_ratio'] = stats['party_count'] / stats['k']

# Dominant per group
idx_max = stats.groupby(quasi_identifiers, dropna=True)['party_ratio'].idxmax()
dominant = stats.loc[idx_max].copy()

eligible = dominant[
    (dominant['party_ratio'] >= dominance_threshold) &
    (dominant['party'].isin(opposite_party.keys()))
].reset_index(drop=True)

# --- Correct minority (opposite) count per eligible group ---
eligible = eligible.assign(opp_party=eligible['party'].map(opposite_party))

# Map from (QI..., party) -> count
key_cols = quasi_identifiers + ['party']
counts_map = party_counts.set_index(key_cols)['party_count']

def get_opp_count(row):
    key = tuple(row[q] for q in quasi_identifiers) + (row['opp_party'],)
    return int(counts_map.get(key, 0))

eligible['opp_count'] = eligible.apply(get_opp_count, axis=1)

# -----------------------------
# Flip with 40% minority cap
# -----------------------------
total_flipped = 0
groups_processed = 0
groups_capped = 0

for _, row in eligible.iterrows():
    # Build group mask
    gmask = pd.Series(True, index=df.index)
    for qi in quasi_identifiers:
        gmask &= (df[qi] == row[qi])

    dom_party = row['party']
    dom_mask = gmask & (df['party'] == dom_party)

    idx_dom = df.index[dom_mask]
    k = int(row['k'])
    n_dom = len(idx_dom)
    n_minority = int(row['opp_count'])

    if k == 0 or n_dom == 0:
        continue

    minority_share_before = n_minority / k
    if minority_share_before >= 0.40:
        # already at/above cap; skip
        continue

    # Draw desired fraction in [0.20, 0.40], then cap to avoid exceeding 40% minority
    desired_flip_frac = float(rng.uniform(flip_frac_low, flip_frac_high))
    max_flip_frac_allowed = max(0.0, 0.40 - minority_share_before)
    actual_flip_frac = min(desired_flip_frac, max_flip_frac_allowed)

    # Convert to integer flips
    x_draw = int(np.floor(k * actual_flip_frac))      # rows from draw (already capped by fraction)
    x_cap  = int(np.floor(k * max_flip_frac_allowed)) # absolute cap in rows from 40% rule

    # Do not exceed available dominant rows
    x_cap = min(x_cap, n_dom)

    # Ensure at least 1 flip if cap allows, even when floor(...) is 0 (small k)
    if x_draw == 0 and x_cap >= 1:
        n_to_flip = 1
    else:
        n_to_flip = min(x_draw, x_cap)

    # Keep this guard — prevents no-op work and accidental negatives
    if n_to_flip <= 0:
        continue

    chosen = rng.choice(idx_dom, size=n_to_flip, replace=False)
    df.loc[chosen, 'party'] = opposite_party[dom_party]

    total_flipped += n_to_flip
    groups_processed += 1
    if actual_flip_frac < desired_flip_frac:
        groups_capped += 1

# -----------------------------
# Save results
# -----------------------------
df.to_csv(OUTPUT_PATH, index=False)

print("=== 40%-Cap Dominance Flip Summary ===")
print(f"Eligible dominant groups (≥ {int(dominance_threshold*100)}%): {len(eligible)}")
print(f"Groups processed (flips > 0): {groups_processed}")
print(f"Groups capped by 40% limit: {groups_capped}")
print(f"Total rows flipped: {total_flipped}")
print(f"Saved updated dataset to: {OUTPUT_PATH}")

# -----------------------------
# Optional: post-check (sanity assertion)
# -----------------------------
# Recompute shares and assert every processed group's (original minority) ≤ 40%.
# Uncomment to enforce:
# post_pc = (
#     df.groupby(quasi_identifiers + ['party'], dropna=True)
#       .size().reset_index(name='cnt')
# )
# post_gt = (
#     df.groupby(quasi_identifiers, dropna=True)
#       .size().reset_index(name='k')
# )
# post_stats = post_pc.merge(post_gt, on=quasi_identifiers, how='left')
# for _, row in eligible.iterrows():
#     gmask = (post_stats[quasi_identifiers] == pd.Series({q: row[q] for q in quasi_identifiers})).all(axis=1)
#     g = post_stats[gmask]
#     if g.empty:
#         continue
#     minority_party = opposite_party[row['party']]
#     min_cnt = g.loc[g['party'] == minority_party, 'cnt'].sum()
#     k = int(g['k'].iloc[0])
#     if k > 0:
#         assert (min_cnt / k) <= 0.40 + 1e-9, f"Cap violated for {row[quasi_identifiers].to_dict()}"
