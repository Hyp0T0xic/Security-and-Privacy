import pandas as pd
import numpy as np
from collections import defaultdict

# =========================================================
# CONFIGURATION
# =========================================================

# Paths
ANONYMIZED_PATH = r"C:\Users\andre\Downloads\invading privacy\anonymised_dataL.xlsx"
PUBLIC_PATH = r"C:\Users\andre\Downloads\invading privacy\anonymised_dataL_predicted.xlsx"
OUTPUT_PATH = r"C:\Users\andre\Downloads\invading privacy\reidentification_results.xlsx"

# Age noise parameter (±5 years)
AGE_NOISE = 5

# =========================================================
# LOAD DATA
# =========================================================

anon_df = pd.read_excel(ANONYMIZED_PATH)
public_df = pd.read_excel(PUBLIC_PATH)

# Convert zip codes to strings to handle "*" suppression properly
anon_df['zip_a'] = anon_df['zip_a'].astype(str)
public_df['zip_a'] = public_df['zip_a'].astype(str)

print(f"Loaded {len(anon_df)} anonymized records")
print(f"Loaded {len(public_df)} public records")
print(f"\nData types:")
print(f"  Anonymized zip: {anon_df['zip_a'].dtype}")
print(f"  Public zip: {public_df['zip_a'].dtype}")
print(f"\nSample zip values (anonymized): {anon_df['zip_a'].unique()[:5]}")
print(f"Sample zip values (public): {public_df['zip_a'].unique()[:5]}")

# =========================================================
# AGE MATCHING LOGIC (accounting for ±5 year noise)
# =========================================================

def age_groups_can_match(age_group1, age_group2):
    """
    Check if two age groups could match given ±5 year noise.
    
    Examples:
    - "<30" could match "<30" or "30-49" (if person is 25-34)
    - "30-49" could match "<30", "30-49", or "50-64"
    - "50-64" could match "30-49", "50-64", or "65+"
    - "65+" could match "50-64" or "65+"
    """
    if pd.isna(age_group1) or pd.isna(age_group2):
        return False
    
    # Same group always matches
    if age_group1 == age_group2:
        return True
    
    # Define age group boundaries
    age_ranges = {
        "<30": (0, 29),
        "30-49": (30, 49),
        "50-64": (50, 64),
        "65+": (65, 150)
    }
    
    if age_group1 not in age_ranges or age_group2 not in age_ranges:
        return False
    
    min1, max1 = age_ranges[age_group1]
    min2, max2 = age_ranges[age_group2]
    
    # Add noise buffer: expand each range by ±5
    min1_expanded = min1 - AGE_NOISE
    max1_expanded = max1 + AGE_NOISE
    min2_expanded = min2 - AGE_NOISE
    max2_expanded = max2 + AGE_NOISE
    
    # Check if ranges overlap
    return not (max1_expanded < min2_expanded or max2_expanded < min1_expanded)

# =========================================================
# MATCHING FUNCTION
# =========================================================

def find_matches(anon_row, public_df):
    """
    Find all potential matches in public data for a given anonymized record.
    Returns list of (index, match_strength) tuples.
    """
    matches = []
    
    for idx, public_row in public_df.iterrows():
        match_score = 0
        max_score = 0
        
        # Age (with noise tolerance)
        max_score += 1
        if age_groups_can_match(anon_row['age_a'], public_row['age_a']):
            match_score += 1
        else:
            continue  # Age must match (within tolerance)
        
        # Sex (exact match required)
        max_score += 1
        if anon_row['sex'] == public_row['sex']:
            match_score += 1
        else:
            continue  # Sex must match exactly
        
        # Marital status (exact match required)
        max_score += 1
        if anon_row['maritalstatus_a'] == public_row['maritalstatus_a']:
            match_score += 1
        else:
            continue  # Marital status must match exactly
        
        # Last voted (exact match required)
        max_score += 1
        if anon_row['last_voted'] == public_row['last_voted']:
            match_score += 1
        else:
            continue  # Last voted must match exactly
        
        # Citizenship (exact match required)
        max_score += 1
        if anon_row['citizenship_a'] == public_row['citizenship_a']:
            match_score += 1
        else:
            continue  # Citizenship must match exactly
        
        # Zip code (handle suppression with "*")
        anon_zip = str(anon_row['zip_a'])
        public_zip = str(public_row['zip_a'])
        
        if anon_zip != "*" and public_zip != "*":
            # Both zips available - must match
            max_score += 1
            if anon_zip == public_zip:
                match_score += 1
            else:
                continue  # Zip mismatch
        # If either is suppressed, we skip zip in scoring
        
        # Calculate match strength (percentage)
        match_strength = (match_score / max_score) * 100
        matches.append((idx, match_strength, public_row['name']))
    
    return matches

# =========================================================
# PERFORM MATCHING
# =========================================================

results = []

for anon_idx, anon_row in anon_df.iterrows():
    matches = find_matches(anon_row, public_df)
    
    if len(matches) == 0:
        result = {
            'anon_index': anon_idx,
            'party': anon_row['party'],
            'match_count': 0,
            'match_type': 'No Match',
            'confidence': 'N/A',
            'matched_name': None,
            'match_strength': 0,
            **{col: anon_row[col] for col in ['age_a', 'sex', 'maritalstatus_a', 'last_voted', 'citizenship_a', 'zip_a']}
        }
        results.append(result)
    
    elif len(matches) == 1:
        # Unique match - high confidence re-identification!
        idx, strength, name = matches[0]
        result = {
            'anon_index': anon_idx,
            'party': anon_row['party'],
            'match_count': 1,
            'match_type': 'UNIQUE MATCH',
            'confidence': 'HIGH',
            'matched_name': name,
            'match_strength': strength,
            **{col: anon_row[col] for col in ['age_a', 'sex', 'maritalstatus_a', 'last_voted', 'citizenship_a', 'zip_a']}
        }
        results.append(result)
    
    else:
        # Multiple matches - ambiguous
        for idx, strength, name in matches:
            result = {
                'anon_index': anon_idx,
                'party': anon_row['party'],
                'match_count': len(matches),
                'match_type': 'Multiple Matches',
                'confidence': 'LOW',
                'matched_name': name,
                'match_strength': strength,
                **{col: anon_row[col] for col in ['age_a', 'sex', 'maritalstatus_a', 'last_voted', 'citizenship_a', 'zip_a']}
            }
            results.append(result)

# =========================================================
# CREATE RESULTS DATAFRAME
# =========================================================

results_df = pd.DataFrame(results)

# =========================================================
# SUMMARY STATISTICS
# =========================================================

print("\n" + "="*60)
print("RE-IDENTIFICATION RESULTS")
print("="*60)

unique_matches = results_df[results_df['match_type'] == 'UNIQUE MATCH']
multiple_matches = results_df[results_df['match_type'] == 'Multiple Matches']
no_matches = results_df[results_df['match_type'] == 'No Match']

print(f"\nTotal anonymized records: {len(anon_df)}")
print(f"Unique matches (re-identified): {len(unique_matches)}")
print(f"Records with multiple possible matches: {len(multiple_matches['anon_index'].unique())}")
print(f"Records with no matches: {len(no_matches)}")

print(f"\n✅ Successfully re-identified: {len(unique_matches)} people ({len(unique_matches)/len(anon_df)*100:.1f}%)")

if len(unique_matches) > 0:
    print("\n" + "="*60)
    print("SUCCESSFULLY RE-IDENTIFIED INDIVIDUALS:")
    print("="*60)
    for _, row in unique_matches.iterrows():
        print(f"\n🎯 {row['matched_name']}")
        print(f"   Party: {row['party']}")
        print(f"   Demographics: {row['sex']}, {row['age_a']}, {row['maritalstatus_a']}")
        print(f"   Zip: {row['zip_a']}, Last voted: {row['last_voted']}")
        print(f"   Match strength: {row['match_strength']:.0f}%")

# Show some ambiguous cases
if len(multiple_matches) > 0:
    print("\n" + "="*60)
    print("SAMPLE AMBIGUOUS CASES (Multiple Matches):")
    print("="*60)
    
    # Get first few unique anon_index values with multiple matches
    sample_ambiguous = multiple_matches['anon_index'].unique()[:3]
    
    for anon_idx in sample_ambiguous:
        subset = multiple_matches[multiple_matches['anon_index'] == anon_idx]
        row = subset.iloc[0]
        print(f"\n⚠️  Anonymized record #{anon_idx} (Party: {row['party']})")
        print(f"   Could be any of {len(subset)} people:")
        for _, match_row in subset.iterrows():
            print(f"   - {match_row['matched_name']} (match: {match_row['match_strength']:.0f}%)")

# =========================================================
# SAVE RESULTS
# =========================================================

results_df.to_excel(OUTPUT_PATH, index=False)
print(f"\n📁 Full results saved to: {OUTPUT_PATH}")

# =========================================================
# PRIVACY RISK ASSESSMENT
# =========================================================

print("\n" + "="*60)
print("PRIVACY RISK ASSESSMENT")
print("="*60)

re_id_rate = len(unique_matches) / len(anon_df) * 100
ambiguous_rate = len(multiple_matches['anon_index'].unique()) / len(anon_df) * 100

print(f"\nRe-identification rate: {re_id_rate:.1f}%")
print(f"Ambiguous records: {ambiguous_rate:.1f}%")

if re_id_rate > 5:
    print("\n⚠️  HIGH RISK: >5% of records can be re-identified")
elif re_id_rate > 0:
    print("\n⚠️  MEDIUM RISK: Some records can be re-identified")
else:
    print("\n✅ LOW RISK: No unique re-identifications found")

print("\n💡 Note: Age has ±5 year noise, so matches account for overlapping age groups")