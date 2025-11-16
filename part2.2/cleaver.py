import pandas as pd

# Read the survey list
with open(r'C:\Users\Gamer\Downloads\survey_listL.txt', 'r') as f:
    survey_names = [line.strip() for line in f if line.strip()]

# Read the Excel file
df = pd.read_excel(r"C:\Users\Gamer\Downloads\public_data_registerL_citizenship_fixed.xlsx")

# Filter to keep only rows where the name is in the survey list
df_filtered = df[df['name'].isin(survey_names)]

# Save the filtered data to a new file
df_filtered.to_csv(r'C:\Users\Gamer\Downloads\filtered_data.csv', index=False)

print(f"Original rows: {len(df)}")
print(f"Filtered rows: {len(df_filtered)}")
print(f"Removed rows: {len(df) - len(df_filtered)}")
print("\nFiltered data saved to 'filtered_data.csv'")