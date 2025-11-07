import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

# === Load data ===
df = pd.read_excel(r"C:\\Users\\andre\\Downloads\\Group goopers Dataset F-20251106\\private_dataF.xlsx")

# Only include valid votes
df = df[df['evote'].isin([0, 1])]

# Demographic columns to analyze
demographic_cols = ['sex', 'education', 'marital_status', 'zip', 'citizenship']

# Store p-values for later visualization
p_values = {}

# Function to plot grouped bar chart for each demographic variable
def plot_grouped_bar(table, title):
    table.plot(kind='bar', figsize=(10,6))
    plt.title(title)
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.legend(['evote = 0', 'evote = 1'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# === Loop through demographic columns ===
for col in demographic_cols:
    temp = df[[col, 'evote']].dropna()

    # Build contingency table
    table = pd.crosstab(temp[col], temp['evote'])

    # Chi-square test
    chi2, p, dof, expected = chi2_contingency(table)
    p_values[col] = p

    # Print summary
    print(f"\n=== {col.upper()} ===")
    print(table)
    print(f"Chi-square = {chi2:.3f}, p-value = {p:.4f}")

    if p < 0.05:
        print("→ Significant relationship: voting channel depends on", col)
    else:
        print("→ No significant relationship.")

    # Plot from actual data
    plot_grouped_bar(table, f'Voting Behavior by {col.capitalize()}')

# === Summary Heatmap of P-values ===
p_df = pd.DataFrame(list(p_values.items()), columns=['Demographic', 'p_value'])

# Sort by p-value ascending, so smallest (lightest) at top → largest (darkest) at bottom
p_df = p_df.sort_values(by='p_value', ascending=True)

# Create heatmap with gradient and colorbar
plt.figure(figsize=(6,4))
sns.heatmap(
    p_df[['p_value']].set_index(p_df['Demographic']),
    annot=True,
    cmap='Blues',          # lighter = lower p-value, darker = higher p-value
    fmt='.4f',
    cbar=True,             # show color spectrum
    cbar_kws={'label': 'p-value'}  # label for clarity
)
plt.title('Chi-Square Test P-Values (Lowest → Highest)')
plt.xlabel('')
plt.ylabel('')
plt.tight_layout()
plt.show()
