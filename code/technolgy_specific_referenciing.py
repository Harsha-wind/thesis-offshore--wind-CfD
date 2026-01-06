import pandas as pd
import numpy as np

# ===== 1. DATA LOADING AND CLEANING =====
csv_path = "D:/Thesis_Project/thesis_data/results/DK/Cap_Cfd/high/Capability_generation_DK_high.csv"
df = pd.read_csv(csv_path)

# Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

# Parse datetime and remove invalid rows
df['hours'] = pd.to_datetime(df['hours'], errors='coerce')
df = df.dropna(subset=['hours'])

print(f"Loaded {len(df):,} hourly records from {df['hours'].min()} to {df['hours'].max()}")

# ===== 2. COLUMN RENAMING =====
# More efficient renaming using a dictionary
column_rename_map = {}
for col in df.columns:
    if 'SpotPrice/MWh' in col:
        column_rename_map[col] = 'SpotPrice_€/MWh'
    elif 'TotalCapability_MWh_withwake' in col:
        column_rename_map[col] = 'TotalCapability_MWh'

df.rename(columns=column_rename_map, inplace=True)

# ===== 3. CALCULATE MONTHLY TECH-SPECIFIC REFERENCE PRICE =====
# Add month column
df['Month'] = df['hours'].dt.to_period('M')

# Calculate weighted spot price (spot price × generation capability)
df['weighted_spot'] = df['SpotPrice_€/MWh'] * df['TotalCapability_MWh']

# Group by month and calculate volume-weighted average
monthly_tech_ref = (
    df.groupby('Month')
    .agg(
        weighted_sum=('weighted_spot', 'sum'),
        cap_sum=('TotalCapability_MWh', 'sum'),
        avg_spot=('SpotPrice_€/MWh', 'mean'),  # Added: simple average for comparison
        hours_count=('hours', 'count')  # Added: number of hours
    )
    .reset_index()
)

# Convert period to string for better CSV compatibility
monthly_tech_ref['Month'] = monthly_tech_ref['Month'].astype(str)

# Calculate tech-specific reference price (volume-weighted average)
# Formula: Σ(spot_price × capability) / Σ(capability)
monthly_tech_ref['tech_specific_ref_price'] = np.where(
    monthly_tech_ref['cap_sum'] > 0,
    monthly_tech_ref['weighted_sum'] / monthly_tech_ref['cap_sum'],
    np.nan
)

# Calculate the difference between weighted and simple average
monthly_tech_ref['price_difference'] = (
    monthly_tech_ref['tech_specific_ref_price'] - monthly_tech_ref['avg_spot']
)

# ===== 4. DISPLAY RESULTS =====
print("\n" + "="*80)
print("MONTHLY TECH-SPECIFIC REFERENCE PRICES")
print("="*80)
print(monthly_tech_ref[['Month', 'tech_specific_ref_price', 'avg_spot', 
                        'price_difference', 'cap_sum', 'hours_count']].to_string(index=False))

# ===== 5. SAVE RESULTS =====
output_path = "D:/Thesis_Project/thesis_data/results/DK/Cap_Cfd/Tech_specific/high/monthly_tech_specific_ref_price_high_dk.csv"
monthly_tech_ref.to_csv(output_path, index=False)
print(f"\n✓ Monthly tech-specific reference prices saved to:\n  {output_path}")

# ===== 6. OPTIONAL: ADDITIONAL STATISTICS =====
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"Average tech-specific ref price: €{monthly_tech_ref['tech_specific_ref_price'].mean():.2f}/MWh")
print(f"Average simple spot price: €{monthly_tech_ref['avg_spot'].mean():.2f}/MWh")
print(f"Total capability generated: {monthly_tech_ref['cap_sum'].sum():,.0f} MWh")
print(f"Number of months analyzed: {len(monthly_tech_ref)}")