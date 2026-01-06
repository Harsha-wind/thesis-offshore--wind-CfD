import pandas as pd
import numpy as np

print("="*80)
print("TRACKING ERROR ANALYSIS: Capability-Based CfD")
print("="*80)

# ==============================================================================
# STEP 1: Load Hourly Data and Calculate Monthly Realized Price
# ==============================================================================

# Load your hourly data
df_hourly = pd.read_csv(
    "D:/Thesis_Project/thesis_data/results/NL/Capability/high/hourly_market_revenue_Ccfd_NL_(High_NL).csv",
    parse_dates=["hour"]
)

print(f"\n✓ Loaded {len(df_hourly)} hourly records")

# Create month column
df_hourly['Month'] = df_hourly['hour'].dt.to_period('M')
for col in df_hourly.columns:
    if 'SpotPrice_' in col:
        df_hourly.rename(columns={col: 'SpotPrice'}, inplace=True)
    if 'Actual_Generation_MWh' in col:
        df_hourly.rename(columns={col: 'ActualGeneration'}, inplace=True)
# Calculate Monthly Realized Price (generation-weighted average)
monthly_realized = df_hourly.groupby('Month').apply(
    lambda x: (x['SpotPrice'] * x['ActualGeneration']).sum() / x['ActualGeneration'].sum()
).reset_index()
monthly_realized.columns = ['Month', 'Realized_Market_Price']

print(f"✓ Calculated monthly realized prices for {len(monthly_realized)} months")

# ==============================================================================
# STEP 2: Load Your Existing Monthly Reference Prices
# ==============================================================================

# Flat Average Reference (monthly)
df_flat = pd.read_csv(
    "D:/Thesis_Project/thesis_data/results/NL/Capability/high/Total_Revenue_monthly_Ccfd_NL_high.csv",
    parse_dates=["Month"]
)
# Extract reference price column - adjust column name as needed
df_flat['Month'] = pd.to_datetime(df_flat['Month']).dt.to_period('M')
for col in df_flat.columns:
    if 'ReferencePrice_' in col:
        df_flat.rename(columns={col: 'ReferencePrice_flat'}, inplace=True)
flat_ref = df_flat[['Month', 'ReferencePrice_flat']].copy()  # Adjust column name!
flat_ref.columns = ['Month', 'Flat_Avg_Ref']

# Tech-Specific Reference (monthly)
df_tech = pd.read_csv(
    "D:/Thesis_Project/thesis_data/results/NL/Capability/Tech_ref/high/Total_Revenue_monthly_Ccfd_NL_high_techref.csv",
    parse_dates=["Month"]
)
df_tech['Month'] = pd.to_datetime(df_tech['Month']).dt.to_period('M')
for col in df_tech.columns:
    if 'ReferencePrice_' in col:
        df_tech.rename(columns={col: 'ReferencePrice_tech'}, inplace=True)
tech_ref = df_tech[['Month', 'ReferencePrice_tech']].copy()  # Adjust column name!
tech_ref.columns = ['Month', 'TechSpec_Ref']

print(f"✓ Loaded flat average references")
print(f"✓ Loaded tech-specific references")

# ==============================================================================
# STEP 3: Merge All Data
# ==============================================================================

df_monthly = monthly_realized.merge(flat_ref, on='Month')
df_monthly = df_monthly.merge(tech_ref, on='Month')

print(f"\n✓ Merged data: {len(df_monthly)} months")

# ==============================================================================
# STEP 4: Calculate BASIS (deviation from realized market price)
# ==============================================================================

df_monthly['Basis_Flat'] = df_monthly['Realized_Market_Price'] - df_monthly['Flat_Avg_Ref']
df_monthly['Basis_Tech'] = df_monthly['Realized_Market_Price'] - df_monthly['TechSpec_Ref']

# ==============================================================================
# STEP 5: Calculate TRACKING ERROR (Standard Deviation of Basis)
# ==============================================================================

te_flat = np.std(df_monthly['Basis_Flat'], ddof=1)
te_tech = np.std(df_monthly['Basis_Tech'], ddof=1)

# Mean basis (systematic deviation)
mean_basis_flat = df_monthly['Basis_Flat'].mean()
mean_basis_tech = df_monthly['Basis_Tech'].mean()

# RMSE (alternative measure)
rmse_flat = np.sqrt(np.mean(df_monthly['Basis_Flat']**2))
rmse_tech = np.sqrt(np.mean(df_monthly['Basis_Tech']**2))

print("\n" + "="*80)
print("TRACKING ERROR RESULTS")
print("="*80)

print(f"\n1. Average Monthly Prices:")
print(f"   Realized Market Price:       €{df_monthly['Realized_Market_Price'].mean():.2f}/MWh")
print(f"   Flat Average Reference:      €{df_monthly['Flat_Avg_Ref'].mean():.2f}/MWh")
print(f"   Tech-Specific Reference:     €{df_monthly['TechSpec_Ref'].mean():.2f}/MWh")

print(f"\n2. Tracking Error (Standard Deviation of Basis):")
print(f"   Flat Average:    {te_flat:.2f} €/MWh")
print(f"   Tech-Specific:   {te_tech:.2f} €/MWh")
print(f"   Improvement:     {(te_flat - te_tech)/te_flat * 100:.1f}%")

print(f"\n3. RMSE (Root Mean Square Error):")
print(f"   Flat Average:    {rmse_flat:.2f} €/MWh")
print(f"   Tech-Specific:   {rmse_tech:.2f} €/MWh")
print(f"   Improvement:     {(rmse_flat - rmse_tech)/rmse_flat * 100:.1f}%")

print(f"\n4. Mean Basis (Systematic Deviation):")
print(f"   Flat Average:    {mean_basis_flat:.2f} €/MWh")
print(f"   Tech-Specific:   {mean_basis_tech:.2f} €/MWh")

# Correlation
corr_flat = np.corrcoef(df_monthly['Realized_Market_Price'], 
                        df_monthly['Flat_Avg_Ref'])[0,1]
corr_tech = np.corrcoef(df_monthly['Realized_Market_Price'], 
                        df_monthly['TechSpec_Ref'])[0,1]

print(f"\n5. Correlation with Realized Market Price:")
print(f"   Flat Average:    {corr_flat:.4f}")
print(f"   Tech-Specific:   {corr_tech:.4f}")

# ==============================================================================
# STEP 6: Summary Statistics
# ==============================================================================

print("\n" + "="*80)
print("SUMMARY FOR THESIS TABLE")
print("="*80)

summary = pd.DataFrame({
    'Metric': [
        'Average Price (€/MWh)',
        'Tracking Error (€/MWh)',
        'RMSE (€/MWh)',
        'Mean Basis (€/MWh)',
        'Correlation',
        'TE Improvement (%)'
    ],
    'Flat_Average': [
        f"{df_monthly['Flat_Avg_Ref'].mean():.2f}",
        f"{te_flat:.2f}",
        f"{rmse_flat:.2f}",
        f"{mean_basis_flat:.2f}",
        f"{corr_flat:.4f}",
        "-"
    ],
    'Tech_Specific': [
        f"{df_monthly['TechSpec_Ref'].mean():.2f}",
        f"{te_tech:.2f}",
        f"{rmse_tech:.2f}",
        f"{mean_basis_tech:.2f}",
        f"{corr_tech:.4f}",
        f"{(te_flat - te_tech)/te_flat * 100:.1f}"
    ]
})

print("\n" + summary.to_string(index=False))

# ==============================================================================
# STEP 7: Export Results
# ==============================================================================

output_file = "D:/Thesis_Project/thesis_data/results/NL/Capability/Basis_risk/tracking_error_high.csv"
df_monthly.to_csv(output_file, index=False)
print(f"\n✓ Monthly data saved to: {output_file}")

summary_file = "D:/Thesis_Project/thesis_data/results/NL/Capability/Basis_risk/tracking_error_summary_high.csv"
summary.to_csv(summary_file, index=False)
print(f"✓ Summary saved to: {summary_file}")

print("\n" + "="*80)
print("✓ TRACKING ERROR ANALYSIS COMPLETE")
print("="*80)