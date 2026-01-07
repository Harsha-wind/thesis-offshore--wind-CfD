""" Basis Risk Anlysis for the referencing method used in Capability based CfD, flat average and the technology spcific reference
This code is used for all case study just by adjusting the input file paths"""

import pandas as pd
import numpy as np


print("TRACKING ERROR ANALYSIS: Capability-Based CfD")


# ==============================================================================
# 1. Load Hourly Data and Calculate Monthly Realized Price
# ==============================================================================

df_hourly = pd.read_csv("hourly_market_revenue_Ccfd_NL_high.csv", parse_dates=["hour"])

# Standardize column names
df_hourly['Month'] = df_hourly['hour'].dt.to_period('M')
df_hourly.rename(columns={
    [c for c in df_hourly.columns if 'SpotPrice_' in c][0]: 'SpotPrice',
    [c for c in df_hourly.columns if 'Actual_Generation' in c][0]: 'Generation'
}, inplace=True)

# Calculate generation-weighted monthly realized price
monthly_realized = df_hourly.groupby('Month').apply(
    lambda x: (x['SpotPrice'] * x['Generation']).sum() / x['Generation'].sum()
).reset_index(name='Realized_Price')

# ==============================================================================
# 2. Load Monthly Reference Prices
# ==============================================================================

# Flat average reference
df_flat = pd.read_csv("Total_Revenue_monthly_Ccfd_NL_high.csv", parse_dates=["Month"])
df_flat['Month'] = pd.to_datetime(df_flat['Month']).dt.to_period('M')
flat_col = [c for c in df_flat.columns if 'ReferencePrice_' in c][0]
flat_ref = df_flat[['Month', flat_col]].rename(columns={flat_col: 'Flat_Ref'})

# Tech-specific reference
df_tech = pd.read_csv("Total_Revenue_monthly_Ccfd_NL_high_techref.csv", parse_dates=["Month"])
df_tech['Month'] = pd.to_datetime(df_tech['Month']).dt.to_period('M')
tech_col = [c for c in df_tech.columns if 'ReferencePrice_' in c][0]
tech_ref = df_tech[['Month', tech_col]].rename(columns={tech_col: 'Tech_Ref'})

# ==============================================================================
# 3. Merge and Calculate Basis (Tracking Error)
# ==============================================================================

df = monthly_realized.merge(flat_ref, on='Month').merge(tech_ref, on='Month')

# Basis = Realized - Reference (deviation from what generator actually gets)
df['Basis_Flat'] = df['Realized_Price'] - df['Flat_Ref']
df['Basis_Tech'] = df['Realized_Price'] - df['Tech_Ref']

# ==============================================================================
# 4. Calculate Key Metrics
# ==============================================================================

# Tracking Error (standard deviation of basis)
te_flat = df['Basis_Flat'].std()
te_tech = df['Basis_Tech'].std()

# Correlation with realized price
corr_flat = df['Realized_Price'].corr(df['Flat_Ref'])
corr_tech = df['Realized_Price'].corr(df['Tech_Ref'])

# ==============================================================================
# 5. Results
# ==============================================================================

print(f"\nTracking Error (€/MWh):")
print(f"  Flat Average:    {te_flat:.2f}")
print(f"  Tech-Specific:   {te_tech:.2f}")
print(f"  Improvement:     {(te_flat - te_tech)/te_flat * 100:.1f}%")

print(f"\nCorrelation with Realized Market Price:")
print(f"  Flat Average:    {corr_flat:.3f}")
print(f"  Tech-Specific:   {corr_tech:.3f}")

# Summary table
summary = pd.DataFrame({
    'Metric': ['Tracking Error (€/MWh)', 'Correlation', 'TE Improvement (%)'],
    'Flat_Average': [f"{te_flat:.2f}", f"{corr_flat:.3f}", "-"],
    'Tech_Specific': [f"{te_tech:.2f}", f"{corr_tech:.3f}", 
                      f"{(te_flat - te_tech)/te_flat * 100:.1f}"]
})

print("\n" + summary.to_string(index=False))

# Save results
df.to_csv("tracking_error_monthly_data.csv", index=False)
summary.to_csv("tracking_error_summary.csv", index=False)

print("\n✓ Analysis complete - files saved")