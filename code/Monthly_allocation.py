import pandas as pd

# ============================================================================
# SETTINGS
# ============================================================================

# Path to your annual revenue file with support payments
annual_file = "D:/Thesis_Project/thesis_data/results/NL/Current/high/Total_Revenue_NL_(High_NL).csv"

# Load annual data to get support payments
annual_df = pd.read_csv(annual_file)
annual_df.columns = annual_df.columns.str.strip()

# Extract support payments for years with non-zero support
# Filter out years with 0 or NaN support
SUPPORT_PAYMENTS = {}
for _, row in annual_df.iterrows():
    year = int(row['year'])
    support = row['SupportPayment_€']
    if pd.notna(support) and support > 0:
        SUPPORT_PAYMENTS[year] = support

print(f"Loaded support payments from: {annual_file}")
print(f"Years with support: {list(SUPPORT_PAYMENTS.keys())}")

# ============================================================================
# LOAD MONTHLY DATA
# ============================================================================
file_path = "D:/Thesis_Project/thesis_data/results/NL/Current/high/Monthly_market_revenue_NL_(High_NL).csv"
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip()

# ============================================================================
# STANDARDIZE COLUMN NAMES
# ============================================================================
column_mapping = {}
for col in df.columns:
    if 'Monthly_Market_Revenue' in col:
        column_mapping[col] = 'Market_Revenue_€'
    elif 'AvgSpotPrice' in col:
        column_mapping[col] = 'AvgSpotPrice_€/MWh'
    elif 'ActualGeneration_MWh' in col:
        column_mapping[col] = 'ActualGeneration_MWh'

df.rename(columns=column_mapping, inplace=True)

# ============================================================================
# PROCESS DATES
# ============================================================================
df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
df['Year'] = df['Month'].dt.year

print(f"Data range: {df['Month'].min().strftime('%Y-%m')} to {df['Month'].max().strftime('%Y-%m')}")
print(f"Total months: {len(df)}")

# ============================================================================
# ADD SDE+ SUPPORT TO MULTIPLE YEARS
# ============================================================================
df['Monthly_SDE_Support_€'] = 0.0

print(f"\n{'='*60}")
print("ADDING SDE+ SUPPORT PAYMENTS")
print(f"{'='*60}")

# Loop through each year with support and distribute monthly
for year, annual_support in SUPPORT_PAYMENTS.items():
    # Count months in this year
    months_in_year = (df['Year'] == year).sum()
    
    if months_in_year > 0:
        # Calculate monthly support (divide annual by 12)
        monthly_support = annual_support / 12
        
        # Add to dataframe
        df.loc[df['Year'] == year, 'Monthly_SDE_Support_€'] = monthly_support
        
        print(f"{year}: €{annual_support:>15,.2f} annual → €{monthly_support:>12,.2f}/month ({months_in_year} months)")
    else:
        print(f"{year}: WARNING - No data found for this year!")

# ============================================================================
# CALCULATE TOTAL REVENUE
# ============================================================================
df['Monthly_Total_Revenue_€'] = df['Market_Revenue_€'] + df['Monthly_SDE_Support_€']

# ============================================================================
# SAVE OUTPUT
# ============================================================================
# Convert Month back to string format for CSV
df['Month'] = df['Month'].dt.strftime('%Y-%m')

# Select output columns
output = df[['Month', 'Year', 'AvgSpotPrice_€/MWh', 'ActualGeneration_MWh', 
              'Market_Revenue_€', 'Monthly_SDE_Support_€', 'Monthly_Total_Revenue_€']]

output_path = "D:/Thesis_Project/thesis_data/results/NL/Current/high/Monthly_Revenue_with_SDE_Support_4years.csv"
output.to_csv(output_path, index=False)

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*60}")
print("ANNUAL REVENUE SUMMARY")
print(f"{'='*60}")

annual_summary = df.groupby('Year')[['Market_Revenue_€', 
                                      'Monthly_SDE_Support_€', 
                                      'Monthly_Total_Revenue_€']].sum()

# Format for better readability
annual_summary_display = annual_summary.copy()
annual_summary_display.columns = ['Market Revenue (€)', 'SDE+ Support (€)', 'Total Revenue (€)']

print(annual_summary_display.round(2))

# ============================================================================
# VERIFICATION
# ============================================================================
print(f"\n{'='*60}")
print("VERIFICATION: Support Payments")
print(f"{'='*60}")

for year, expected_support in SUPPORT_PAYMENTS.items():
    actual_support = df[df['Year'] == year]['Monthly_SDE_Support_€'].sum()
    difference = actual_support - expected_support
    
    status = "✓" if abs(difference) < 0.01 else "✗"
    print(f"{year} {status}: Expected €{expected_support:>13,.2f} | Actual €{actual_support:>13,.2f} | Diff: €{difference:>8,.2f}")

# Total support across all years
total_expected = sum(SUPPORT_PAYMENTS.values())
total_actual = df['Monthly_SDE_Support_€'].sum()
print(f"\nTotal Support: €{total_actual:,.2f} (Expected: €{total_expected:,.2f})")

print(f"\n{'='*60}")
print(f"✓ Output saved to:\n  {output_path}")
print(f"{'='*60}")