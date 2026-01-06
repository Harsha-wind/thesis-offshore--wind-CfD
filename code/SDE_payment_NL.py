import pandas as pd

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("Loading data...")

# File paths
revenue_path = "D:/Thesis_Project/thesis_data/results/NL/Current/high/Market_revenue_NL_(High_NL).csv"
reference_path = "D:/Thesis_Project/thesis_data/results/NL/EP/High/Annual_Reference_Price_NL_High.csv"

# Load datasets
df_ref = pd.read_csv(reference_path)
df_rev = pd.read_csv(revenue_path, parse_dates=["hour"])

# Clean column names (remove whitespace)
df_ref.columns = df_ref.columns.str.strip()
df_rev.columns = df_rev.columns.str.strip()

print(f"Loaded {len(df_rev)} hourly revenue records")
print(f"Loaded {len(df_ref)} years of reference prices")

# ============================================================================
# STEP 2: SDE++ CONTRACT PARAMETERS
# ============================================================================

# Strike price from your SDE++ contract
STRIKE_PRICE = 54.5  # EUR/MWh

# NOTE: Annual_Reference_Price already includes offshore corrections
# applied during the annual aggregation step, so no additional 
# correction factor is needed here

# ============================================================================
# STEP 3: CLEAN AND RENAME COLUMNS
# ============================================================================
print("\nCleaning data...")

# Rename reference price columns
df_ref.rename(columns={
    col: 'Annual_Reference_Price_€/MWh' 
    for col in df_ref.columns 
    if 'Annual_Reference_Price' in col
}, inplace=True)

df_ref.rename(columns={
    col: 'year' 
    for col in df_ref.columns 
    if 'Year' in col
}, inplace=True)

# Rename revenue columns
df_rev.rename(columns={
    col: 'SpotPrice_€/MWh' 
    for col in df_rev.columns 
    if 'SpotPrice' in col
}, inplace=True)

df_rev.rename(columns={
    col: 'ActualGeneration_MWh' 
    for col in df_rev.columns 
    if 'Dispatched_MWh' in col
}, inplace=True)

df_rev.rename(columns={
    col: 'Market_Revenue_€' 
    for col in df_rev.columns 
    if 'Market_Revenue' in col
}, inplace=True)

# Keep only necessary columns
df_rev = df_rev[['hour', 'SpotPrice_€/MWh', 'ActualGeneration_MWh', 'Market_Revenue_€']].copy()

# ============================================================================
# STEP 4: AGGREGATE TO ANNUAL LEVEL
# ============================================================================
print("Aggregating to annual totals...")

# Filter to analysis period (2021-2035)
df_rev = df_rev[
    (df_rev['hour'].dt.year >= 2021) & 
    (df_rev['hour'].dt.year <= 2035)
].copy()

# Extract year
df_rev['year'] = df_rev['hour'].dt.year

# Aggregate to annual values
df_annual = df_rev.groupby('year').agg({
    'Market_Revenue_€': 'sum',           # Total market revenue
    'SpotPrice_€/MWh': 'mean',           # Average spot price
    'ActualGeneration_MWh': 'sum'        # Total generation
}).reset_index()

print(f"\nAnnual summary (last 5 years):")
print(df_annual.tail())

# ============================================================================
# STEP 5: MERGE WITH REFERENCE PRICES
# ============================================================================
print("\nMerging with reference prices...")

df_output = df_annual.merge(
    df_ref[["year", "Annual_Reference_Price_€/MWh"]], 
    on="year", 
    how="left"
)

# Check for missing reference prices
missing_refs = df_output['Annual_Reference_Price_€/MWh'].isna().sum()
if missing_refs > 0:
    print(f"⚠ Warning: {missing_refs} years missing reference prices")

# ============================================================================
# STEP 6: CALCULATE SDE++ SUPPORT PAYMENT
# ============================================================================
print("\nCalculating SDE++ support payments...")

# Standard SDE++ formula: Support = (Strike - Reference) × Generation
# - When RefPrice < Strike: You receive subsidy (positive support)
# - When RefPrice >= Strike: No subsidy needed (support = 0)
df_output['SupportPayment_€'] = (
    (STRIKE_PRICE - df_output["Annual_Reference_Price_€/MWh"]) * 
    df_output['ActualGeneration_MWh']
)

# Ensure no negative support payments (generator doesn't pay back)
# This happens when market prices exceed the strike price
df_output['SupportPayment_€'] = df_output['SupportPayment_€'].clip(lower=0)

# Calculate total revenue (market + support)
df_output['TotalRevenue_€'] = (
    df_output['Market_Revenue_€'] + 
    df_output['SupportPayment_€']
)

# ============================================================================
# STEP 7: SUMMARY AND OUTPUT
# ============================================================================
print("\n" + "="*70)
print("SDE++ SUPPORT PAYMENT SUMMARY")
print("="*70)

print(f"\nContract Parameters:")
print(f"  Strike Price: €{STRIKE_PRICE:.2f}/MWh")
print(f"  Reference Price: Includes offshore correction (pre-calculated)")

print(f"\nTotal Results ({df_output['year'].min()}-{df_output['year'].max()}):")
print(f"  Total Generation: {df_output['ActualGeneration_MWh'].sum():,.0f} MWh")
print(f"  Market Revenue: €{df_output['Market_Revenue_€'].sum():,.2f}")
print(f"  SDE++ Support: €{df_output['SupportPayment_€'].sum():,.2f}")
print(f"  Total Revenue: €{df_output['TotalRevenue_€'].sum():,.2f}")

# Count years with support
years_with_support = (df_output['SupportPayment_€'] > 0).sum()
print(f"\nYears receiving support: {years_with_support}/{len(df_output)}")

# Show which years received support vs market-only
print("\nSupport breakdown by year:")
for _, row in df_output.iterrows():
    year = int(row['year'])
    ref_price = row['Annual_Reference_Price_€/MWh']
    support = row['SupportPayment_€']
    
    if support > 0:
        status = f"✓ Support: €{support:,.0f} (RefPrice: €{ref_price:.2f}/MWh)"
    else:
        status = f"✗ No support (RefPrice €{ref_price:.2f}/MWh >= Strike €{STRIKE_PRICE:.2f}/MWh)"
    
    print(f"  {year}: {status}")

# Save output
output_path = "D:/Thesis_Project/thesis_data/results/NL/Current/high/Total_Revenue_NL_(High_NL).csv"
df_output.to_csv(output_path, index=False)
print(f"\n✓ Output saved to: {output_path}")