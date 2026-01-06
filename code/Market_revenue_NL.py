import pandas as pd
import numpy as np

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("Loading data...")
act_path = "D:/Thesis_Project/thesis_data/results/NL/Current/high/hourly_actual_generation_wake_NL_(High_NL).csv"
df = pd.read_csv(act_path, parse_dates=["hours"])

# ============================================================================
# STEP 2: CLEAN AND PREPARE DATA
# ============================================================================
print("Cleaning column names...")

# Remove whitespace from column names
df.columns = df.columns.str.strip()

# Rename columns to standardized names
column_mapping = {
    'hours': 'hour',
    'SpotPrice/MWh': 'SpotPrice_€/MWh',
    'Actual_Generation_MWh': 'Actual_Generation_MWh',
    'WindSpeed_m/s': 'WindSpeed_m/s'
}

# Apply flexible renaming (handles columns that contain these strings)
for old_pattern, new_name in column_mapping.items():
    for col in df.columns:
        if old_pattern in col:
            df.rename(columns={col: new_name}, inplace=True)

# Keep only required columns
df = df[['hour', 'SpotPrice_€/MWh', 'Actual_Generation_MWh', 'WindSpeed_m/s']].copy()

# Convert to datetime, enforce UTC timezone
df['hour'] = pd.to_datetime(df['hour'], utc=True, errors='coerce')

# Remove invalid timestamps and duplicates, then sort
df = df.dropna(subset=['hour']).drop_duplicates(subset=['hour']).sort_values('hour').reset_index(drop=True)

print(f"Data loaded: {len(df)} hourly records from {df['hour'].min()} to {df['hour'].max()}")

# ============================================================================
# STEP 3: IMPLEMENT DUTCH SDE++ NEGATIVE PRICE POLICY
# ============================================================================
print("\nApplying SDE++ policy rules...")

# Define column shortcuts for readability
PRICE = 'SpotPrice_€/MWh'
GEN = 'Actual_Generation_MWh'

# --- Identify negative/zero price hours ---
is_negative_price = df[PRICE] <= 0

# --- Count consecutive negative price hours ---
# Create a block ID that increments each time we switch between negative/positive
price_change = is_negative_price.ne(is_negative_price.shift(fill_value=False))
block_id = price_change.cumsum()

# Count how many consecutive hours in each negative block
consecutive_neg_hours = is_negative_price.groupby(block_id).transform('size')

# Track position within each negative block (1st hour, 2nd hour, etc.)
df['neg_run'] = np.where(
    is_negative_price,
    is_negative_price.groupby(block_id).cumcount() + 1,
    0
)

# ============================================================================
# STEP 4: CALCULATE ELIGIBLE AND DISPATCHED GENERATION
# ============================================================================

# ELIGIBILITY FOR SUBSIDY (Policy Rule):
# - Positive price hours: Always eligible
# - First 5 consecutive negative hours: Eligible for subsidy
# - 6th+ consecutive negative hours: NOT eligible for subsidy
eligible_for_subsidy = (~is_negative_price) | (is_negative_price & (consecutive_neg_hours < 6))
df['Eligible_MWh'] = np.where(eligible_for_subsidy, df[GEN], 0.0)

# DISPATCH DECISION (Operator Choice):
# - Generators curtail from 6th consecutive negative hour onwards
# - Why? No subsidy = operating at a loss, so they shut down
keep_generating = ~is_negative_price | (consecutive_neg_hours < 6)
df['Dispatched_MWh'] = np.where(keep_generating, df[GEN], 0.0)

# MARKET REVENUE:
# Revenue = Spot Price × Actually Dispatched Generation
df['Market_Revenue_€'] = df[PRICE] * df['Dispatched_MWh']

# ============================================================================
# STEP 5: SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

# Count negative price occurrences
neg_hours = is_negative_price.sum()
total_hours = len(df)
neg_pct = 100 * neg_hours / total_hours

print(f"Total hours analyzed: {total_hours:,}")
print(f"Hours with negative/zero prices: {neg_hours:,} ({neg_pct:.1f}%)")

# Find longest consecutive negative stretch
if neg_hours > 0:
    max_consecutive = df[is_negative_price].groupby(block_id).size().max()
    print(f"Longest consecutive negative stretch: {max_consecutive} hours")
    
    # Count how many hours were curtailed (6+ consecutive)
    curtailed_hours = (df['neg_run'] >= 6).sum()
    curtailed_energy = df[df['neg_run'] >= 6][GEN].sum()
    print(f"Hours curtailed (6+ consecutive): {curtailed_hours}")
    print(f"Energy curtailed: {curtailed_energy:,.0f} MWh")

# Revenue summary
total_market_revenue = df['Market_Revenue_€'].sum()
print(f"\nTotal market revenue: €{total_market_revenue:,.2f}")
print(f"Average spot price: €{df[PRICE].mean():.2f}/MWh")

# ============================================================================
# STEP 6: SAVE OUTPUT
# ============================================================================
output_path = "D:/Thesis_Project/thesis_data/results/NL/Current/high/Market_revenue_NL_(High_NL).csv"
df.to_csv(output_path, index=False)
print(f"\n✓ Output saved to: {output_path}")

# Display first few rows
print("\nFirst 10 rows of output:")
print(df[['hour', PRICE, GEN, 'neg_run', 'Eligible_MWh', 'Dispatched_MWh', 'Market_Revenue_€']].head(10))

