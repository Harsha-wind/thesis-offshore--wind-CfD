import pandas as pd

# =========================
# CONFIG — set file paths
# =========================
hist_path = "E:/Thesis_Project/thesis_data/results/NL/EP/merged_wind_SP_NL.csv"
sim_path  = "E:/Thesis_Project/thesis_data/results/NL/EP/High/future_prices_Scenario3_NL.csv"
out_path  = "E:/Thesis_Project/thesis_data/results/NL/EP/High/merged_wind_price_full_high_NL.csv"

# === Load datasets ===
df_hist = pd.read_csv(hist_path)
df_sim  = pd.read_csv(sim_path)

# Strip column names (avoid hidden spaces)
df_hist.columns = df_hist.columns.str.strip()
df_sim.columns  = df_sim.columns.str.strip()

print("=== Initial columns ===")
print("Historical:", df_hist.columns.tolist())
print("Simulated:", df_sim.columns.tolist())

# Parse datetime columns FIRST before any renaming
df_hist["datetime"] = pd.to_datetime(df_hist["datetime"], utc=True, errors="coerce")
df_sim["datetime"] = pd.to_datetime(df_sim["datetime"], utc=True, errors="coerce")

print(f"\nParsed historical datetime range: {df_hist['datetime'].min()} to {df_hist['datetime'].max()}")
print(f"Parsed simulated datetime range: {df_sim['datetime'].min()} to {df_sim['datetime'].max()}")

# === Remove overlap BEFORE renaming ===
last_hist_time = df_hist["datetime"].max()
print(f"\nLast historical timestamp: {last_hist_time}")

# Filter simulated data to only include dates AFTER the last historical date
df_sim = df_sim[df_sim["datetime"] > last_hist_time].copy()
print(f"Simulated data after filtering: {len(df_sim)} rows")
if len(df_sim) > 0:
    print(f"Simulated date range after filter: {df_sim['datetime'].min()} to {df_sim['datetime'].max()}")

# Select and rename columns for historical data
#df_hist = df_hist[['hour', 'SpotPrice', 'Dispatched_MWh', 'WindSpeed_m/s', 'Market_Revenue']].copy()
#df_hist.columns = ['hour', 'SpotPrice_€/MWh', 'ActualGeneration_MWh', 'Wind_Speed_m/s', 'Market_Revenue_€']

# Select and rename columns for simulated data (drop the original 'hour' and 'month' columns)
#df_sim = df_sim[['hour', 'SpotPrice', 'Dispatched_MWh', 'WindSpeed_m/s', 'Market_Revenue']].copy()
#df_sim.columns = ['hour', 'SpotPrice_€/MWh', 'ActualGeneration_MWh', 'Wind_Speed_m/s', 'Market_Revenue_€']

#Rename columns for historical data
for col in df_hist.columns:
    if 'spot_price_eur_per_mwh' in col:
        df_hist.rename(columns={col: 'SpotPrice_€/MWh'}, inplace=True)
    if 'wind_speed_hub' in col:
        df_hist.rename(columns={col: 'Wind_Speed_m/s'}, inplace=True)
    if 'datetime' in col:
        df_hist.rename(columns={col: 'hours'}, inplace=True)
    if 'wind_dir_deg' in col:
        df_hist.rename(columns={col: 'Wind_Direction_deg'}, inplace=True)
for col in df_sim.columns:
    if 'spot_price_eur_per_mwh' in col:
        df_sim.rename(columns={col: 'SpotPrice_€/MWh'}, inplace=True)
    if 'wind_dir_deg' in col:
        df_sim.rename(columns={col: 'Wind_Direction_deg'}, inplace=True)
    if 'wind_speed_hub' in col:
        df_sim.rename(columns={col: 'Wind_Speed_m/s'}, inplace=True)
    if 'datetime' in col:
        df_sim.rename(columns={col: 'hours'}, inplace=True)
# Ensure numeric price
df_hist["SpotPrice_€/MWh"] = pd.to_numeric(df_hist["SpotPrice_€/MWh"], errors="coerce")
df_sim["SpotPrice_€/MWh"] = pd.to_numeric(df_sim["SpotPrice_€/MWh"], errors="coerce")

# Ensure numeric wind data
df_hist["Wind_Speed_m/s"] = pd.to_numeric(df_hist["Wind_Speed_m/s"], errors="coerce")
df_hist["SpotPrice_€/MWh"] = pd.to_numeric(df_hist["SpotPrice_€/MWh"], errors="coerce")
df_hist["Wind_Direction_deg"] = pd.to_numeric(df_hist["Wind_Direction_deg"], errors="coerce")
#df_hist["Market_Revenue_€"] = pd.to_numeric(df_hist["Market_Revenue_€"], errors="coerce")
df_sim["Wind_Speed_m/s"] = pd.to_numeric(df_sim["Wind_Speed_m/s"], errors="coerce")
df_sim["SpotPrice_€/MWh"] = pd.to_numeric(df_sim["SpotPrice_€/MWh"], errors="coerce")
df_sim["Wind_Direction_deg"] = pd.to_numeric(df_sim["Wind_Direction_deg"], errors="coerce")
#df_sim["Market_Revenue_€"] = pd.to_numeric(df_sim["Market_Revenue_€"], errors="coerce")

print(f"\nHistorical data: {len(df_hist)} rows")
print(f"Simulated data (filtered): {len(df_sim)} rows")
print(f"\nHistorical columns after processing: {df_hist.columns.tolist()}")
print(f"Simulated columns after processing: {df_sim.columns.tolist()}")

# --- Merge, sort, de-duplicate on timestamp ---
df_merged = pd.concat([df_hist, df_sim], ignore_index=True)
df_merged = df_merged.drop_duplicates(subset=["hours"]).sort_values("hours").reset_index(drop=True)

# Optional sanity check
if not df_merged["hours"].is_monotonic_increasing:
    df_merged = df_merged.sort_values("hours").reset_index(drop=True)

# === Save ===
df_merged.to_csv(out_path, index=False)

print("\n" + "="*50)
print("✓ Merged dataset saved:", out_path)
print(f"Total rows: {len(df_merged)}")
print(f"Date range: {df_merged['hours'].min()} to {df_merged['hours'].max()}")
print("="*50)

# ====================================
# DETAILED GAP ANALYSIS
# ====================================
print("\n" + "="*50)
print("=== DETAILED GAP ANALYSIS ===")
print("="*50)

# Calculate time differences between consecutive rows
df_merged_sorted = df_merged.sort_values('hours').reset_index(drop=True)
time_diffs = df_merged_sorted['hours'].diff()

# Find all non-hourly gaps (excluding the first NaT)
non_hourly_mask = (time_diffs != pd.Timedelta(hours=1)) & (time_diffs.notna())
gap_indices = df_merged_sorted[non_hourly_mask].index.tolist()

if len(gap_indices) > 0:
    print(f"\n⚠ Found {len(gap_indices)} gap(s) in the data:\n")
    
    for i, idx in enumerate(gap_indices, 1):
        prev_time = df_merged_sorted.loc[idx-1, 'hours']
        curr_time = df_merged_sorted.loc[idx, 'hours']
        gap_duration = time_diffs.loc[idx]
        
        print(f"Gap #{i}:")
        print(f"  Previous timestamp: {prev_time}")
        print(f"  Next timestamp:     {curr_time}")
        print(f"  Gap duration:       {gap_duration}")
        print(f"  Missing hours:      {gap_duration.total_seconds() / 3600 - 1:.0f}")
        print()
        
        # Show a few rows before and after the gap
        start_idx = max(0, idx - 3)
        end_idx = min(len(df_merged_sorted), idx + 3)
        print(f"  Context (rows {start_idx} to {end_idx}):")
        print(df_merged_sorted.loc[start_idx:end_idx, ['hours', 'SpotPrice_€/MWh', 'Wind_Direction_deg', 'Wind_Speed_m/s']].to_string(index=True))
        print("\n" + "-"*50 + "\n")
    
    # Summary statistics
    print("="*50)
    print("GAP SUMMARY:")
    print(f"Total gaps found: {len(gap_indices)}")
    print(f"Total missing hours: {sum((time_diffs[non_hourly_mask].dt.total_seconds() / 3600) - 1):.0f}")
    print("="*50)
    
    # Create a report of gaps
    gap_report = []
    for idx in gap_indices:
        gap_report.append({
            'Gap_Number': len(gap_report) + 1,
            'Previous_Timestamp': df_merged_sorted.loc[idx-1, 'hours'],
            'Next_Timestamp': df_merged_sorted.loc[idx, 'hours'],
            'Gap_Duration': time_diffs.loc[idx],
            'Missing_Hours': (time_diffs.loc[idx].total_seconds() / 3600) - 1
        })
    
    gap_df = pd.DataFrame(gap_report)
    print("\nGap Report Table:")
    print(gap_df.to_string(index=False))
    
    # Save gap report to CSV
    gap_report_path = out_path.replace('.csv', '_gap_report.csv')
    gap_df.to_csv(gap_report_path, index=False)
    print(f"\n✓ Gap report saved to: {gap_report_path}")
    
else:
    print("\n✓ No gaps found! All data is continuous hourly records.")

# Additional checks
print("\n" + "="*50)
print("=== ADDITIONAL DATA QUALITY CHECKS ===")
print("="*50)
print(f"First timestamp: {df_merged_sorted['hours'].iloc[0]}")
print(f"Last timestamp:  {df_merged_sorted['hours'].iloc[-1]}")
print(f"Total records:   {len(df_merged_sorted)}")

# Calculate expected number of hours
expected_hours = (df_merged_sorted['hours'].iloc[-1] - df_merged_sorted['hours'].iloc[0]).total_seconds() / 3600 + 1
print(f"Expected hours (based on date range): {expected_hours:.0f}")
print(f"Actual records: {len(df_merged_sorted)}")
print(f"Difference: {expected_hours - len(df_merged_sorted):.0f} hours")

# Check for duplicate timestamps
duplicates = df_merged_sorted[df_merged_sorted.duplicated(subset=['hours'], keep=False)]
if len(duplicates) > 0:
    print(f"\n⚠ Warning: Found {len(duplicates)} duplicate timestamps:")
    print(duplicates[['hours', 'SpotPrice_€/MWh', 'Wind_Direction_deg', 'Wind_Speed_m/s']].head(20))
else:
    print("\n✓ No duplicate timestamps found")

print("\n" + "="*50)