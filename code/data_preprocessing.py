"""
1_data_preprocessing.py
Process historical wind and spot price data
Analyze different time periods and prepare baseline statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# === CONFIGURATION ===
INPUT_PATH = "E:/Thesis_Project/thesis_data/results/merged_wind_speed_direction_price_DK.csv"
OUTPUT_DIR = "E:/Thesis_Project/thesis_data/results/DK/EP/"

# === LOAD AND CLEAN DATA ===
print("="*60)
print("STEP 1: LOADING AND CLEANING HISTORICAL DATA")
print("="*60)

# Load data
# NEW CODE - explicitly convert to datetime
# Load data
df = pd.read_csv(INPUT_PATH)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index("datetime", inplace=True)
# Clean column names
df.columns = df.columns.str.strip()

# Standardize column names
column_mappings = {
    "spot_price_eur_per_mwh": "SpotPrice/MWh",
    "wind_speed_hub": "WindSpeed",
    "wind_dir_deg": "WindDirection_deg"
}

for old_name, new_name in column_mappings.items():
    for col in df.columns:
        if old_name in col and new_name not in df.columns:
            df.rename(columns={col: new_name}, inplace=True)

print(f"Data loaded: {len(df)} hours")
print(f"Period: {df.index.min()} to {df.index.max()}")
print(f"Columns: {list(df.columns)}")

# === ANALYZE DIFFERENT TIME PERIODS ===
print("\n" + "="*60)
print("STEP 2: ANALYZING DIFFERENT TIME PERIODS")
print("="*60)

def calculate_statistics(data):
    """Calculate key statistics for a dataset"""
    stats = {
        "mean_price": data["SpotPrice/MWh"].mean(),
        "std_price": data["SpotPrice/MWh"].std(),
        "cv_price": data["SpotPrice/MWh"].std() / data["SpotPrice/MWh"].mean(),
        "min_price": data["SpotPrice/MWh"].min(),
        "max_price": data["SpotPrice/MWh"].max(),
        "median_price": data["SpotPrice/MWh"].median(),
        "mean_wind": data["WindSpeed"].mean(),
        "std_wind": data["WindSpeed"].std(),
        "correlation": data["SpotPrice/MWh"].corr(data["WindSpeed"]),
        "negative_hours": (data["SpotPrice/MWh"] < 0).sum(),
        "negative_hours_pct": 100 * (data["SpotPrice/MWh"] < 0).sum() / len(data)
    }
    return stats

# Analyze different periods
periods_analysis = {}

# 1. Full dataset
print("\n1. FULL DATASET (including 2022 crisis)")
stats_full = calculate_statistics(df)
periods_analysis["full"] = stats_full
print(f"   Mean price: {stats_full['mean_price']:.2f} EUR/MWh")
print(f"   CV: {stats_full['cv_price']:.3f}")
print(f"   Correlation: {stats_full['correlation']:.3f}")

# 2. Excluding 2022 (crisis year)
print("\n2. EXCLUDING 2022 CRISIS")
#exclude 2022
df_no_2022 = df[df.index.year != 2022]
stats_no_2022 = calculate_statistics(df_no_2022)
periods_analysis["no_2022"] = stats_no_2022
print(f"   Mean price: {stats_no_2022['mean_price']:.2f} EUR/MWh")
print(f"   CV: {stats_no_2022['cv_price']:.3f}")
print(f"   Correlation: {stats_no_2022['correlation']:.3f}")

# 3. Only 2022 (to see crisis impact)
print("\n3. 2022 CRISIS YEAR ONLY")
df_2022 = df[df.index.year == 2022]
if len(df_2022) > 0:
    stats_2022 = calculate_statistics(df_2022)
    periods_analysis["2022_only"] = stats_2022
    print(f"   Mean price: {stats_2022['mean_price']:.2f} EUR/MWh")
    print(f"   CV: {stats_2022['cv_price']:.3f}")
    print(f"   Correlation: {stats_2022['correlation']:.3f}")

# 4. Post-crisis (2023-2025)
print("\n4. POST-CRISIS (2023-2025)")
df_post_crisis = df[df.index.year >= 2023]
stats_post = calculate_statistics(df_post_crisis)
periods_analysis["post_crisis"] = stats_post
print(f"   Mean price: {stats_post['mean_price']:.2f} EUR/MWh")
print(f"   CV: {stats_post['cv_price']:.3f}")
print(f"   Correlation: {stats_post['correlation']:.3f}")

# === SELECT BASELINE PERIOD ===
print("\n" + "="*60)
print("STEP 3: SELECTING BASELINE FOR FUTURE PROJECTIONS")
print("="*60)

# Use data excluding 2022 as baseline
df_baseline = df_no_2022.copy()
baseline_stats = stats_no_2022

print("\nRECOMMENDATION: Using data EXCLUDING 2022 crisis as baseline")
print("Reason: 2022 prices were exceptional and not representative of normal market conditions")
print("\nBASELINE STATISTICS:")
print(f"  Mean price: {baseline_stats['mean_price']:.2f} EUR/MWh")
print(f"  Std dev: {baseline_stats['std_price']:.2f}")
print(f"  CV: {baseline_stats['cv_price']:.3f}")
print(f"  Wind-Price Correlation: {baseline_stats['correlation']:.3f}")
print(f"  Negative hours: {baseline_stats['negative_hours']} ({baseline_stats['negative_hours_pct']:.2f}%)")

# === VISUALIZATION ===
print("\n" + "="*60)
print("STEP 4: CREATING VISUALIZATIONS")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Monthly average prices over time
ax1 = axes[0, 0]
monthly_avg = df.resample('M')['SpotPrice/MWh'].mean()
ax1.plot(monthly_avg.index, monthly_avg.values, linewidth=2)
ax1.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2022-12-31'), 
            alpha=0.2, color='red', label='2022 Crisis')
ax1.axhline(y=baseline_stats['mean_price'], color='g', linestyle='--', 
            alpha=0.5, label=f"Baseline Mean ({baseline_stats['mean_price']:.0f} EUR)")
ax1.set_title("Monthly Average Spot Prices")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price (EUR/MWh)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Price distribution comparison
ax2 = axes[0, 1]
ax2.hist(df_no_2022['SpotPrice/MWh'], bins=50, alpha=0.5, label='Excluding 2022', color='blue', density=True)
if len(df_2022) > 0:
    ax2.hist(df_2022['SpotPrice/MWh'], bins=50, alpha=0.5, label='2022 Only', color='red', density=True)
ax2.set_title("Price Distribution Comparison")
ax2.set_xlabel("Price (EUR/MWh)")
ax2.set_ylabel("Density")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-50, 300)

# Plot 3: Wind vs Price scatter
ax3 = axes[0, 2]
sample = df_baseline.sample(n=min(5000, len(df_baseline)))
ax3.scatter(sample['WindSpeed'], sample['SpotPrice/MWh'], alpha=0.3, s=1)
z = np.polyfit(df_baseline['WindSpeed'], df_baseline['SpotPrice/MWh'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_baseline['WindSpeed'].min(), df_baseline['WindSpeed'].max(), 100)
ax3.plot(x_line, p(x_line), "r-", linewidth=2, label=f"ρ = {baseline_stats['correlation']:.3f}")
ax3.set_title("Wind Speed vs Spot Price (Baseline)")
ax3.set_xlabel("Wind Speed (m/s)")
ax3.set_ylabel("Price (EUR/MWh)")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Comparison of key metrics
ax4 = axes[1, 0]
periods = ['Full Data', 'Excl. 2022', '2022 Only', 'Post-Crisis']
means = [periods_analysis[k]['mean_price'] for k in ['full', 'no_2022', '2022_only', 'post_crisis']]
ax4.bar(periods, means, color=['gray', 'green', 'red', 'blue'])
ax4.set_title("Mean Price by Period")
ax4.set_ylabel("EUR/MWh")
ax4.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(means):
    ax4.text(i, v + 2, f'{v:.0f}', ha='center')

# Plot 5: CV comparison
ax5 = axes[1, 1]
cvs = [periods_analysis[k]['cv_price'] for k in ['full', 'no_2022', '2022_only', 'post_crisis']]
ax5.bar(periods, cvs, color=['gray', 'green', 'red', 'blue'])
ax5.set_title("Coefficient of Variation by Period")
ax5.set_ylabel("CV")
ax5.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(cvs):
    ax5.text(i, v + 0.01, f'{v:.3f}', ha='center')

# Plot 6: Correlation comparison
ax6 = axes[1, 2]
corrs = [periods_analysis[k]['correlation'] for k in ['full', 'no_2022', '2022_only', 'post_crisis']]
ax6.bar(periods, corrs, color=['gray', 'green', 'red', 'blue'])
ax6.set_title("Wind-Price Correlation by Period")
ax6.set_ylabel("Correlation")
ax6.grid(True, alpha=0.3, axis='y')
ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
for i, v in enumerate(corrs):
    ax6.text(i, v + 0.01 if v > 0 else v - 0.02, f'{v:.3f}', ha='center')

plt.suptitle("Historical Data Analysis - Period Comparison", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "historical_analysis_DK.png", dpi=150, bbox_inches='tight')
plt.show()

# === SAVE PROCESSED DATA ===
print("\n" + "="*60)
print("STEP 5: SAVING PROCESSED DATA")
print("="*60)

# Save baseline data
baseline_file = OUTPUT_DIR + "baseline_data_DK.csv"
df_baseline.to_csv(baseline_file)
print(f"Baseline data saved to: {baseline_file}")

# Save statistics
stats_file = OUTPUT_DIR + "baseline_statistics_DK.json"
with open(stats_file, 'w') as f:
    json.dump(baseline_stats, f, indent=4)
print(f"Statistics saved to: {stats_file}")

# Save wind data for sampling
wind_data = df_baseline[['WindSpeed', 'WindDirection_deg']].copy()
wind_data['hour'] = wind_data.index.hour
wind_data['month'] = wind_data.index.month
wind_file = OUTPUT_DIR + "wind_data_for_sampling.csv"
wind_data.to_csv(wind_file)
print(f"Wind data saved to: {wind_file}")

print("\n" + "="*60)
print("PREPROCESSING COMPLETE")
print("="*60)
print("\nSummary:")
print(f"1. Baseline period: Excluding 2022 crisis")
print(f"2. Baseline mean price: {baseline_stats['mean_price']:.2f} EUR/MWh")
print(f"3. Baseline CV: {baseline_stats['cv_price']:.3f}")
print(f"4. Ready for scenario generation with Mehta paper parameters")