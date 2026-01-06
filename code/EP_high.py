"""
================================================================================
SCENARIO 3: HIGH - AGGRESSIVE RENEWABLE PENETRATION
================================================================================
Mehta Method: CV kept constant, Mean and Correlation evolve
Starting: Mean=91.01 EUR/MWh, CV=0.646, ρ=-0.207
Target:   Mean=40 EUR/MWh, CV=1, ρ=-0.75
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================

# Input file
INPUT_FILE = "D:/Thesis_Project/thesis_data/results/merged_wind_speed_direction_price_DK.csv"

# Output files
OUTPUT_PRICES = "D:/Thesis_Project/thesis_data/results/DK/EP/high/future_prices_Scenario3_DK_new.csv"
OUTPUT_STATS = "D:/Thesis_Project/thesis_data/results/DK/EP/high/scenario3_high_stats_DK_new.csv"
OUTPUT_PLOT = "D:/Thesis_Project/thesis_data/results/DK/EP/high/scenario3_high_plot_DK_new.png"

# Projection period
START_DATE = "2025-04-01"

END_DATE = "2032-03-31 23:00"

# Exclude anomalous years
EXCLUDE_YEARS = [2022]

# Random seed
RANDOM_SEED = 42

# ============================================================================
# SCENARIO PARAMETERS (MEHTA METHOD)
# ============================================================================

# Historical baseline (from your data, excluding 2022)
HISTORICAL_MEAN = 63.5
HISTORICAL_CV = 0.81          # KEPT CONSTANT (Mehta approach)
HISTORICAL_CORRELATION = -0.299

# Target by 2032 (Mehta high renewable scenario)
TARGET_MEAN_2032 = 40.0        # Aggressive decline (Mehta: 40-50 EUR range)
TARGET_CV_2032 = 1.0         # CONSTANT (no change)
TARGET_CORRELATION_2032 = -0.75  # Strong cannibalization

# Price floor
#PRICE_FLOOR = -100  # EUR/MWh (frequent deep negatives)

# ============================================================================
# SCENARIO INFO
# ============================================================================

SCENARIO_NAME = "High - Aggressive Renewable Penetration"
SCENARIO_DESC = "CV constant, dramatic price decline with strong cannibalization"

print("="*80)
print(f"  {SCENARIO_NAME.upper()}")
print("="*80)
print(f"  {SCENARIO_DESC}")
print("="*80)

# ============================================================================
# STEP 1: LOAD HISTORICAL DATA
# ============================================================================

print("\n[STEP 1] Loading Historical Data...")

df = pd.read_csv(INPUT_FILE)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)
df = df.dropna()

# Filter years
if EXCLUDE_YEARS:
    df = df[~df.index.year.isin(EXCLUDE_YEARS)]
    print(f"  Excluded years: {EXCLUDE_YEARS}")

print(f"  Period: {df.index.min()} to {df.index.max()}")
print(f"  Total hours: {len(df):,}")

# ============================================================================
# STEP 2: DISPLAY PARAMETERS
# ============================================================================

print("\n[STEP 2] Scenario Parameters")
print("-"*80)

print(f"\n  STARTING POINT (2025):")
print(f"    Mean:         {HISTORICAL_MEAN:.2f} EUR/MWh")
print(f"    CV:           {HISTORICAL_CV:.4f}  [CONSTANT]")
print(f"    Correlation:  {HISTORICAL_CORRELATION:.4f}")

print(f"\n  TARGET (2032) - MEHTA HIGH RENEWABLE:")
print(f"    Mean:         {TARGET_MEAN_2032:.2f} EUR/MWh  (Δ: {TARGET_MEAN_2032-HISTORICAL_MEAN:+.2f})")
print(f"    CV:           {TARGET_CV_2032:.4f}  [NO CHANGE]")
print(f"    Correlation:  {TARGET_CORRELATION_2032:.4f}  (Δ: {TARGET_CORRELATION_2032-HISTORICAL_CORRELATION:+.4f})")

print(f"\n  MEHTA PRINCIPLE: CV remains constant at {HISTORICAL_CV:.4f}")
print(f"  Extreme renewable dominance: Dramatic price decline + strong cannibalization")

# ============================================================================
# STEP 3: PREPARE FUTURE TIME SERIES
# ============================================================================

print("\n[STEP 3] Preparing Future Data...")

# Create future dates
future_dates = pd.date_range(START_DATE, END_DATE, freq='H')
n_hours = len(future_dates)

# Tile historical wind data
hist_wind = df['wind_speed_hub'].values
hist_wind_dir = df['wind_dir_deg'].values

n_repeat = int(np.ceil(n_hours / len(hist_wind)))
future_wind = np.tile(hist_wind, n_repeat)[:n_hours]
future_wind_dir = np.tile(hist_wind_dir, n_repeat)[:n_hours]

print(f"  Future period: {START_DATE} to {END_DATE}")
print(f"  Hours: {n_hours:,}")
print(f"  Years: {future_dates.year.min()} to {future_dates.year.max()}")

# ============================================================================
# STEP 4: DEFINE YEARLY EVOLUTION (MEHTA METHOD)
# ============================================================================

print("\n[STEP 4] Yearly Parameter Evolution")
print("-"*80)
print(f"  {'Year':<6} {'Mean (EUR)':<12} {'CV':<10} {'Correlation':<12}")
print("-"*80)

results = pd.DataFrame(index=future_dates)
results['wind_speed_hub'] = future_wind
results['wind_dir_deg'] = future_wind_dir
results['spot_price_eur_per_mwh'] = np.nan

years = range(future_dates.year.min(), future_dates.year.max() + 1)
yearly_params = []

for year in years:
    # Linear progress
    progress = (year - 2025) / (2032 - 2025)
    
    # Mean evolves
    mean = HISTORICAL_MEAN + (TARGET_MEAN_2032 - HISTORICAL_MEAN) * progress
    
    # CV stays CONSTANT (Mehta approach)
    # OLD (wrong):
# NEW (correct):
    cv = HISTORICAL_CV + (TARGET_CV_2032 - HISTORICAL_CV) * progress  # Evolves!

    # Standard deviation = Mean × CV
    std = mean * cv
    
    # Correlation evolves
    rho = HISTORICAL_CORRELATION + (TARGET_CORRELATION_2032 - HISTORICAL_CORRELATION) * progress
    
    yearly_params.append({
        'year': year,
        'mean': mean,
        'cv': cv,
        'std': std,
        'rho': rho
    })
    
    print(f"  {year:<6} {mean:<12.2f} {cv:<10.4f} {rho:<12.4f}")

# ============================================================================
# STEP 5: GENERATE PRICES (MEHTA METHOD)
# ============================================================================

print("\n[STEP 5] Generating Spot Prices (Mehta Method)")
print("-"*80)

np.random.seed(RANDOM_SEED)
yearly_stats = []

for params in yearly_params:
    year = params['year']
    mean = params['mean']
    std = params['std']
    rho = params['rho']
    
    # Get year data
    year_mask = results.index.year == year
    wind = results.loc[year_mask, 'wind_speed_hub'].values
    n = len(wind)
    
    if n == 0:
        continue
    
    # Mehta Method Steps:
    
    # 1. Standardize wind
    wind_mean = wind.mean()
    wind_std = wind.std()
    if wind_std > 0:
        wind_norm = (wind - wind_mean) / wind_std
    else:
        wind_norm = np.zeros(n)
    
    # 2. Generate independent noise
    noise = np.random.randn(n)
    
    # 3. Orthogonalize noise to wind
    if np.dot(wind_norm, wind_norm) > 0:
        proj = np.dot(noise, wind_norm) / np.dot(wind_norm, wind_norm)
        noise_orth = noise - proj * wind_norm
        if noise_orth.std() > 0:
            noise_orth = noise_orth / noise_orth.std()
    else:
        noise_orth = noise
    
    # 4. Create correlated driver
    driver = rho * wind_norm + np.sqrt(max(0, 1 - rho**2)) * noise_orth
    
    # 5. Generate prices: P = μ + σ × driver
    prices = mean + std * driver
    
    # Apply floor
    #prices = np.maximum(prices, PRICE_FLOOR)
    
    # Store
    results.loc[year_mask, 'spot_price_eur_per_mwh'] = prices
    
    # Calculate statistics
    real_mean = prices.mean()
    real_std = prices.std()
    real_cv = real_std / abs(real_mean) if real_mean != 0 else 0
    real_rho = np.corrcoef(prices, wind)[0, 1]
    neg_hours = (prices < 0).sum()
    neg_pct = 100 * neg_hours / n
    
    yearly_stats.append({
        'Year': year,
        'Target_Mean': mean,
        'Realized_Mean': real_mean,
        'Target_CV': params['cv'],
        'Realized_CV': real_cv,
        'Target_Rho': rho,
        'Realized_Rho': real_rho,
        'Negative_Hours': neg_hours,
        'Negative_Pct': neg_pct,
        'Min': prices.min(),
        'Max': prices.max()
    })
    
    print(f"  {year}: Mean={real_mean:.2f}, CV={real_cv:.4f}, ρ={real_rho:.4f}, Neg={neg_pct:.1f}%")

stats_df = pd.DataFrame(yearly_stats)

# ============================================================================
# STEP 6: OVERALL STATISTICS
# ============================================================================

print("\n[STEP 6] Overall Statistics (2025-2036)")
print("="*80)

prices = results['spot_price_eur_per_mwh'].values
winds = results['wind_speed_hub'].values

mean_overall = prices.mean()
std_overall = prices.std()
cv_overall = std_overall / abs(mean_overall)
rho_overall = np.corrcoef(prices, winds)[0, 1]
neg_total = (prices < 0).sum()
neg_pct = 100 * neg_total / len(prices)

print(f"  Mean:            {mean_overall:.2f} EUR/MWh")
print(f"  Std Dev:         {std_overall:.2f} EUR/MWh")
print(f"  CV:              {cv_overall:.4f}")
print(f"  Correlation:     {rho_overall:.4f}")
print(f"  Negative Hours:  {neg_total:,} ({neg_pct:.2f}%)")
print(f"  Range:           [{prices.min():.2f}, {prices.max():.2f}] EUR/MWh")

# Price brackets
print(f"\n  PRICE DISTRIBUTION:")
brackets = {
    'Below -50': (prices < -50).sum(),
    '-50 to 0': ((prices >= -50) & (prices < 0)).sum(),
    '0 to 20': ((prices >= 0) & (prices < 20)).sum(),
    '20 to 40': ((prices >= 20) & (prices < 40)).sum(),
    '40 to 60': ((prices >= 40) & (prices < 60)).sum(),
    'Above 60': (prices >= 60).sum()
}
for bracket, count in brackets.items():
    pct = 100 * count / len(prices)
    print(f"    {bracket:>10}: {count:>8,} hours ({pct:>5.1f}%)")

# ============================================================================
# STEP 7: VISUALIZATION
# ============================================================================

print("\n[STEP 7] Creating Visualization...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Duration Curve with negative emphasis
ax1 = axes[0, 0]
sorted_p = np.sort(prices)[::-1]
duration = np.linspace(0, 100, len(sorted_p))
ax1.plot(duration, sorted_p, 'b-', linewidth=2)
ax1.axhline(mean_overall, color='g', linestyle='--', linewidth=2, label=f'Mean ({mean_overall:.1f})')
ax1.axhline(0, color='r', linestyle='--', linewidth=2, label='Zero')
ax1.fill_between(duration, sorted_p, 0, where=(sorted_p<0), color='red', alpha=0.4, label='Negative')
ax1.set_title("Price Duration Curve - High Renewable", fontweight='bold', fontsize=11)
ax1.set_xlabel("Duration (%)")
ax1.set_ylabel("Price (EUR/MWh)")
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Wind-Price Strong Correlation
ax2 = axes[0, 1]
sample = slice(None, None, 100)
colors = ['red' if p<0 else 'blue' for p in prices[sample]]
ax2.scatter(winds[sample], prices[sample], c=colors, alpha=0.5, s=5)
z = np.polyfit(winds, prices, 1)
p_fit = np.poly1d(z)
x_line = np.linspace(winds.min(), winds.max(), 100)
ax2.plot(x_line, p_fit(x_line), 'k-', linewidth=3, label=f'ρ={rho_overall:.4f}')
ax2.axhline(0, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
ax2.set_title("Strong Wind Cannibalization", fontweight='bold', fontsize=11)
ax2.set_xlabel("Wind Speed (m/s)")
ax2.set_ylabel("Price (EUR/MWh)")
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: Mean Decline & Negative Hours Growth
ax3 = axes[0, 2]
ax3.plot(stats_df['Year'], stats_df['Target_Mean'], 'b-', linewidth=2, marker='o', label='Target')
ax3.plot(stats_df['Year'], stats_df['Realized_Mean'], 'b--', linewidth=2, marker='s', label='Realized')
ax3_twin = ax3.twinx()
ax3_twin.bar(stats_df['Year'], stats_df['Negative_Pct'], alpha=0.4, color='red', label='Neg %')
ax3.set_title("Price Decline & Negative Hours", fontweight='bold', fontsize=11)
ax3.set_xlabel("Year")
ax3.set_ylabel("Mean Price (EUR/MWh)", color='b')
ax3_twin.set_ylabel("Negative Hours (%)", color='r')
ax3.legend(loc='upper left')
ax3_twin.legend(loc='upper right')
ax3.grid(alpha=0.3)

# Plot 4: CV Constant
ax4 = axes[1, 0]
ax4.plot(stats_df['Year'], stats_df['Target_CV'], 'g-', linewidth=3, marker='o', label='Target (Constant)')
ax4.plot(stats_df['Year'], stats_df['Realized_CV'], 'g--', linewidth=2, marker='s', label='Realized')
ax4.axhline(HISTORICAL_CV, color='gray', linestyle=':', alpha=0.5)
ax4.set_title("CV Kept Constant (Mehta Method)", fontweight='bold', fontsize=11)
ax4.set_xlabel("Year")
ax4.set_ylabel("Coefficient of Variation")
ax4.legend()
ax4.grid(alpha=0.3)

# Plot 5: Correlation Evolution
ax5 = axes[1, 1]
ax5.plot(stats_df['Year'], stats_df['Target_Rho'], 'r-', linewidth=3, marker='o', label='Target')
ax5.plot(stats_df['Year'], stats_df['Realized_Rho'], 'r--', linewidth=2, marker='s', label='Realized')
ax5.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax5.fill_between(stats_df['Year'], stats_df['Target_Rho'], 0, alpha=0.3, color='red')
ax5.set_title("Strong Cannibalization Evolution", fontweight='bold', fontsize=11)
ax5.set_xlabel("Year")
ax5.set_ylabel("Wind-Price Correlation (ρ)")
ax5.legend()
ax5.grid(alpha=0.3)

# Plot 6: Price Distribution
ax6 = axes[1, 2]
bins = np.linspace(prices.min(), prices.max(), 50)
n_hist, bins_hist, patches = ax6.hist(prices, bins=bins, alpha=0.7, edgecolor='black', linewidth=0.5)
for i, patch in enumerate(patches):
    patch.set_facecolor('red' if bins_hist[i] < 0 else 'blue')
ax6.axvline(mean_overall, color='g', linestyle='--', linewidth=2, label=f'Mean ({mean_overall:.1f})')
ax6.axvline(0, color='r', linestyle='--', linewidth=2.5, label='Zero')
ax6.set_title("Price Distribution (Many Negatives)", fontweight='bold', fontsize=11)
ax6.set_xlabel("Price (EUR/MWh)")
ax6.set_ylabel("Frequency")
ax6.legend()
ax6.grid(alpha=0.3, axis='y')

plt.suptitle(f"Scenario 3: {SCENARIO_NAME}\n"
             f"Mean: {mean_overall:.1f} EUR/MWh | CV: {cv_overall:.4f} (constant) | "
             f"ρ: {rho_overall:.4f} | Negative: {neg_pct:.1f}%",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_PLOT}")

# ============================================================================
# STEP 8: SAVE RESULTS
# ============================================================================

print("\n[STEP 8] Saving Results...")
results.index.name = 'datetime'



results.to_csv(OUTPUT_PRICES)
print(f"  Prices: {OUTPUT_PRICES}")

stats_df.to_csv(OUTPUT_STATS, index=False)
print(f"  Stats:  {OUTPUT_STATS}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("  SCENARIO 3 COMPLETE")
print("="*80)
print(f"  {SCENARIO_NAME}")
print(f"  Mehta Method: CV constant at {HISTORICAL_CV:.4f}")
print(f"  Evolution: Mean {HISTORICAL_MEAN:.0f}→{TARGET_MEAN_2032:.0f} EUR/MWh, "
      f"ρ {HISTORICAL_CORRELATION:.3f}→{TARGET_CORRELATION_2032:.3f}")
print(f"  Extreme renewable scenario with {neg_pct:.1f}% negative hours")
print(f"  Aligned with Mehta high renewable scenario (40-50 EUR/MWh range)")
print("="*80)