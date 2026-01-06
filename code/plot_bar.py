import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# LOAD DATA
# ============================================

INPUT_FILE = "D:/Thesis_Project/thesis_data/results/NL/EP/merged_wind_SP_NL.csv"
df = pd.read_csv(INPUT_FILE)

# Convert datetime column
df['datetime'] = pd.to_datetime(df['datetime'])
df['year'] = df['datetime'].dt.year

# ============================================
# CALCULATE STATISTICS FOR EACH TIME PERIOD
# ============================================

# 1. Full Data
full_data = df['spot_price_eur_per_mwh']
mean_full = full_data.mean()
cv_full = full_data.std() / full_data.mean()

# 2. Excluding 2022
excluding_2022 = df[df['year'] != 2022]['spot_price_eur_per_mwh']
mean_excluding = excluding_2022.mean()
cv_excluding = excluding_2022.std() / excluding_2022.mean()

# 3. 2022 Only
only_2022 = df[df['year'] == 2022]['spot_price_eur_per_mwh']
mean_2022 = only_2022.mean()
cv_2022 = only_2022.std() / only_2022.mean()

# 4. Post-Crisis (2023-2025)
post_crisis = df[df['year'].isin([2023, 2024, 2025])]['spot_price_eur_per_mwh']
mean_post = post_crisis.mean()
cv_post = post_crisis.std() / post_crisis.mean()

# ============================================
# PREPARE DATA FOR PLOTTING
# ============================================

periods = ['Full\nData', 'Excluding\n2022', '2022\nOnly', 'Post-Crisis\n(2023-2025)']
mean_values = [mean_full, mean_excluding, mean_2022, mean_post]
cv_values = [cv_full, cv_excluding, cv_2022, cv_post]
colors = ['#808080', '#2ca02c', '#ff6b6b', '#4169E1']  # Gray, Green, Red, Blue

# ============================================
# PLOT 1: MEAN SPOT PRICE BY TIME PERIOD
# ============================================

fig, ax = plt.subplots(figsize=(10, 7))

bars = ax.bar(periods, mean_values, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on top of bars
for i, (bar, value) in enumerate(zip(bars, mean_values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.1f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Format plot
ax.set_title('Mean Spot Price by Time Period:NL', fontsize=16, fontweight='bold')
ax.set_xlabel('Time Period', fontsize=13, fontweight='bold')
ax.set_ylabel('Mean Price (EUR/MWh)', fontsize=13, fontweight='bold')

# No grid
ax.grid(False)

plt.tight_layout()

# Save as PDF
output_path1 = "D:/Thesis_Project/thesis_data/Plot/mean_spot_price_by_period_NL.pdf"
plt.savefig(output_path1, format='pdf', dpi=300, bbox_inches='tight')
print(f"Plot 1 saved: {output_path1}")

plt.show()

# ============================================
# PLOT 2: COEFFICIENT OF VARIATION BY TIME PERIOD
# ============================================

fig, ax = plt.subplots(figsize=(10, 7))

bars = ax.bar(periods, cv_values, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on top of bars
for i, (bar, value) in enumerate(zip(bars, cv_values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.3f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Format plot
ax.set_title('Coefficient of Variation by Time Period:NL', fontsize=16, fontweight='bold')
ax.set_xlabel('Time Period', fontsize=13, fontweight='bold')
ax.set_ylabel('Coefficient of Variation (CV = σ/μ)', fontsize=13, fontweight='bold')

# Add note at bottom
fig.text(0.5, 0.02, 'Note: Higher CV indicates greater price volatility relative to mean', 
         ha='center', fontsize=10, style='italic')

# No grid
ax.grid(False)

plt.tight_layout()

# Save as PDF
output_path2 = "D:/Thesis_Project/thesis_data/Plot/coefficient_variation_by_period_NL.pdf"
plt.savefig(output_path2, format='pdf', dpi=300, bbox_inches='tight')
print(f"Plot 2 saved: {output_path2}")

plt.show()

# ============================================
# PRINT SUMMARY
# ============================================

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"\n{'Period':<25} {'Mean Price':<15} {'CV':<10}")
print("-"*60)
print(f"{'Full Data':<25} {mean_full:>10.1f} EUR/MWh {cv_full:>10.3f}")
print(f"{'Excluding 2022':<25} {mean_excluding:>10.1f} EUR/MWh {cv_excluding:>10.3f}")
print(f"{'2022 Only':<25} {mean_2022:>10.1f} EUR/MWh {cv_2022:>10.3f}")
print(f"{'Post-Crisis (2023-2025)':<25} {mean_post:>10.1f} EUR/MWh {cv_post:>10.3f}")
print("="*60)