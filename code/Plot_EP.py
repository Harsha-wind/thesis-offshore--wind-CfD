import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# LOAD DATA
# ============================================

input_path = "D:/Thesis_Project/thesis_data/results/NL/EP/moderate/merged_wind_price_full_moderate.csv"
df = pd.read_csv(input_path)
df.columns = df.columns.str.strip()

# Convert to datetime and sort
df['hours'] = pd.to_datetime(df['hours'])
df = df.sort_values('hours')

OUTPUT_DIR = "D:/Thesis_Project/thesis_data/Plot/"

print(f"Total rows: {len(df)}")
print(f"Date range: {df['hours'].min()} to {df['hours'].max()}")

# ============================================
# RENAME COLUMNS
# ============================================

for col in df.columns:
    if "SpotPrice" in col:
        df.rename(columns={col: 'SpotPrice_€/MWh'}, inplace=True)

# ============================================
# CREATE PLOT
# ============================================

plt.figure(figsize=(16, 6))
plt.plot(df['hours'], df['SpotPrice_€/MWh'], linewidth=0.3, color='orange')

plt.xlabel('Year', fontsize=12, fontweight='bold')
plt.ylabel('Spot Price (€/MWh)', fontsize=12, fontweight='bold')
plt.title('Electricity Spot Prices Moderate Scenario: NL', fontsize=14, fontweight='bold')
plt.tight_layout()

# ============================================
# SAVE AS PDF (VECTOR FORMAT - NO PIXELATION!)
# ============================================

plt.savefig(OUTPUT_DIR + 'spot_prices_full_NL_moderate.pdf', 
            format='pdf', 
            dpi=300, 
            bbox_inches='tight')

print("Plot saved as PDF successfully!")
print(f"Location: {OUTPUT_DIR}spot_prices_full_NL_moderate.pdf")

plt.show()