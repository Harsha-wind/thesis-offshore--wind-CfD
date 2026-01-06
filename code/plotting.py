import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================
# LOAD DATA FROM CSV FILES
# ============================================

# Capability CfD data
baseline_cfd_path = "D:/Thesis_Project/thesis_data/results/DK/Cap_Cfd/baseline/Total_Revenue_monthly_Cap_Cfd_DK_baseline.csv"   
df_baseline = pd.read_csv(baseline_cfd_path)
df_baseline.columns = df_baseline.columns.str.strip()

# Tech specific CfD data
techspecific_cfd_path = "D:/Thesis_Project/thesis_data/results/DK/Cap_Cfd/Tech_specific/baseline/Total_Revenue_monthly_Cap_Cfd_DK_baseline_techref.csv"   
df_techspecific = pd.read_csv(techspecific_cfd_path)
df_techspecific.columns = df_techspecific.columns.str.strip()

# FiP data
fip_path = "D:/Thesis_Project/thesis_data/results/DK/FiP/baseline/Total_Revenue_monthly_FiP_DK_baseline.csv"
df_fip = pd.read_csv(fip_path)
df_fip.columns = df_fip.columns.str.strip()



# ============================================
# RENAME COLUMNS FOR CONSISTENCY
# ============================================

for col in df_baseline.columns:
    if 'TotalRevenue' in col:
        df_baseline.rename(columns={col: 'TotalRevenue_Capability_€'}, inplace=True)

for col in df_techspecific.columns:
    if 'TotalRevenue' in col:
        df_techspecific.rename(columns={col: 'TotalRevenue_TechSpecific_€'}, inplace=True)

for col in df_fip.columns:
    if 'TotalRevenue' in col:
        df_fip.rename(columns={col: 'TotalRevenue_FiP_€'}, inplace=True)

for col in df_fip.columns:
    if 'Market_Revenue' in col:
        df_fip.rename(columns={col: 'MarketRevenue_€'}, inplace=True)

# ============================================
# CALCULATE MONTHLY CASH FLOWS
# ============================================

# Monthly OPEX in EUR
Total_annual_OPEX = 60  # euro/KW
installed_Capacity = 740  # MW
Annual_OPEX = Total_annual_OPEX * 1000 * installed_Capacity
monthly_opex = Annual_OPEX / 12

# Calculate cash flow for each scenario (Revenue - OPEX)
monthly_cash_flow = {
    'Capability CfD': df_baseline['TotalRevenue_Capability_€'] - monthly_opex,
    'Tech-Specific CfD': df_techspecific['TotalRevenue_TechSpecific_€'] - monthly_opex,
    'FiP': df_fip['TotalRevenue_FiP_€'] - monthly_opex,
    "Market Revenue": df_fip['MarketRevenue_€'] - monthly_opex  
}

# ============================================
# PREPARE DATE COLUMN
# ============================================

df_baseline['Month'] = pd.to_datetime(df_baseline['Month'])

# ============================================
# CREATE PLOT
# ============================================

fig, ax = plt.subplots(figsize=(14, 8))

# Define colors
colors = {
    'Capability CfD': '#1f77b4',
    'Tech-Specific CfD': '#ff7f0e',
    'FiP': '#2ca02c',
    'Market Revenue': '#d62728'       # Red
}

# Plot each cash flow scenario
for label, cash_flow in monthly_cash_flow.items():
    ax.plot(df_baseline['Month'], 
            cash_flow, 
            label=label,
            color=colors[label],
            linewidth=2.5,
            marker='o',
            markersize=4)

# ============================================
# FORMAT PLOT
# ============================================

ax.set_title('Monthly Cash Flow Comparison: DK\n(Price Scenario - Baseline)', 
             fontsize=16, fontweight='bold')
ax.set_xlabel('Month', fontsize=13, fontweight='bold')
ax.set_ylabel('Monthly Cash Flow (€)', fontsize=13, fontweight='bold')

# Format x-axis dates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45, ha='right')

# Format y-axis with thousand separators
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

# Add grid
ax.grid(True, alpha=0.3, linestyle='--')

# Add zero line
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Add legend
ax.legend(loc='upper left', fontsize=11)

plt.tight_layout()

# ============================================
# SAVE AS PDF (NO PIXELATION!)
# ============================================

output_path = "D:/Thesis_Project/thesis_data/Plot/monthly_cashflow_comparison_DK_baseline.pdf"
plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')

print("Plot saved as PDF successfully!")
print(f"Location: {output_path}")

plt.show()