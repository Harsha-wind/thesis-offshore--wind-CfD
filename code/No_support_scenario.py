import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from plot_style import set_style

set_style()

# ============================================================================
# LOAD DATA FROM CSV FILES
# ============================================================================

# High monthly cash flow for Denmark
# This is the baseline capability CfD data
baseline_cfd_path = "D:/Thesis_Project/thesis_data/results/DK/Cap_Cfd/high/Total_Revenue_monthly_Cap_Cfd_DK_high.csv"   
df_baseline = pd.read_csv(baseline_cfd_path)
df_baseline.columns = df_baseline.columns.str.strip()

# This is the tech specific CfD data
techspecific_cfd_path = "D:/Thesis_Project/thesis_data/results/DK/Cap_Cfd/Tech_specific/high/Total_Revenue_monthly_Cap_Cfd_DK_high_techref.csv"   
df_techspecific = pd.read_csv(techspecific_cfd_path)
df_techspecific.columns = df_techspecific.columns.str.strip()

# This is the FiP data
fip_path = "D:/Thesis_Project/thesis_data/results/DK/FiP/high/Total_Revenue_monthly_FiP_DK_high.csv"
df_fip = pd.read_csv(fip_path)
df_fip.columns = df_fip.columns.str.strip()

# ============================================================================
# RENAME COLUMNS FOR CONSISTENCY
# ============================================================================

for col in df_baseline.columns:
    if 'TotalRevenue' in col:
        df_baseline.rename(columns={col: 'TotalRevenue_Capability_€'}, inplace=True)

for col in df_techspecific.columns:
    if 'TotalRevenue' in col:
        df_techspecific.rename(columns={col: 'TotalRevenue_TechSpecific_€'}, inplace=True)

for col in df_fip.columns:
    if 'TotalRevenue' in col:
        df_fip.rename(columns={col: 'TotalRevenue_FiP_€'}, inplace=True)
    if 'Market_Revenue' in col:
        df_fip.rename(columns={col: 'Market_Revenue_DK_€'}, inplace=True)

# ============================================================================
# CALCULATE MONTHLY OPEX AND CASH FLOWS
# ============================================================================

# Monthly OPEX in EUR
Total_annual_OPEX = 60  # euro/KW
installed_Capacity = 740  # MW
Annual_OPEX = Total_annual_OPEX * 1000 * installed_Capacity  # Convert to total annual OPEX in EUR
monthly_opex = Annual_OPEX / 12  # Convert to monthly OPEX in EUR

# Calculate monthly cash flow for each scenario (Revenue - OPEX)
monthly_cash_flow = {
    'Capability CfD': df_baseline['TotalRevenue_Capability_€'] - monthly_opex,
    'Tech-Specific CfD': df_techspecific['TotalRevenue_TechSpecific_€'] - monthly_opex,
    'FiP': df_fip['TotalRevenue_FiP_€'] - monthly_opex,
    'Market Revenue': df_fip['Market_Revenue_DK_€'] - monthly_opex
}

# ============================================================================
# PREPARE DATE COLUMN FOR X-AXIS
# ============================================================================

# Convert Month column to datetime format for proper plotting
# Using the first dataframe (they should all have the same months)
df_baseline['Month'] = pd.to_datetime(df_baseline['Month'])

# ============================================================================
# CREATE THE PLOT
# ============================================================================

# Create figure and axis with a good size
fig, ax = plt.subplots(figsize=(14, 8))

# Define colors for each cash flow scenario
colors = {
    'Capability CfD': '#1f77b4',      # Blue
    'Tech-Specific CfD': '#ff7f0e',   # Orange
    'FiP': '#2ca02c',                 # Green
    'Market Revenue': '#d62728'       # Red
}

# Define line styles for better distinction
linestyles = {
    'Capability CfD': '-',       # Solid line
    'Tech-Specific CfD': '-',   # Solid line
    'FiP': '-',                 # Solid line
    'Market Revenue': '-'        # Solid line
}

# Plot each cash flow scenario
for label, cash_flow in monthly_cash_flow.items():
    ax.plot(df_baseline['Month'], 
            cash_flow, 
            label=label,
            color=colors[label],
            linestyle=linestyles[label],
            linewidth=2.5,
            marker='o',
            markersize=4,
            alpha=0.8)

# ============================================================================
# FORMAT THE PLOT
# ============================================================================

# Add title and axis labels
ax.set_title('Monthly Cash Flow Comparison: Denmark \n(Price Scenario - High)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Month', fontsize=13, fontweight='bold')
ax.set_ylabel('Monthly Cash Flow (€)', fontsize=13, fontweight='bold')

# Format x-axis to show dates nicely
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Show every 6 months
plt.xticks(rotation=45, ha='right')

# Format y-axis with thousand separators
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

# Grid removed for cleaner look
# ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Add horizontal line at y=0 to show break-even point
ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

# Add legend with good positioning
ax.legend(loc='upper left', 
          fontsize=11, 
          framealpha=0.95,
          edgecolor='gray',
          shadow=True)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# ============================================================================
# SAVE AND DISPLAY THE PLOT
# ============================================================================

# Save the figure in high resolution
output_path = "D:/Thesis_Project/thesis_data/Plot/monthly_cashflow_comparison_DK_high.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved successfully to: {output_path}")

# Display the plot
plt.show()

