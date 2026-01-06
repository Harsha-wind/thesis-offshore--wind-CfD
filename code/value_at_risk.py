import pandas as pd
import numpy as np

# =============================================================================
# DOWNSIDE CoV ANALYSIS - Revenue Volatility in Worst 5% Months
# =============================================================================
# INPUT PARAMETERS - MODIFY THESE
# =============================================================================

# Input file path - monthly total revenue CSV
input_path = "C:/Thesis_Project/thesis_data/results/NL/Current/moderate/Monthly_Revenue_with_SDE_Support.csv"

# Output file path - where to save results
output_path = "C:/Thesis_Project/thesis_data/results/NL/Current/moderate/downside_Cov_SDE+_moderate.csv"

# Monthly OPEX in EUR
Total_annual_OPEX = 69  # euro/kW
installed_Capacity = 740  # MW
Annual_OPEX = Total_annual_OPEX * 1000 * installed_Capacity  # Convert to total annual OPEX in EUR
monthly_opex = Annual_OPEX / 12  # Convert to monthly OPEX in EUR

# Mechanism and scenario names (for labeling results)
mechanism_name = "SDE+"
scenario_name = "Moderate"

# =============================================================================
# READ AND CLEAN DATA
# =============================================================================

# Read the monthly revenue data
df = pd.read_csv(input_path)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Rename the column if needed
for col in df.columns:
    if 'Monthly_Total_Revenue' in col:
        df.rename(columns={col: 'Monthly_Revenue'}, inplace=True)

revenue_column_name = 'Monthly_Revenue'  # Column name for monthly revenue in the CSV

# Save cleaned data back (optional - remove if not needed)
df.to_csv(input_path, index=False)

# =============================================================================
# CALCULATE CASH FLOW
# =============================================================================

# Extract revenue column
monthly_revenue = df[revenue_column_name].values

# Calculate monthly cash flow = revenue - OPEX
monthly_cashflow = monthly_revenue - monthly_opex

# Remove any NaN values
monthly_cashflow_clean = monthly_cashflow[~np.isnan(monthly_cashflow)]

# =============================================================================
# CALCULATE OVERALL METRICS (for context)
# =============================================================================

overall_mean = np.mean(monthly_cashflow_clean)
overall_std = np.std(monthly_cashflow_clean, ddof=1)
overall_cov = overall_std / overall_mean if overall_mean != 0 else np.nan
overall_min = np.min(monthly_cashflow_clean)
overall_max = np.max(monthly_cashflow_clean)
total_months = len(monthly_cashflow_clean)

# =============================================================================
# CALCULATE DOWNSIDE METRICS (worst 5%) - SAME LOGIC AS VAR/CVAR
# =============================================================================

# Define confidence level and alpha (consistent with VaR/CVaR code)
confidence_level = 0.95
alpha = 1 - confidence_level  # 0.05 = 5% worst cases

# VaR: 5th percentile using quantile (same as your VaR/CVaR code)
try:
    threshold = np.quantile(monthly_cashflow_clean, alpha, method="higher")
except TypeError:
    threshold = np.quantile(monthly_cashflow_clean, alpha)

# Filter worst 5% - same as CVaR calculation (x <= var_cf)
downside_cashflow = monthly_cashflow_clean[monthly_cashflow_clean <= threshold]

# Calculate downside metrics
downside_mean = downside_cashflow.mean() if downside_cashflow.size > 0 else threshold  # This is CVaR
downside_std = np.std(downside_cashflow, ddof=1)  # Sample std
downside_cov = downside_std / downside_mean if downside_mean != 0 else np.nan
downside_min = downside_cashflow.min()
downside_max = downside_cashflow.max()
n_downside_months = len(downside_cashflow)

# =============================================================================
# PRINT RESULTS
# =============================================================================

print(f"\n{'='*70}")
print(f"DOWNSIDE CoV ANALYSIS")
print(f"Mechanism: {mechanism_name} | Scenario: {scenario_name}")
print(f"Analyzing Worst 5% Performance (95% Confidence Level)")
print(f"{'='*70}\n")

print(f"OVERALL CASH FLOW METRICS")
print(f"-" * 70)
print(f"Mean:                  €{overall_mean:,.0f}")
print(f"Std Dev:               €{overall_std:,.0f}")
print(f"CoV:                   {overall_cov:.3f}")
print(f"Min:                   €{overall_min:,.0f}")
print(f"Max:                   €{overall_max:,.0f}")
print(f"Total Months:          {total_months}")

print(f"\nDOWNSIDE METRICS (Worst 5%)")
print(f"-" * 70)
print(f"VaR (5th %ile):        €{threshold:,.2f}")
print(f"CVaR (mean worst 5%):  €{downside_mean:,.2f}")
print(f"N Downside Months:     {n_downside_months}")
print(f"Downside Std:          €{downside_std:,.0f}")
print(f"Downside CoV:          {downside_cov:.3f}")
print(f"Downside Min:          €{downside_min:,.0f}")
print(f"Downside Max:          €{downside_max:,.0f}")

print(f"\nCOMPARISON")
print(f"-" * 70)
print(f"Overall CoV:           {overall_cov:.3f}")
print(f"Downside CoV:          {downside_cov:.3f}")

if downside_cov < overall_cov:
    print(f"✓ Downside CoV is LOWER → Worst months are more clustered (less volatile)")
else:
    print(f"⚠ Downside CoV is HIGHER → Worst months are more spread out (more volatile)")

# =============================================================================
# SAVE RESULTS
# =============================================================================

# Store results in dictionary
result = {
    'Mechanism': [mechanism_name],
    'Scenario': [scenario_name],
    'Monthly_OPEX_EUR': [monthly_opex],
    'Total_Months': [total_months],
    'Confidence_Level': [confidence_level],
    'Overall_Mean_EUR': [overall_mean],
    'Overall_Std_EUR': [overall_std],
    'Overall_CoV': [overall_cov],
    'Overall_Min_EUR': [overall_min],
    'Overall_Max_EUR': [overall_max],
    'VaR_5pct_EUR': [threshold],
    'CVaR_5pct_EUR': [downside_mean],
    'N_Downside_Months': [n_downside_months],
    'Downside_Std_EUR': [downside_std],
    'Downside_CoV': [downside_cov],
    'Downside_Min_EUR': [downside_min],
    'Downside_Max_EUR': [downside_max]
}

# Convert to DataFrame
results_df = pd.DataFrame(result)

# Save to CSV
results_df.to_csv(output_path, index=False)

print(f"\n{'='*70}")
print(f"✅ Results saved to: {output_path}")
print(f"{'='*70}\n")

print("INTERPRETATION GUIDE")
print("-" * 70)
print("• Downside CoV measures volatility within the worst 5% of months")
print("• Lower Downside CoV = More stable/predictable in bad times")
print("• High Downside CoV = Unpredictable revenue during worst periods")
print("• Banks care about downside stability for debt service coverage")
print("• If Downside CoV < Overall CoV: Less risk concentration in bad periods")
print("• If Downside CoV > Overall CoV: More risk concentration in bad periods")
print("=" * 70)