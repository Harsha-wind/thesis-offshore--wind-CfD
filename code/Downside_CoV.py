import pandas as pd
import numpy as np

# =============================================================================
# DOWNSIDE RISK ANALYSIS - VaR/CVaR + Downside CoV
# =============================================================================

# INPUT PARAMETERS
input_path = "D:/Thesis_Project/thesis_data/results/DK/Cap_Cfd/baseline/Total_Revenue_monthly_Cap_Cfd_DK_baseline.csv"
#output_path = "D:/Thesis_Project/thesis_data/results/DK/Market_Merchant/high/downside_risk_analysis_Market_Merchant_high.csv"

# OPEX Parameters
Total_annual_OPEX = 60  # euro/kW
installed_Capacity = 740  # MW
Annual_OPEX = Total_annual_OPEX * 1000 * installed_Capacity
monthly_opex = Annual_OPEX / 12

# Analysis parameters
mechanism_name = "Capability CfD "
scenario_name = "high"
confidence_level = 0.95  # Analyze worst 5%

# =============================================================================
# READ AND CLEAN DATA
# =============================================================================

df = pd.read_csv(input_path)
df.columns = df.columns.str.strip()

# Find and rename revenue column
for col in df.columns:
    if 'Market_Revenue' in col:
        df.rename(columns={col: 'Monthly_Revenue'}, inplace=True)
        break

# =============================================================================
# CALCULATE CASH FLOW
# =============================================================================

monthly_revenue = df['Monthly_Revenue'].values
monthly_cashflow = monthly_revenue - monthly_opex
monthly_cashflow_clean = monthly_cashflow[~np.isnan(monthly_cashflow)]

# =============================================================================
# OVERALL METRICS
# =============================================================================

overall_mean = np.mean(monthly_cashflow_clean)
overall_std = np.std(monthly_cashflow_clean, ddof=1)
overall_cov = overall_std / overall_mean if overall_mean != 0 else np.nan
total_months = len(monthly_cashflow_clean)

# =============================================================================
# DOWNSIDE RISK METRICS (VaR/CVaR + Downside CoV)
# =============================================================================

alpha = 1 - confidence_level  # 0.05 for worst 5%

# VaR: 5th percentile threshold (using quantile for standard approach)
try:
    var_threshold = np.quantile(monthly_cashflow_clean, alpha, method="higher")
except TypeError:
    var_threshold = np.quantile(monthly_cashflow_clean, alpha)

# Get worst 5% months (all cashflows <= VaR)
downside_cashflow = monthly_cashflow_clean[monthly_cashflow_clean <= var_threshold]

# Calculate downside metrics
cvar = downside_cashflow.mean()  # CVaR = average of worst 5%
downside_std = np.std(downside_cashflow, ddof=1)
downside_cov = downside_std / cvar if cvar != 0 else np.nan
downside_min = downside_cashflow.min()
downside_max = downside_cashflow.max()
n_downside_months = len(downside_cashflow)

# =============================================================================
# PRINT RESULTS
# =============================================================================

print(f"\n{'='*70}")
print(f"DOWNSIDE RISK ANALYSIS - {mechanism_name} ({scenario_name})")
print(f"Analyzing Worst {int(alpha*100)}% of Months (Confidence Level: {confidence_level*100}%)")
print(f"{'='*70}\n")

print("OVERALL CASH FLOW")
print(f"-" * 70)
print(f"Mean:              €{overall_mean:>15,.0f}")
print(f"Std Dev:           €{overall_std:>15,.0f}")
print(f"CoV:               {overall_cov:>16.3f}")
print(f"Total Months:      {total_months:>16}")

print(f"\nDOWNSIDE RISK METRICS (Worst 5%)")
print(f"-" * 70)
print(f"VaR (5th %ile):    €{var_threshold:>15,.0f}")
print(f"CVaR (mean worst): €{cvar:>15,.0f}")
print(f"N Worst Months:    {n_downside_months:>16}")
print(f"Downside Std:      €{downside_std:>15,.0f}")
print(f"Downside CoV:      {downside_cov:>16.3f}")

print(f"\nRISK ASSESSMENT")
print(f"-" * 70)
print(f"Overall CoV:       {overall_cov:>16.3f}")
print(f"Downside CoV:      {downside_cov:>16.3f}")
print(f"CoV Ratio:         {downside_cov/overall_cov:>16.3f}x")
print(f"Downside Min:          €{downside_min:,.0f}")
print(f"Downside Max:          €{downside_max:,.0f}")
if downside_cov < overall_cov:
    print(f"\n✓ GOOD: Downside CoV < Overall CoV")
    print(f"  → Worst months are MORE PREDICTABLE (clustered together)")
    print(f"  → Lower tail risk - better for financing")
else:
    print(f"\n⚠ CAUTION: Downside CoV > Overall CoV")
    print(f"  → Worst months are MORE VOLATILE (widely spread)")
    print(f"  → Higher tail risk - challenging for debt service")

# =============================================================================
# SAVE RESULTS
# =============================================================================

results = pd.DataFrame({
    'Mechanism': [mechanism_name],
    'Scenario': [scenario_name],
    'Monthly_OPEX_EUR': [monthly_opex],
    'Total_Months': [total_months],
    'Confidence_Level': [confidence_level],
    
    # Overall metrics
    'Overall_Mean_EUR': [overall_mean],
    'Overall_Std_EUR': [overall_std],
    'Overall_CoV': [overall_cov],
    
    # Downside risk metrics
    'VaR_5pct_EUR': [var_threshold],
    'CVaR_5pct_EUR': [cvar],
    'N_Downside_Months': [n_downside_months],
    'Downside_Std_EUR': [downside_std],
    'Downside_CoV': [downside_cov],
    
    # Risk comparison
    'CoV_Ratio_Downside_to_Overall': [downside_cov / overall_cov]
})

#results.to_csv(output_path, index=False)
#print(f"\n{'='*70}")
#print(f"✅ Results saved to: {output_path}")
#print(f"{'='*70}\n")