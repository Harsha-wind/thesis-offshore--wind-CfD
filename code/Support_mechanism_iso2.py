import pandas as pd
import numpy as np

# =============================================================================
# INPUT PARAMETERS 
# =============================================================================

# Input file paths
market_revenue_path = "D:/Thesis_Project/thesis_data/results/DK/FiP/baseline/Monthly_market_revenue_DK_(Baseline).csv"
total_revenue_path = "D:/Thesis_Project/thesis_data/results/DK/Cap_Cfd/Tech_specific/baseline/Total_Revenue_monthly_Cap_Cfd_DK_baseline_techref.csv"

# Output file path
output_path = "D:/Thesis_Project/thesis_data/results/DK/Cap_Cfd/Tech_specific/baseline/mechanism_separation_Cap_Cfd_baseline_techref.csv"
# OPEX Parameters
Total_annual_OPEX = 60  # euro/kW
installed_Capacity = 740  # MW
Annual_OPEX = Total_annual_OPEX * 1000 * installed_Capacity
monthly_opex = Annual_OPEX / 12

# Analysis parameters
mechanism_name = "Capability CfD (tech ref)"
scenario_name = "baseline"
confidence_level = 0.95  # For VaR/CVaR (worst 5%)

# =============================================================================
# READ AND CLEAN DATA
# =============================================================================



# Load market revenue (without support)
market_df = pd.read_csv(market_revenue_path)
market_df.columns = market_df.columns.str.strip()

# Load total revenue (market + support)
total_df = pd.read_csv(total_revenue_path)
total_df.columns = total_df.columns.str.strip()

# Find and rename revenue columns
for col in market_df.columns:
    if 'Monthly_Market_Revenue' in col:
        market_df.rename(columns={col: 'Market_Revenue'}, inplace=True)

for col in total_df.columns:
    if 'TotalRevenue' in col:
        total_df.rename(columns={col: 'Total_Revenue'}, inplace=True)


# Extract revenues
market_revenue = market_df['Market_Revenue'].values
total_revenue = total_df['Total_Revenue'].values

# Calculate support payments
support_payment = total_revenue - market_revenue

# =============================================================================
# CALCULATE CASH FLOWS
# =============================================================================

# Market only cashflow
market_cashflow = market_revenue - monthly_opex
market_cashflow_clean = market_cashflow[~np.isnan(market_cashflow)]

# Total cashflow (market + support)
total_cashflow = total_revenue - monthly_opex
total_cashflow_clean = total_cashflow[~np.isnan(total_cashflow)]

# Support only
support_clean = support_payment[~np.isnan(support_payment)]

print(f"Loaded {len(market_cashflow_clean)} months of data")

# =============================================================================
# 1. REVENUE CERTAINTY METRICS - MARKET ONLY
# =============================================================================

# Basic statistics
market_mean = np.mean(market_cashflow_clean)
market_median = np.median(market_cashflow_clean)
market_std = np.std(market_cashflow_clean, ddof=1)
market_cv = market_std / market_mean if market_mean != 0 else np.nan

# Range metrics
market_min = np.min(market_cashflow_clean)
market_max = np.max(market_cashflow_clean)

# Percentiles
market_q25 = np.percentile(market_cashflow_clean, 25)
market_q75 = np.percentile(market_cashflow_clean, 75)
market_iqr = market_q75 - market_q25

# Negative months
market_pct_negative = (market_cashflow_clean < 0).sum() / len(market_cashflow_clean) * 100

# =============================================================================
# 2. REVENUE CERTAINTY METRICS - MARKET + SUPPORT
# =============================================================================

# Basic statistics
total_mean = np.mean(total_cashflow_clean)
total_median = np.median(total_cashflow_clean)
total_std = np.std(total_cashflow_clean, ddof=1)
total_cv = total_std / total_mean if total_mean != 0 else np.nan

# Range metrics
total_min = np.min(total_cashflow_clean)
total_max = np.max(total_cashflow_clean)

# Percentiles
total_q25 = np.percentile(total_cashflow_clean, 25)
total_q75 = np.percentile(total_cashflow_clean, 75)
total_iqr = total_q75 - total_q25

# Negative months
total_pct_negative = (total_cashflow_clean < 0).sum() / len(total_cashflow_clean) * 100

# =============================================================================
# 3. DOWNSIDE PROTECTION - MARKET ONLY
# =============================================================================

alpha = 1 - confidence_level  # 0.05 for worst 5%

# VaR and CVaR - Market Only
try:
    market_var = np.quantile(market_cashflow_clean, alpha, method="higher")
except TypeError:
    market_var = np.quantile(market_cashflow_clean, alpha)

market_downside = market_cashflow_clean[market_cashflow_clean <= market_var]
market_cvar = market_downside.mean()
market_downside_std = np.std(market_downside, ddof=1)
market_downside_cov = market_downside_std / market_cvar if market_cvar != 0 else np.nan
market_n_downside = len(market_downside)

# =============================================================================
# 4. DOWNSIDE PROTECTION - MARKET + SUPPORT
# =============================================================================

# VaR and CVaR - Total
try:
    total_var = np.quantile(total_cashflow_clean, alpha, method="higher")
except TypeError:
    total_var = np.quantile(total_cashflow_clean, alpha)

total_downside = total_cashflow_clean[total_cashflow_clean <= total_var]
total_cvar = total_downside.mean()
total_downside_std = np.std(total_downside, ddof=1)
total_downside_cov = total_downside_std / total_cvar if total_cvar != 0 else np.nan
total_n_downside = len(total_downside)

# =============================================================================
# 5. CALCULATE IMPROVEMENTS
# =============================================================================

# Revenue Certainty Improvements
cv_reduction = ((market_cv - total_cv) / market_cv) * 100 if market_cv != 0 else 0
std_reduction = ((market_std - total_std) / market_std) * 100 if market_std != 0 else 0
mean_increase = ((total_mean - market_mean) / market_mean) * 100 if market_mean != 0 else 0

# Downside Protection Improvements
downside_cov_reduction = ((market_downside_cov - total_downside_cov) / market_downside_cov) * 100 if market_downside_cov != 0 else 0
var_improvement = ((total_var - market_var) / abs(market_var)) * 100 if market_var != 0 else 0
cvar_improvement = ((total_cvar - market_cvar) / abs(market_cvar)) * 100 if market_cvar != 0 else 0

# Support statistics
support_mean = np.mean(support_clean)
support_total = np.sum(support_clean)

# =============================================================================
# PRINT RESULTS
# =============================================================================

print("\n" + "="*70)
print(f"SUPPORT MECHANISM SEPARATION ANALYSIS")
print(f"Mechanism: {mechanism_name} | Scenario: {scenario_name}")
print("="*70)

print("\n1. REVENUE CERTAINTY")
print("-"*70)
print(f"{'Metric':<30} {'Market Only':>15} {'Market+Support':>15} {'Change':>10}")
print("-"*70)
print(f"{'Mean Cashflow (EUR)':<30} {market_mean:>15,.0f} {total_mean:>15,.0f} {mean_increase:>9.1f}%")
print(f"{'Std Dev (EUR)':<30} {market_std:>15,.0f} {total_std:>15,.0f} {std_reduction:>9.1f}%")
print(f"{'CV':<30} {market_cv:>15.4f} {total_cv:>15.4f} {cv_reduction:>9.1f}%")
print(f"{'Min Cashflow (EUR)':<30} {market_min:>15,.0f} {total_min:>15,.0f}")
print(f"{'Max Cashflow (EUR)':<30} {market_max:>15,.0f} {total_max:>15,.0f}")
print(f"{'% Negative Months':<30} {market_pct_negative:>14.1f}% {total_pct_negative:>14.1f}%")

print("\n2. DOWNSIDE PROTECTION (Worst 5%)")
print("-"*70)
print(f"{'Metric':<30} {'Market Only':>15} {'Market+Support':>15} {'Change':>10}")
print("-"*70)
print(f"{'VaR (5th %ile) (EUR)':<30} {market_var:>15,.0f} {total_var:>15,.0f} {var_improvement:>9.1f}%")
print(f"{'CVaR (EUR)':<30} {market_cvar:>15,.0f} {total_cvar:>15,.0f} {cvar_improvement:>9.1f}%")
print(f"{'Downside Std (EUR)':<30} {market_downside_std:>15,.0f} {total_downside_std:>15,.0f}")
print(f"{'Downside CoV':<30} {market_downside_cov:>15.4f} {total_downside_cov:>15.4f} {downside_cov_reduction:>9.1f}%")
print(f"{'N Worst Months':<30} {market_n_downside:>15} {total_n_downside:>15}")

print("\n3. SUPPORT MECHANISM CONTRIBUTION")
print("-"*70)
print(f"Mean Monthly Support Payment:     €{support_mean:>12,.0f}")
print(f"Total Support (all months):       €{support_total:>12,.0f}")
print(f"Support as % of Total Revenue:    {(support_total/np.sum(total_revenue))*100:>11.1f}%")

print("\n4. KEY IMPROVEMENTS")
print("-"*70)
print(f"CV Reduction:                     {cv_reduction:>11.1f}%")
print(f"Downside CoV Reduction:           {downside_cov_reduction:>11.1f}%")
print(f"VaR Improvement:                  {var_improvement:>11.1f}%")
print(f"CVaR Improvement:                 {cvar_improvement:>11.1f}%")

print("\n" + "="*70)

# =============================================================================
# SAVE RESULTS TO CSV
# =============================================================================

results = pd.DataFrame({
    'Mechanism': [mechanism_name],
    'Scenario': [scenario_name],
    'Monthly_OPEX_EUR': [monthly_opex],
    'Total_Months': [len(market_cashflow_clean)],
    
    # Revenue Certainty - Market Only
    'Market_Mean_EUR': [market_mean],
    'Market_Std_EUR': [market_std],
    'Market_CV': [market_cv],
    'Market_Min_EUR': [market_min],
    'Market_Max_EUR': [market_max],
    'Market_Pct_Negative': [market_pct_negative],
    
    # Revenue Certainty - Total
    'Total_Mean_EUR': [total_mean],
    'Total_Std_EUR': [total_std],
    'Total_CV': [total_cv],
    'Total_Min_EUR': [total_min],
    'Total_Max_EUR': [total_max],
    'Total_Pct_Negative': [total_pct_negative],
    
    # Downside Protection - Market Only
    'Market_VaR_5pct_EUR': [market_var],
    'Market_CVaR_EUR': [market_cvar],
    'Market_Downside_Std_EUR': [market_downside_std],
    'Market_Downside_CoV': [market_downside_cov],
    
    # Downside Protection - Total
    'Total_VaR_5pct_EUR': [total_var],
    'Total_CVaR_EUR': [total_cvar],
    'Total_Downside_Std_EUR': [total_downside_std],
    'Total_Downside_CoV': [total_downside_cov],
    
    # Improvements
    'CV_Reduction_Pct': [cv_reduction],
    'Downside_CoV_Reduction_Pct': [downside_cov_reduction],
    'VaR_Improvement_Pct': [var_improvement],
    'CVaR_Improvement_Pct': [cvar_improvement],
    
    # Support Stats
    'Mean_Monthly_Support_EUR': [support_mean],
    'Total_Support_EUR': [support_total]
})

results.to_csv(output_path, index=False)

print(f"\nResults saved to: {output_path}")
print("="*70)