""" The downside risk analysis is quantified by using the Downside CoV, VaR and CVaR metrics.
The below shows for the baseline and flat average capability CfD scheme in Denmark.
Change the input file path for other scenarios and support mechanisms in Dk and also for NL case study"""


""" Also change the financial parameters based on the country, here denmark is shown"""

import pandas as pd
import numpy as np



# INPUT FILE 
input_path = (
    "thesis_data/results/DK/Cap_Cfd/baseline/"
    "Total_Revenue_monthly_Cap_Cfd_DK_baseline.csv"
)

# OUTPUT FILE 
output_path = (
    "thesis_data/results/DK/Cap_Cfd/baseline/"
    "downside_risk_baseline_DK_CapCfd_flat.csv"
)

# OPEX PARAMETERS
Total_annual_OPEX = 60      # euro/kW/year
installed_Capacity = 740   # MW

Annual_OPEX = Total_annual_OPEX * 1000 * installed_Capacity
monthly_opex = Annual_OPEX / 12

confidence_level = 0.95  # worst 5%

# READ AND CLEAN DATA
df = pd.read_csv(input_path)
df.columns = df.columns.str.strip()

# Identify revenue column
revenue_col = None
for col in df.columns:
    if "Market_Revenue" in col:
        revenue_col = col
        break
    if "TotalRevenue" in col:
        revenue_col = col
        break
    if "Monthly_Revenue" in col:
        revenue_col = col
        break

if revenue_col is None:
    raise ValueError("Revenue column not found in input file.")

df.rename(columns={revenue_col: "Monthly_Revenue"}, inplace=True)

# CASH FLOW SERIES

monthly_revenue = df["Monthly_Revenue"].values.astype(float)
monthly_cashflow = monthly_revenue - monthly_opex
monthly_cashflow = monthly_cashflow[~np.isnan(monthly_cashflow)]

total_months = len(monthly_cashflow)

# OVERALL METRICS

overall_mean = np.mean(monthly_cashflow)
overall_std = np.std(monthly_cashflow, ddof=1)
overall_cov = overall_std / overall_mean if overall_mean != 0 else np.nan


# DOWNSIDE RISK METRICS (Worst 5%)

alpha = 1 - confidence_level

try:
    var_threshold = np.quantile(monthly_cashflow, alpha, method="higher")
except TypeError:
    var_threshold = np.quantile(monthly_cashflow, alpha)

downside_cashflow = monthly_cashflow[monthly_cashflow <= var_threshold]
n_downside_months = len(downside_cashflow)

cvar = np.mean(downside_cashflow) if n_downside_months > 0 else np.nan
downside_std = np.std(downside_cashflow, ddof=1) if n_downside_months > 1 else np.nan
downside_cov = downside_std / cvar if cvar not in [0, np.nan] else np.nan

downside_min = np.min(downside_cashflow) if n_downside_months > 0 else np.nan
downside_max = np.max(downside_cashflow) if n_downside_months > 0 else np.nan

cov_ratio = downside_cov / overall_cov if overall_cov not in [0, np.nan] else np.nan

# OUTPUT TABLE

results = pd.DataFrame(
    [{
        
        "Monthly_OPEX_EUR": monthly_opex,
        "Total_Months": total_months,
        "Confidence_Level": confidence_level,

        "Overall_Mean_EUR": overall_mean,
        "Overall_Std_EUR": overall_std,
        "Overall_CoV": overall_cov,

        "VaR_5pct_EUR": var_threshold,
        "CVaR_5pct_EUR": cvar,
        "N_Downside_Months": n_downside_months,
        "Downside_Std_EUR": downside_std,
        "Downside_CoV": downside_cov,

        "CoV_Ratio_Downside_to_Overall": cov_ratio,
        "Downside_Min_EUR": downside_min,
        "Downside_Max_EUR": downside_max,
    }]
)

results.to_csv(output_path, index=False)
