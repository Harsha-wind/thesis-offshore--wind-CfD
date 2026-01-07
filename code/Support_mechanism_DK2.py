"""Market isolation analysis comparing Market Merchant and the Support Mechanism used in DK
This can be used for other scenarios and the support mechanisms used in the DK case study
Just by changing the input file paths and the output file paths accordingly"""


import pandas as pd
import numpy as np


# INPUTS
market_revenue_path = (
    "thesis_data/results/DK/FiP/baseline/"
    "Monthly_market_revenue_DK_(Baseline).csv"
)

total_revenue_path = (
    "thesis_data/results/DK/Cap_Cfd/Tech_specific/baseline/"
    "Total_Revenue_monthly_Cap_Cfd_DK_baseline_techref.csv"
)

output_path = (
    "thesis_data/results/DK/Cap_Cfd/Tech_specific/baseline/"
    "mechanism_separation_Cap_Cfd_baseline_techref.csv"
)

Total_annual_OPEX = 60  # euro/kW/year
installed_Capacity = 740  # MW
Annual_OPEX = Total_annual_OPEX * 1000 * installed_Capacity
monthly_opex = Annual_OPEX / 12


confidence_level = 0.95

# =============================================================================
# READ AND CLEAN DATA
# =============================================================================
market_df = pd.read_csv(market_revenue_path)
market_df.columns = market_df.columns.str.strip()

total_df = pd.read_csv(total_revenue_path)
total_df.columns = total_df.columns.str.strip()

for col in market_df.columns:
    if "Monthly_Market_Revenue" in col:
        market_df.rename(columns={col: "Market_Revenue"}, inplace=True)
        break

for col in total_df.columns:
    if "TotalRevenue" in col:
        total_df.rename(columns={col: "Total_Revenue"}, inplace=True)
        break

if "Market_Revenue" not in market_df.columns:
    raise ValueError("Market revenue column not found (expected a column containing 'Monthly_Market_Revenue').")

if "Total_Revenue" not in total_df.columns:
    raise ValueError("Total revenue column not found (expected a column containing 'TotalRevenue').")

# Align lengths (row-by-row correspondence)
n = min(len(market_df), len(total_df))
market_revenue = market_df["Market_Revenue"].iloc[:n].values.astype(float)
total_revenue = total_df["Total_Revenue"].iloc[:n].values.astype(float)

support_payment = total_revenue - market_revenue


# CASH FLOWS

market_cashflow = market_revenue - monthly_opex
total_cashflow = total_revenue - monthly_opex

market_cashflow_clean = market_cashflow[~np.isnan(market_cashflow)]
total_cashflow_clean = total_cashflow[~np.isnan(total_cashflow)]
support_clean = support_payment[~np.isnan(support_payment)]


# REVENUE CERTAINTY METRICS

market_mean = np.mean(market_cashflow_clean)
market_median = np.median(market_cashflow_clean)
market_std = np.std(market_cashflow_clean, ddof=1)
market_cv = market_std / market_mean if market_mean != 0 else np.nan
market_min = np.min(market_cashflow_clean)
market_max = np.max(market_cashflow_clean)
market_q25 = np.percentile(market_cashflow_clean, 25)
market_q75 = np.percentile(market_cashflow_clean, 75)
market_iqr = market_q75 - market_q25
market_pct_negative = (market_cashflow_clean < 0).sum() / len(market_cashflow_clean) * 100

total_mean = np.mean(total_cashflow_clean)
total_median = np.median(total_cashflow_clean)
total_std = np.std(total_cashflow_clean, ddof=1)
total_cv = total_std / total_mean if total_mean != 0 else np.nan
total_min = np.min(total_cashflow_clean)
total_max = np.max(total_cashflow_clean)
total_pct_negative = (total_cashflow_clean < 0).sum() / len(total_cashflow_clean) * 100


# DOWNSIDE PROTECTION (VaR/CVaR + Downside CoV)

alpha = 1 - confidence_level

try:
    market_var = np.quantile(market_cashflow_clean, alpha, method="higher")
except TypeError:
    market_var = np.quantile(market_cashflow_clean, alpha)

market_downside = market_cashflow_clean[market_cashflow_clean <= market_var]
market_cvar = market_downside.mean()
market_downside_std = np.std(market_downside, ddof=1) if len(market_downside) > 1 else np.nan
market_downside_cov = market_downside_std / market_cvar if market_cvar != 0 else np.nan

try:
    total_var = np.quantile(total_cashflow_clean, alpha, method="higher")
except TypeError:
    total_var = np.quantile(total_cashflow_clean, alpha)

total_downside = total_cashflow_clean[total_cashflow_clean <= total_var]
total_cvar = total_downside.mean()
total_downside_std = np.std(total_downside, ddof=1) if len(total_downside) > 1 else np.nan
total_downside_cov = total_downside_std / total_cvar if total_cvar != 0 else np.nan


#IMPROVEMENTS 

cv_reduction = ((market_cv - total_cv) / market_cv) * 100 if market_cv != 0 else 0
std_reduction = ((market_std - total_std) / market_std) * 100 if market_std != 0 else 0
mean_increase = ((total_mean - market_mean) / market_mean) * 100 if market_mean != 0 else 0

downside_cov_reduction = (
    ((market_downside_cov - total_downside_cov) / market_downside_cov) * 100
    if market_downside_cov != 0 else 0
)

var_improvement = ((total_var - market_var) / abs(market_var)) * 100 if market_var != 0 else 0
cvar_improvement = ((total_cvar - market_cvar) / abs(market_cvar)) * 100 if market_cvar != 0 else 0

support_mean = np.mean(support_clean)
support_total = np.sum(support_clean)


# SAVE RESULTS 
results = pd.DataFrame([{
    
    "Monthly_OPEX_EUR": monthly_opex,
    "Total_Months": len(market_cashflow_clean),

    "Market_Mean_EUR": market_mean,
    "Market_Std_EUR": market_std,
    "Market_CV": market_cv,
    "Market_Min_EUR": market_min,
    "Market_Max_EUR": market_max,
    "Market_Q25_EUR": market_q25,
    "Market_Q75_EUR": market_q75,
    "Market_IQR_EUR": market_iqr,
    "Market_Pct_Negative": market_pct_negative,

    "Total_Mean_EUR": total_mean,
    "Total_Std_EUR": total_std,
    "Total_CV": total_cv,
    "Total_Min_EUR": total_min,
    "Total_Max_EUR": total_max,
    "Total_Pct_Negative": total_pct_negative,

    "Market_VaR_5pct_EUR": market_var,
    "Market_CVaR_EUR": market_cvar,
    "Market_Downside_Std_EUR": market_downside_std,
    "Market_Downside_CoV": market_downside_cov,

    "Total_VaR_5pct_EUR": total_var,
    "Total_CVaR_EUR": total_cvar,
    "Total_Downside_Std_EUR": total_downside_std,
    "Total_Downside_CoV": total_downside_cov,

    "Mean_Increase_Pct": mean_increase,
    "Std_Reduction_Pct": std_reduction,
    "CV_Reduction_Pct": cv_reduction,
    "Downside_CoV_Reduction_Pct": downside_cov_reduction,
    "VaR_Improvement_Pct": var_improvement,
    "CVaR_Improvement_Pct": cvar_improvement,

    "Mean_Monthly_Support_EUR": support_mean,
    "Total_Support_EUR": support_total,
}])

results.to_csv(output_path, index=False)
