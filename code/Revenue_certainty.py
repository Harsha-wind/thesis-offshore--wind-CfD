""" This code can be used to calculate the revenue certainty metrics for the support mechanism and the respective price scenatios and case studies
Here the Market Merchat revenue certainity for the high scenario is calcualted, just by changing the input files the others can be calculated"""

import pandas as pd
import numpy as np


input_path = "thesis_data/results/DK/FiP/high/Monthly_Market_Revenue_DK_(High).csv"

output_path = (
    "thesis_data/results/DK/Market_Merchant/high/"
    "revenue_certainty_DK_high_Market_Merchant.csv"
)
#Input Parameters, changed based on the case study (DK/NL)
Total_annual_OPEX = 60   # euro/kW/year
installed_Capacity = 740 # MW
Annual_OPEX = Total_annual_OPEX * 1000 * installed_Capacity
monthly_opex = Annual_OPEX / 12



# LOAD DATA

df = pd.read_csv(input_path)
df.columns = df.columns.str.strip()

# Rename revenue column to a consistent name
for col in df.columns:
    if "Monthly_Market_Revenue" in col:
        df.rename(columns={col: "Monthly_Revenue"}, inplace=True)
        break

revenue_column_name = "Monthly_Revenue"
if revenue_column_name not in df.columns:
    raise ValueError("Revenue column not found (expected a column containing 'Monthly_Market_Revenue').")


# CASH FLOW SERIES

monthly_revenue = df[revenue_column_name].values.astype(float)
monthly_cashflow = monthly_revenue - monthly_opex
monthly_cashflow_clean = monthly_cashflow[~np.isnan(monthly_cashflow)]

n_months = len(monthly_cashflow_clean)
if n_months == 0:
    raise ValueError("No valid monthly cashflow values after cleaning NaNs.")

# REVENUE CERTAINTY METRICS

mean_cashflow = np.mean(monthly_cashflow_clean)
median_cashflow = np.median(monthly_cashflow_clean)
std_cashflow = np.std(monthly_cashflow_clean, ddof=1)

cv = std_cashflow / mean_cashflow if mean_cashflow != 0 else np.nan

min_cashflow = np.min(monthly_cashflow_clean)
max_cashflow = np.max(monthly_cashflow_clean)
range_cashflow = max_cashflow - min_cashflow



pct_below_mean = (monthly_cashflow_clean < mean_cashflow).sum() / n_months * 100
pct_negative = (monthly_cashflow_clean < 0).sum() / n_months * 100


# SAVE RESULTS (one row)
results_df = pd.DataFrame([{
    "Monthly_OPEX_EUR": monthly_opex,
    "N_Months": n_months,

    "Mean_Monthly_Cashflow_EUR": mean_cashflow,
    "Median_Monthly_Cashflow_EUR": median_cashflow,
    "Std_Dev_EUR": std_cashflow,
    "CV": cv,

    "Min_Cashflow_EUR": min_cashflow,
    "Max_Cashflow_EUR": max_cashflow,
    "Range_EUR": range_cashflow,

    "Pct_Months_Below_Mean": pct_below_mean,
    "Pct_Negative_Months": pct_negative,
}])

results_df.to_csv(output_path, index=False)
