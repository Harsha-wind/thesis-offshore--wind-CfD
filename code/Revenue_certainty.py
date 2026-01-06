import pandas as pd
import numpy as np

# =============================================================================
# INPUT PARAMETERS - MODIFY THESE
# =============================================================================

# Input file path - monthly total revenue CSV
input_path = "D:/Thesis_Project/thesis_data/results/DK/FiP/high/Monthly_Market_Revenue_DK_(High).csv"
# clean column names
# Strip whitespace from column names
# Read the monthly revenue data
df = pd.read_csv(input_path)
df.columns = df.columns.str.strip()
df.to_csv(input_path, index=False)

# Output file path - where to save results
output_path = "D:/Thesis_Project/thesis_data/results/DK/Market_Merchant/high/revenue_certainty_DK_high_Market_Merchant.csv"

# Monthly OPEX in EUR
Total_annual_OPEX = 60 # euro/Kw
installed_Capacity = 740 # MW
Annual_OPEX = Total_annual_OPEX * 1000 * installed_Capacity  # Convert to total annual OPEX in EUR
monthly_opex = Annual_OPEX / 12  # Convert to monthly OPEX in EUR/MW

# Mechanism and scenario names (for labeling results)
mechanism_name = " Market Merchant"
scenario_name = "High"

#Rename the column name
for col in df.columns:
    if 'Monthly_Market_Revenue' in col:
        df.rename(columns={col: 'Monthly_Revenue'}, inplace=True)

revenue_column_name = 'Monthly_Revenue'  # Column name for monthly revenue in the CSV


# =============================================================================
#  CALCULATE CASH FLOW
# =============================================================================


# Extract revenue column
monthly_revenue = df[revenue_column_name].values

# Calculate monthly cash flow = revenue - OPEX
monthly_cashflow = monthly_revenue - monthly_opex

# Remove any NaN values
monthly_cashflow_clean = monthly_cashflow[~np.isnan(monthly_cashflow)]


# =============================================================================
# CALCULATE REVENUE CERTAINTY METRICS
# =============================================================================

# Basic statistics
mean_cashflow = np.mean(monthly_cashflow_clean)
median_cashflow = np.median(monthly_cashflow_clean)
std_cashflow = np.std(monthly_cashflow_clean, ddof=1)

# Coefficient of Variation
cv = std_cashflow / mean_cashflow if mean_cashflow != 0 else np.nan

# Range metrics
min_cashflow = np.min(monthly_cashflow_clean)
max_cashflow = np.max(monthly_cashflow_clean)
range_cashflow = max_cashflow - min_cashflow

# Percentiles
q25 = np.percentile(monthly_cashflow_clean, 25)
q75 = np.percentile(monthly_cashflow_clean, 75)
iqr = q75 - q25

# Relative IQR
relative_iqr = iqr / median_cashflow if median_cashflow != 0 else np.nan

# Percentage of months below mean
pct_below_mean = (monthly_cashflow_clean < mean_cashflow).sum() / len(monthly_cashflow_clean) * 100

# Percentage of negative cash flow months
pct_negative = (monthly_cashflow_clean < 0).sum() / len(monthly_cashflow_clean) * 100

# Total number of months
n_months = len(monthly_cashflow_clean)


# =============================================================================
# CREATE RESULTS DATAFRAME
# =============================================================================

results = {
    'Mechanism': [mechanism_name],
    'Scenario': [scenario_name],
    'N_Months': [n_months],
    'Mean_Monthly_Cashflow_EUR': [mean_cashflow],
    'Median_Monthly_Cashflow_EUR': [median_cashflow],
    'Std_Dev_EUR': [std_cashflow],
    'CV': [cv],
    'Min_Cashflow_EUR': [min_cashflow],
    'Max_Cashflow_EUR': [max_cashflow],
    'Range_EUR': [range_cashflow],
    'Q25_EUR': [q25],
    'Q75_EUR': [q75],
    'IQR_EUR': [iqr],
    'Relative_IQR': [relative_iqr],
    'Pct_Months_Below_Mean': [pct_below_mean],
    'Pct_Negative_Months': [pct_negative]
}

results_df = pd.DataFrame(results)


# =============================================================================
# SAVE RESULTS
# =============================================================================

results_df.to_csv(output_path, index=False)

print("Revenue Certainty Analysis Complete")
print("=" * 70)
print(f"Mechanism: {mechanism_name}")
print(f"Scenario: {scenario_name}")
print(f"Months Analyzed: {n_months}")
print("-" * 70)
print(f"Mean Monthly Cashflow: EUR {mean_cashflow:,.0f}")
print(f"Standard Deviation: EUR {std_cashflow:,.0f}")
print(f"Coefficient of Variation (CV): {cv:.3f}")
print(f"Minimum Cashflow: EUR {min_cashflow:,.0f}")
print(f"Maximum Cashflow: EUR {max_cashflow:,.0f}")
print(f"Months Below Mean: {pct_below_mean:.1f}%")
print(f"Negative Cashflow Months: {pct_negative:.1f}%")
print("-" * 70)
print(f"Results saved to: {output_path}")
print("=" * 70)