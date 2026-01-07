import pandas as pd
import numpy as np


# NPV FUNCTION

def calculate_npv(cash_flows, discount_rate, initial_investment=0.0):
    cash_flows = np.asarray(cash_flows, dtype=float)
    periods = np.arange(1, len(cash_flows) + 1)
    discount_factors = 1 / (1 + discount_rate) ** periods
    discounted_flows = cash_flows * discount_factors
    return float(discounted_flows.sum() - initial_investment)


#LOAD MONTHLY REVENUE DATA

csv_path = (
    "thesis_data/results/DK/Cap_Cfd/Tech_specific/high/"
    "Total_Revenue_monthly_Cap_Cfd_DK_high_techref.csv"
)

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# Rename revenue column to a consistent name
for col in df.columns:
    if "Market_Revenue" in col:
        df.rename(columns={col: "Monthly_Revenue_€"}, inplace=True)
        break

# Sort by month if Month exists
if "Month" in df.columns:
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    df = df.sort_values("Month").reset_index(drop=True)

if "Monthly_Revenue_€" not in df.columns:
    raise ValueError("Revenue column not found (expected a column containing 'Market_Revenue').")


# FINANCIAL PARAMETERS (based on the case study update it)

installed_capacity_mw = 740  # MW
capex_kw = 3395              # €/kW
opex_kw = 60                 # €/kW/year

debt_ratio = 0.7
cost_of_debt = 0.048
annual_inflation = 0.018
tax_rate = 0.22

annual_wacc = 0.0526  # pre-tax REAL WACC 

real_cost_of_debt = (1 + cost_of_debt) / (1 + annual_inflation) - 1
tax_benefit = debt_ratio * real_cost_of_debt * tax_rate
WACC = annual_wacc - tax_benefit
monthly_wacc = (1 + WACC) ** (1 / 12) - 1

total_capex = capex_kw * 1000 * installed_capacity_mw
monthly_opex = (opex_kw * 1000 * installed_capacity_mw) / 12


# NET CASH FLOWS AND NPV

monthly_revenues = df["Monthly_Revenue_€"].values.astype(float)
monthly_net_cash_flows = monthly_revenues - monthly_opex

npv = calculate_npv(
    cash_flows=monthly_net_cash_flows,
    discount_rate=monthly_wacc,
    initial_investment=total_capex
)

# SAVE SUMMARY 
summary = pd.DataFrame([{
    "Months": len(df),
    "Total_CAPEX_EUR": total_capex,
    "Monthly_OPEX_EUR": monthly_opex,
    "Annual_WACC_pretax_real": annual_wacc,
    "Annual_WACC_after_tax": WACC,
    "Monthly_WACC_after_tax": monthly_wacc,
    "NPV_EUR": npv,
}])

output_path = (
    "thesis_data/results/DK/Cap_Cfd/Tech_specific/high/"
    "NPV_summary_CapCfd_DK_high_techref_tax_adjusted.csv"
)

summary.to_csv(output_path, index=False)
