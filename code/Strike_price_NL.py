"""Stike Price Sensitivity Analysis for Netherlands Technology-Specific Capability CfD Scheme"""



import pandas as pd
import numpy as np
from pathlib import Path

# PATHS
DATA_DIR = Path("thesis_data/results/NL")

market_revenue_path = DATA_DIR / "Capability/baseline/Monthly_market_revenue_NL_(baseline).csv"
ref_price_path = DATA_DIR / "Capability/Tech_ref/baseline/monthly_tech_specific_ref_price_baseline.csv"

output_dir = DATA_DIR / "Capability/strike_price"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "strike_sensitivity_balance_baseline.csv"

#Parameters
#Replace the DATA_DIR and paths above for other scenarios
# Financial parameters (Netherlands)
OPEX_EUR_PER_KW_YEAR = 69
INSTALLED_CAPACITY_MW = 740

CAPEX_EUR_PER_KW = 4023
DEBT_RATIO = 0.7
COST_OF_DEBT = 0.04
LOAN_TERM_YEARS = 20

CONTRACT_YEARS = 15

strike_prices = [54.5, 64, 74, 84, 94, 103.1]

annual_opex = OPEX_EUR_PER_KW_YEAR * 1000 * INSTALLED_CAPACITY_MW
monthly_opex = annual_opex / 12

total_capex = CAPEX_EUR_PER_KW * 1000 * INSTALLED_CAPACITY_MW
total_debt = total_capex * DEBT_RATIO

annual_debt_service = total_debt * (COST_OF_DEBT * (1 + COST_OF_DEBT) ** LOAN_TERM_YEARS) / (
    (1 + COST_OF_DEBT) ** LOAN_TERM_YEARS - 1
)
monthly_debt_service = annual_debt_service / 12


# LOAD DATA

df_market = pd.read_csv(market_revenue_path)
df_ref = pd.read_csv(ref_price_path)

df_market.columns = df_market.columns.str.strip()
df_ref.columns = df_ref.columns.str.strip()

# Rename columns for consistency
for col in df_market.columns:
    if "Monthly_Market_Revenue" in col:
        df_market.rename(columns={col: "Monthly_Market_Revenue"}, inplace=True)

for col in df_ref.columns:
    if "tech_specific_ref_price" in col:
        df_ref.rename(columns={col: "tech_specific_ref_price"}, inplace=True)
    if "cap_sum" in col:
        df_ref.rename(columns={col: "Capability_Generation_MWh"}, inplace=True)

# Standardise Month format (YYYY-MM) to avoid merge mismatches
df_market["Month"] = df_market["Month"].astype(str).str[:7]
df_ref["Month"] = df_ref["Month"].astype(str).str[:7]

df = pd.merge(
    df_market[["Month", "Monthly_Market_Revenue"]],
    df_ref[["Month", "tech_specific_ref_price", "Capability_Generation_MWh"]],
    on="Month",
    how="inner"
)


# STRIKE PRICE BALANCE ANALYSIS

results = []

for strike in strike_prices:
    df_tmp = df.copy()

    df_tmp["cfd_payment"] = (strike - df_tmp["tech_specific_ref_price"]) * df_tmp["Capability_Generation_MWh"]
    df_tmp["total_revenue"] = df_tmp["Monthly_Market_Revenue"] + df_tmp["cfd_payment"]
    df_tmp["monthly_cashflow"] = df_tmp["total_revenue"] - monthly_opex
    df_tmp["dscr"] = df_tmp["monthly_cashflow"] / monthly_debt_service

    avg_dscr = df_tmp["dscr"].mean()
    min_dscr = df_tmp["dscr"].min()

    support_payments = df_tmp.loc[df_tmp["cfd_payment"] > 0, "cfd_payment"].sum()
    clawback_payments = abs(df_tmp.loc[df_tmp["cfd_payment"] < 0, "cfd_payment"].sum())
    net_support = support_payments - clawback_payments

    balance_ratio = (
        min(support_payments, clawback_payments) / max(support_payments, clawback_payments)
        if max(support_payments, clawback_payments) > 0
        else 0
    )
    abs_net_fiscal_cost = abs(net_support)

    support_months = int((df_tmp["cfd_payment"] > 0).sum())
    clawback_months = int((df_tmp["cfd_payment"] < 0).sum())

    scaling_factor = (CONTRACT_YEARS * 12) / len(df_tmp) if len(df_tmp) > 0 else np.nan
    total_support_contract = net_support * scaling_factor

    consumer_cost_volatility = df_tmp["cfd_payment"].std()

    results.append({
        "Strike_EUR_MWh": strike,
        "Gen_DSCR_Avg": round(avg_dscr, 2),
        "Gen_DSCR_Min": round(min_dscr, 2),
        "Gen_Support_Received_B": round(support_payments / 1e9, 3),
        "Gen_Clawback_Paid_B": round(clawback_payments / 1e9, 3),
        "Net_Fiscal_Cost_B": round(net_support / 1e9, 3),
        "Abs_Net_Fiscal_Cost_B": round(abs_net_fiscal_cost / 1e9, 3),
        "Balance_Ratio": round(balance_ratio, 3),
        "Cons_Net_Cost_Contract_B": round(total_support_contract / 1e9, 2),
        "Cons_Cost_Volatility_M": round(consumer_cost_volatility / 1e6, 2),
        "Support_Months": support_months,
        "Clawback_Months": clawback_months
    })

results_df = pd.DataFrame(results)




# EXPORT RESULTS
results_df.to_csv(output_path, index=False)
