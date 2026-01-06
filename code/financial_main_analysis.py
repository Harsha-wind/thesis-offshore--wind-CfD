import pandas as pd
import numpy as np
from financial_metrics import calculate_npv, calculate_discounted_cash_flows
from plot_style import set_style

set_style()

# ============================================================================
# 1. LOAD REVENUE DATA (Support Payment Duration)
# ============================================================================
csv_path = "D:/Thesis_Project/thesis_data/results/DK/Cap_Cfd/Tech_specific/high/Total_Revenue_monthly_Cap_Cfd_DK_high_techref.csv"
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# Clean column names - rename Monthly_Market_Revenue column
for col in df.columns:
    if 'Market_Revenue' in col:
        df.rename(columns={col: 'Total_monthly_Revenue_mm_€'}, inplace=True)

# Parse dates and sort chronologically
df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
df = df.sort_values(by='Month').reset_index(drop=True)

print(f"Analysis period: {df['Month'].min()} to {df['Month'].max()}")
print(f"Total months in support payment: {len(df)}\n")

# ============================================================================
# 2. SET FINANCIAL PARAMETERS
# ============================================================================
# Project specifications
installed_capacity_mw = 740  # MW
capex_kw = 3395  # €/kW
opex_kw = 60  # €/kW/year
debt_ratio = 0.7
cost_of_debt = 0.048  # Pre-tax cost of debt
cost_of_equity = 0.128  # Pre-tax cost of equity
annual_inflation = 0.018  # Inflation rate
tax_rate = 0.22  # Corporate tax rate

# Discount rate
annual_wacc = 0.0526  # 5.26% per year (pretax REAL WACC) -Denmark , 4.81% -Netherlands (pretax REAL WACC)
real_cost_of_debt = (1 + cost_of_debt) / (1 + annual_inflation) - 1
tax_benefit = debt_ratio * real_cost_of_debt * tax_rate
WACC = annual_wacc -tax_benefit
monthly_wacc = (1 + WACC) ** (1/12) - 1  # Convert to monthly rate

# Calculate total costs
capex_per_mw = capex_kw * 1000  # Convert to €/MW
total_capex = capex_per_mw * installed_capacity_mw  # Total initial investment (€)

opex_per_mw = opex_kw * 1000  # Convert to €/MW/year
annual_opex_total = opex_per_mw * installed_capacity_mw  # Total annual OPEX (€)
monthly_opex = annual_opex_total / 12  # Monthly OPEX (€)

print("=" * 60)
print("FINANCIAL PARAMETERS")
print("=" * 60)
print(f"Installed Capacity: {installed_capacity_mw:,.0f} MW")
print(f"CAPEX per kW: €{capex_kw:,.0f}")
print(f"OPEX per kW per year: €{opex_kw:,.0f}")
print(f"\nTotal CAPEX (Initial Investment): €{total_capex:,.0f}")
print(f"Annual OPEX : €{annual_opex_total:,.0f}")
print(f"Monthly OPEX: €{monthly_opex:,.0f}")
print(f"\nAnnual WACC(pre-tax): {annual_wacc:.2%}")
print(f"Annual WACC(after-tax): {WACC:.2%}")
print(f"Monthly WACC(after-tax): {monthly_wacc:.4%}")
print("=" * 60 + "\n")

# ============================================================================
# 3. CALCULATE NET CASH FLOWS
# ============================================================================
# Extract monthly revenues from the data
monthly_revenues = df['Total_monthly_Revenue_mm_€'].values

# Calculate net cash flows = Revenue - OPEX (for each month)
monthly_net_cash_flows = monthly_revenues - monthly_opex

print("=" * 60)
print("CASH FLOW SUMMARY")
print("=" * 60)
print(f"Total Revenue (all months): €{monthly_revenues.sum():,.0f}")
print(f"Total OPEX (all months): €{monthly_opex * len(df):,.0f}")
print(f"Total Net Cash Flow: €{monthly_net_cash_flows.sum():,.0f}")
print("=" * 60 + "\n")

# ============================================================================
# 4. CALCULATE NPV (During Support Payment Period Only)
# ============================================================================
npv = calculate_npv(
    cash_flows=monthly_net_cash_flows,
    discount_rate=monthly_wacc,
    initial_investment=total_capex
)

print("=" * 60)
print("NET PRESENT VALUE (NPV) RESULTS")
print("=" * 60)
print(f"NPV over support payment period: €{npv:,.0f}")
print(f"\nInterpretation:")
if npv > 0:
    print(f"✓ Positive NPV: The project creates €{npv:,.0f} in value")
    print("  → Investment is financially attractive")
elif npv < 0:
    print(f"✗ Negative NPV: The project loses €{abs(npv):,.0f} in value")
    print("  → Investment may not be financially viable")
else:
    print("○ NPV = 0: The project breaks even (return equals discount rate)")
print("=" * 60 + "\n")

# ============================================================================
# 5. DETAILED BREAKDOWN (Optional: For Visualization)
# ============================================================================
# Calculate discounted cash flows for each period
discounted_monthly_cf = calculate_discounted_cash_flows(
    cash_flows=monthly_net_cash_flows,
    discount_rate=monthly_wacc
)

# Calculate cumulative NPV over time
cumulative_npv = np.cumsum(discounted_monthly_cf) - total_capex

print("=" * 60)
print("ADDITIONAL METRICS")
print("=" * 60)
print(f"Sum of discounted cash flows: €{discounted_monthly_cf.sum():,.0f}")
print(f"Final cumulative NPV: €{cumulative_npv[-1]:,.0f}")
print(f"Average monthly net cash flow: €{monthly_net_cash_flows.mean():,.0f}")
print("=" * 60 + "\n")

# Optional: Save results to CSV for further analysis
results_df = pd.DataFrame({
    'Month': df['Month'],
    'monthly_revenue': monthly_revenues,
    'monthly_opex': -monthly_opex,
    'net_cash_flow': monthly_net_cash_flows,
    'discounted_cash_flow': discounted_monthly_cf,
    'cumulative_npv': cumulative_npv
})

output_path = "D:/Thesis_Project/thesis_data/results/DK/Market_Merchant/high/NPV_monthly_breakdown_Market_Merchant_high_tax_adjusted.csv"
results_df.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")