import pandas as pd
import numpy as np
from financial_metrics import (calculate_npv, calculate_discounted_cash_flows,
                                calculate_debt_service, calculate_dscr_metrics)
from plot_style import set_style
import matplotlib.pyplot as plt

set_style()

# ============================================================================
# 1. LOAD REVENUE DATA
# ============================================================================
csv_path = "D:/Thesis_Project/thesis_data/results/NL/Capability/Tech_ref/baseline/Total_Revenue_monthly_Ccfd_NL_baseline_techref.csv"
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# Clean column names
for col in df.columns:
    if 'TotalRevenue' in col:
        df.rename(columns={col: 'TotalRevenue_€'}, inplace=True)

# Parse dates and sort
df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
df = df.sort_values(by='Month').reset_index(drop=True)

print(f"Analysis period: {df['Month'].min()} to {df['Month'].max()}")
print(f"Total months: {len(df)}\n")

# ============================================================================
# 2. PROJECT FINANCIAL PARAMETERS
# ============================================================================
# Project specifications
installed_capacity_mw = 740  # MW
capex_kw = 4023  # €/kW
opex_kw = 69  # €/kW/year

# Calculate costs
capex_per_mw = capex_kw * 1000  # Convert to €/MW
total_capex = capex_per_mw * installed_capacity_mw  # Total CAPEX (€)
annual_opex_total = opex_kw * 1000 * installed_capacity_mw  # Total annual OPEX (€)
monthly_opex = annual_opex_total / 12  # Monthly OPEX (€)

# Discount rate (Source: IEA Task 26, Denmark)
# Using REAL WACC because electricity prices are in real terms (inflation-adjusted)
annual_wacc = 0.0481  # 4.81% per year (pretax REAL WACC) -Netherlands
annual_wacc_tax_adjusted = annual_wacc * (1 - 0.25)  # Adjust for corporate tax (25%)
monthly_wacc = (1 + annual_wacc_tax_adjusted) ** (1/12) - 1

# ============================================================================
# 3. DEBT FINANCING PARAMETERS (Source: IEA Task 26, Netherlands)
# ============================================================================
# Project financing structure
debt_to_total_ratio = 0.70  # 70% debt, 30% equity (IEA Task 26)
debt_amount = total_capex * debt_to_total_ratio
equity_amount = total_capex - debt_amount

# Loan terms (IEA Task 26 parameters)
annual_interest_rate = 0.04  # 4.0% cost of debt (Netherlands) # 4.8% cost of debt (Denmark)
cost_of_equity = 0.13  # 13.0% cost of equity (Netherlands) # 12.8% cost of equity (Denmark)
loan_term_years = 25  # 25-year loan term (standard for offshore wind)

# Note: Loan term is 25 years, but DSCR analysis focuses on support period only
print(f"\nLoan Configuration:")
print(f"  Loan term: {loan_term_years} years (standard offshore wind financing)")
print(f"  Support payment period: {len(df)} months ({len(df)/12:.1f} years)")
print(f"  Analysis approach: Assess 25-year loan serviceability during support period\n")

# Calculate monthly debt service payment
monthly_debt_service = calculate_debt_service(
    debt_amount=debt_amount,
    annual_interest_rate=annual_interest_rate,
    loan_term_years=loan_term_years,
    periods_per_year=12
)

print("=" * 70)
print("PROJECT FINANCIAL STRUCTURE")
print("=" * 70)
print("Source: IEA Task 26 - Financial Inputs for Denmark")
print(f"\nTotal CAPEX: €{total_capex:,.0f}")
print(f"  - Debt (70%): €{debt_amount:,.0f}")
print(f"  - Equity (30%): €{equity_amount:,.0f}")
print(f"\nLoan Terms:")
print(f"  - Interest Rate: {annual_interest_rate:.2%} per year (Cost of Debt)")
print(f"  - Cost of Equity: {cost_of_equity:.2%} per year")
print(f"  - WACC (pretax REAL): {annual_wacc:.2%} per year")
print(f"    Note: Using REAL WACC for real (inflation-adjusted) prices")
print(f"  - Loan Term: {loan_term_years} years")
print(f"  - Monthly Debt Service: €{monthly_debt_service:,.0f}")
print(f"\nAnalysis Scope:")
print(f"  - Support period: {len(df)} months ({len(df)/12:.1f} years)")
print(f"  - DSCR assessed during support period only")
print(f"  - Loan continues beyond support period (years {int(len(df)/12)}-{loan_term_years})")
print(f"\nOperating Costs:")
print(f"  - Monthly OPEX: €{monthly_opex:,.0f}")
print(f"  - Total Monthly Fixed Costs: €{monthly_opex + monthly_debt_service:,.0f}")
print("=" * 70 + "\n")

# ============================================================================
# 4. CALCULATE NET CASH FLOWS
# ============================================================================
monthly_revenues = df['TotalRevenue_€'].values
monthly_net_cash_flows = monthly_revenues - monthly_opex  # Operating cash flow

print("=" * 70)
print("CASH FLOW SUMMARY")
print("=" * 70)
print(f"Average Monthly Revenue: €{monthly_revenues.mean():,.0f}")
print(f"Average Monthly OPEX: €{monthly_opex:,.0f}")
print(f"Average Net Operating Cash Flow: €{monthly_net_cash_flows.mean():,.0f}")
print(f"\nTotal Period:")
print(f"  - Total Revenue: €{monthly_revenues.sum():,.0f}")
print(f"  - Total OPEX: €{monthly_opex * len(df):,.0f}")
print(f"  - Total Net Cash Flow: €{monthly_net_cash_flows.sum():,.0f}")
print("=" * 70 + "\n")

# ============================================================================
# 5. NPV CALCULATION
# ============================================================================
npv = calculate_npv(
    cash_flows=monthly_net_cash_flows,
    discount_rate=monthly_wacc,
    initial_investment=total_capex
)

print("=" * 70)
print("NET PRESENT VALUE (NPV)")
print("=" * 70)
print(f"NPV: €{npv:,.0f}")
if npv > 0:
    print(f"✓ Project creates €{npv:,.0f} in value")
else:
    print(f"✗ Project destroys €{abs(npv):,.0f} in value")
print("=" * 70 + "\n")

# ============================================================================
# 6. DSCR ANALYSIS (Financial Risk Assessment)
# ============================================================================
dscr_results = calculate_dscr_metrics(
    net_cash_flows=monthly_net_cash_flows,
    debt_service_payment=monthly_debt_service
)

print("=" * 70)
print("DSCR ANALYSIS - SUPPORT PAYMENT PERIOD")
print("=" * 70)
print(f"Analysis Period: {len(df)} months ({len(df)/12:.1f} years)")
print(f"Loan Term: {loan_term_years} years (debt service based on full loan term)")
print(f"Approach: Assess ability to service 25-year loan during support period")
print(f"\nAverage DSCR: {dscr_results['mean_dscr']:.2f}x")
print(f"Minimum DSCR: {dscr_results['min_dscr']:.2f}x")
print(f"5th Percentile DSCR (worst 5%): {dscr_results['percentile_5_dscr']:.2f}x")
print(f"\nCovenant Breach Risk (DSCR < 1.25):")
print(f"  - Probability: {dscr_results['prob_below_1_25']:.1%}")
print(f"  - Number of months: {dscr_results['months_below_1_25']} out of {len(df)}")
print(f"\nDefault Risk (DSCR < 1.0):")
print(f"  - Probability: {dscr_results['prob_below_1_0']:.1%}")
print(f"\nBankability Assessment (Support Period):")
if dscr_results['mean_dscr'] >= 1.25 and dscr_results['min_dscr'] >= 1.0:
    print("  ✓ Project can service 25-year loan during support period")
    print("  ✓ Meets typical bank requirements (DSCR ≥ 1.25)")
elif dscr_results['mean_dscr'] >= 1.25:
    print("  ⚠ Average DSCR acceptable, but minimum DSCR concerning")
    print("  ⚠ High volatility may require additional credit support")
elif dscr_results['mean_dscr'] >= 1.0:
    print("  ⚠ Below bank covenant requirements but no default risk")
    print("  ⚠ Needs higher strike price or additional revenue support")
else:
    print("  ✗ Cannot service debt during support period")
    print("  ✗ Project not bankable under current terms")
print(f"\nNote: Analysis covers support period only; post-support debt service")
print(f"      (years {int(len(df)/12)+1}-{loan_term_years}) depends on merchant revenues")
print("=" * 70 + "\n")

# ============================================================================
# 7. DETAILED BREAKDOWN FOR VISUALIZATION
# ============================================================================
discounted_monthly_cf = calculate_discounted_cash_flows(
    cash_flows=monthly_net_cash_flows,
    discount_rate=monthly_wacc
)

# Create comprehensive results DataFrame
results_df = pd.DataFrame({
    'month': df['Month'],
    'monthly_revenue': monthly_revenues,
    'monthly_opex': -monthly_opex,
    'net_cash_flow': monthly_net_cash_flows,
    'debt_service': -monthly_debt_service,
    'cash_after_debt': monthly_net_cash_flows - monthly_debt_service,
    'dscr': dscr_results['dscr_values'],
    'discounted_cash_flow': discounted_monthly_cf,
    'dscr_status': ['✓ Pass' if x >= 1.25 else '⚠ Breach' if x >= 1.0 else '✗ Default' 
                    for x in dscr_results['dscr_values']]
})

# Save results
output_path = "D:/Thesis_Project/thesis_data/results/NL/Capability/Tech_ref/baseline/dscr_monthly_analysis_edited_Ccfd_NL_baseline_techref_(new).csv"
results_df.to_csv(output_path, index=False)
print(f"Detailed results saved to: {output_path}\n")

# ============================================================================
# 8. VISUALIZATION: DSCR OVER TIME
# ============================================================================
set_style(body_pt=10, theme="latex_like")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot 1: DSCR over time
ax1.plot(df['Month'], dscr_results['dscr_values'], color='darkblue', linewidth=1.5)
ax1.axhline(y=1.25, color='orange', linestyle='--', linewidth=1, label='Bank Covenant (1.25x)')
ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Default Threshold (1.0x)')
ax1.fill_between(df['Month'], 1.25, dscr_results['dscr_values'], 
                  where=(dscr_results['dscr_values'] >= 1.25), 
                  alpha=0.3, color='green', label='Safe Zone')
ax1.fill_between(df['Month'], 1.0, 1.25, alpha=0.2, color='orange', label='Covenant Breach Zone')
ax1.set_ylabel('DSCR (x)')
ax1.set_title('Debt Service Coverage Ratio - Financial Risk Over Time')
ax1.legend(loc='best', fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: Cash flows
ax2.plot(df['Month'], monthly_net_cash_flows / 1e6, 
         color='darkgreen', linewidth=1.5, label='Net Operating Cash Flow')
ax2.axhline(y=monthly_debt_service / 1e6, color='red', 
            linestyle='--', linewidth=1, label='Debt Service Required')
ax2.fill_between(df['Month'], 0, monthly_net_cash_flows / 1e6, 
                  where=(monthly_net_cash_flows >= monthly_debt_service), 
                  alpha=0.3, color='green')
ax2.set_xlabel('Month')
ax2.set_ylabel('Cash Flow (€M)')
ax2.set_title('Operating Cash Flow vs Debt Service Requirement')
ax2.legend(loc='best', fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = "D:/Thesis_Project/thesis_data/results/NL/Capability/Tech_ref/baseline/dscr_risk_analysis_Ccfd_NL_baseline_techref(new).pdf"
"_baseline_new.pdf"
plt.savefig(plot_path)
print(f"DSCR visualization saved to: {plot_path}")
plt.show()

# ============================================================================
# 9. SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY: FINANCIAL VIABILITY & RISK ASSESSMENT")
print("=" * 70)
print(f"Analysis Scope: {len(df)} months ({len(df)/12:.1f} years) support period")
print(f"Loan Structure: {loan_term_years}-year term (continues beyond support period)")
print(f"\n1. Investment Value (Support Period):")
print(f"   NPV = €{npv:,.0f} → {'POSITIVE (viable)' if npv > 0 else 'NEGATIVE (not viable)'}")
print(f"   Note: NPV based on {len(df)/12:.1f} years of cash flows only")
print(f"\n2. Debt Coverage (During Support Period):")
print(f"   Average DSCR = {dscr_results['mean_dscr']:.2f}x")
print(f"   {'✓ Can service 25-year loan' if dscr_results['mean_dscr'] >= 1.25 else '✗ Insufficient to service 25-year loan'}")
print(f"\n3. Financial Risk (Support Period):")
print(f"   Covenant breach probability = {dscr_results['prob_below_1_25']:.1%}")
print(f"   {'✓ Low risk' if dscr_results['prob_below_1_25'] < 0.05 else '⚠ High volatility risk'}")
print(f"\n4. Overall Assessment:")
if npv > 0 and dscr_results['mean_dscr'] >= 1.25 and dscr_results['prob_below_1_25'] < 0.10:
    print("   ✓✓ Support mechanism provides sufficient revenue certainty")
    print("   ✓ Project can service debt during support period")
elif dscr_results['mean_dscr'] >= 1.25 and dscr_results['prob_below_1_25'] < 0.30:
    print("   ⚠ Average DSCR adequate but high breach probability")
    print("   ⚠ Revenue volatility remains a concern despite support")
elif dscr_results['mean_dscr'] >= 1.0:
    print("   ⚠ Can avoid default but below bank covenant requirements")
    print("   ⚠ Requires strike price optimization or additional support")
else:
    print("   ✗ Cannot service debt even during support period")
    print("   ✗ Mechanism ineffective at current strike price")
print(f"\n5. Post-Support Considerations:")
print(f"   Years {int(len(df)/12)+1}-{loan_term_years} require merchant revenues to service remaining debt")
