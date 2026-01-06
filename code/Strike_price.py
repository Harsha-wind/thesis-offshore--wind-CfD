import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PARAMETERS
# =============================================================================

market_revenue_path = "D:/Thesis_Project/thesis_data/results/NL/Capability/baseline/Monthly_market_revenue_NL_(baseline).csv"
ref_price_path = "D:/Thesis_Project/thesis_data/results/NL/Capability/Tech_ref/baseline/monthly_tech_specific_ref_price_baseline.csv"
output_path = "D:/Thesis_Project/thesis_data/results/NL/Capability/strike_price/strike_sensitivity_balance_baseline.csv"
plot_output_path = "D:/Thesis_Project/thesis_data/results/NL/Capability/strike_price/"

Total_annual_OPEX = 69  # euro/kW
installed_Capacity = 740  # MW
capex_per_kw = 4023  # euro/kW
debt_ratio = 0.7
cost_of_debt = 0.04  # Pre-tax cost of debt
loan_term = 20
contract_years = 15

Annual_OPEX = Total_annual_OPEX * 1000 * installed_Capacity
monthly_opex = Annual_OPEX / 12
total_capex = capex_per_kw * 1000 * installed_Capacity
total_debt = total_capex * debt_ratio
annual_debt_service = total_debt * (cost_of_debt * (1 + cost_of_debt)**loan_term) / ((1 + cost_of_debt)**loan_term - 1)
monthly_debt_service = annual_debt_service / 12

# Strike price range: 54.5 (historical) to 103.1 with 6 evenly distributed points
strike_prices = [54.5, 64, 74, 84, 94, 103.1]

# =============================================================================
# LOAD DATA
# =============================================================================

df_market = pd.read_csv(market_revenue_path)
df_ref = pd.read_csv(ref_price_path)

for col in df_market.columns:
    if 'Monthly_Market_Revenue' in col:
        df_market.rename(columns={col: 'Monthly_Market_Revenue'}, inplace=True)
        
for col in df_ref.columns:
    if 'tech_specific_ref_price' in col:
        df_ref.rename(columns={col: 'tech_specific_ref_price'}, inplace=True)
    if 'cap_sum' in col:
        df_ref.rename(columns={col: 'Capability_Generation_MWh'}, inplace=True)

df = pd.merge(df_market[['Month', 'Monthly_Market_Revenue']], 
              df_ref[['Month', 'tech_specific_ref_price', 'Capability_Generation_MWh']], 
              on='Month', how='inner')

# =============================================================================
# STRIKE PRICE BALANCE ANALYSIS
# =============================================================================

results = []

for strike in strike_prices:
    df['cfd_payment'] = (strike - df['tech_specific_ref_price']) * df['Capability_Generation_MWh']
    df['total_revenue'] = df['Monthly_Market_Revenue'] + df['cfd_payment']
    df['monthly_cashflow'] = df['total_revenue'] - monthly_opex
    df['dscr'] = df['monthly_cashflow'] / monthly_debt_service
    
    # Generator metrics
    avg_dscr = df['dscr'].mean()
    min_dscr = df['dscr'].min()
    
    # Support vs clawback analysis
    support_payments = df[df['cfd_payment'] > 0]['cfd_payment'].sum()
    clawback_payments = abs(df[df['cfd_payment'] < 0]['cfd_payment'].sum())
    net_support = support_payments - clawback_payments
    
    balance_ratio = min(support_payments, clawback_payments) / max(support_payments, clawback_payments) if max(support_payments, clawback_payments) > 0 else 0
    abs_net_fiscal_cost = abs(net_support)
    
    support_months = (df['cfd_payment'] > 0).sum()
    clawback_months = (df['cfd_payment'] < 0).sum()
    
    # Consumer metrics
    total_support_15yr = net_support * (contract_years * 12 / len(df))
    consumer_cost_volatility = df['cfd_payment'].std()
    
    results.append({
        'Strike_EUR_MWh': strike,
        'Gen_DSCR_Avg': round(avg_dscr, 2),
        'Gen_DSCR_Min': round(min_dscr, 2),
        'Gen_Support_Received_B': round(support_payments / 1e9, 3),
        'Gen_Clawback_Paid_B': round(clawback_payments / 1e9, 3),
        'Net_Fiscal_Cost_B': round(net_support / 1e9, 3),
        'Abs_Net_Fiscal_Cost_B': round(abs_net_fiscal_cost / 1e9, 3),
        'Balance_Ratio': round(balance_ratio, 3),
        'Cons_Net_Cost_15yr_B': round(total_support_15yr / 1e9, 2),
        'Cons_Cost_Volatility_M': round(consumer_cost_volatility / 1e6, 2),
        'Support_Months': support_months,
        'Clawback_Months': clawback_months
    })

# =============================================================================
# RESULTS
# =============================================================================

results_df = pd.DataFrame(results)

# Find fiscally balanced strike
bankable_strikes = results_df[results_df['Gen_DSCR_Avg'] >= 1.25].copy()

if len(bankable_strikes) > 0:
    fiscal_balance_idx = bankable_strikes['Abs_Net_Fiscal_Cost_B'].idxmin()
    balanced_strike = results_df.loc[fiscal_balance_idx, 'Strike_EUR_MWh']
    
    print(f"Strike Price Range: €{strike_prices[0]}/MWh (historical) to €{strike_prices[-1]}/MWh")
    print(f"Fiscally Balanced Strike Price: €{balanced_strike}/MWh\n")
    print(results_df[['Strike_EUR_MWh', 'Gen_DSCR_Avg', 
                       'Gen_Support_Received_B', 'Gen_Clawback_Paid_B', 
                       'Net_Fiscal_Cost_B', 'Balance_Ratio']].to_string(index=False))
else:
    balanced_strike = None
    print("No bankable strike price found in range")

# =============================================================================
# PLOTS
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: DSCR vs Strike Price
ax1 = axes[0]
ax1.plot(results_df['Strike_EUR_MWh'], results_df['Gen_DSCR_Avg'], 
         'o-', linewidth=2.5, markersize=8, color='#2E86AB', label='Average DSCR')
ax1.axhline(y=1.25, color='red', linestyle='--', linewidth=2, label='Bankability Threshold (1.25x)')
if balanced_strike:
    ax1.axvline(x=balanced_strike, color='green', linestyle=':', linewidth=2, 
                label=f'Balanced Strike (€{balanced_strike}/MWh)')
ax1.set_xlabel('Strike Price (EUR/MWh)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Debt Service Coverage Ratio (DSCR)', fontsize=12, fontweight='bold')
ax1.set_title('Generator Bankability', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, loc='best')
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=10)

# Plot 2: Net Fiscal Cost vs Strike Price
ax2 = axes[1]
ax2.plot(results_df['Strike_EUR_MWh'], results_df['Net_Fiscal_Cost_B'], 
         'o-', linewidth=2.5, markersize=8, color='#D62828', label='Net Fiscal Cost')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Zero Net Cost')
if balanced_strike:
    ax2.axvline(x=balanced_strike, color='green', linestyle=':', linewidth=2, 
                label=f'Balanced Strike (€{balanced_strike}/MWh)')
ax2.set_xlabel('Strike Price (EUR/MWh)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Net Fiscal Cost (Billion EUR)', fontsize=12, fontweight='bold')
ax2.set_title('Net Fiscal Cost (Support - Clawback)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10, loc='best')
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=10)

# Plot 3: Support vs Clawback vs Strike Price
ax3 = axes[2]
width = 3.0
x_pos = results_df['Strike_EUR_MWh']
ax3.bar(x_pos - width/2, results_df['Gen_Support_Received_B'], width, 
        label='Support Paid to Generator', color='#F18F01', alpha=0.8, edgecolor='black')
ax3.bar(x_pos + width/2, results_df['Gen_Clawback_Paid_B'], width, 
        label='Clawback from Generator', color='#06A77D', alpha=0.8, edgecolor='black')
if balanced_strike:
    ax3.axvline(x=balanced_strike, color='green', linestyle=':', linewidth=2, 
                label=f'Balanced Strike (€{balanced_strike}/MWh)')
ax3.set_xlabel('Strike Price (EUR/MWh)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Amount (Billion EUR)', fontsize=12, fontweight='bold')
ax3.set_title('Fiscal Balance: Support vs Clawback', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10, loc='best')
ax3.grid(True, alpha=0.3, axis='y')
ax3.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig(plot_output_path + 'strike_price_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Plots saved: {plot_output_path}strike_price_analysis.png")

# =============================================================================
# SAVE RESULTS
# =============================================================================

results_df.to_csv(output_path, index=False)
print(f"✓ Results saved: {output_path}")