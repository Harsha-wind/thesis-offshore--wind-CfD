import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# STRIKE PRICE SENSITIVITY: DENMARK TECH-SPECIFIC CfD
# Based on Systematic Bias Analysis
# =============================================================================

print("="*80)
print("STRIKE PRICE ANALYSIS: Denmark Capability-Based CfD")
print("Objective: Demonstrate fiscal neutrality through strike price adjustment")
print("="*80)

# =============================================================================
# PARAMETERS
# =============================================================================

# File paths
monthly_market_revenue_path = "D:/Thesis_Project/thesis_data/results/DK/FiP/high/Monthly_Market_Revenue_DK_(High).csv"
ref_price_path = "D:/Thesis_Project/thesis_data/results/DK/Cap_Cfd/Tech_specific/high/monthly_tech_specific_ref_price_high_dk.csv"
output_path = "D:/Thesis_Project/thesis_data/results/DK/Cap_Cfd/strike_price/strike_sensitivity_DK_high_new.csv"
plot_output_path = "D:/Thesis_Project/thesis_data/results/DK/Cap_Cfd/strike_price/"

# Financial parameters for Denmark
Total_annual_OPEX = 60  # euro/kW
installed_Capacity = 740  # MW

capex_per_kw = 3395  # euro/kW
debt_ratio = 0.7
cost_of_debt = 0.048  # Pre-tax cost of debt
loan_term = 25  # years
contract_years = 13

# Calculate monthly fixed costs
Annual_OPEX = Total_annual_OPEX * 1000 * installed_Capacity
monthly_opex = Annual_OPEX / 12
total_capex = capex_per_kw * 1000 * installed_Capacity
total_debt = total_capex * debt_ratio
annual_debt_service = total_debt * (cost_of_debt * (1 + cost_of_debt)**loan_term) / ((1 + cost_of_debt)**loan_term - 1)
monthly_debt_service = annual_debt_service / 12

# Strike price range: 103.1 (historical) to 70 with 6 evenly distributed points
strike_prices = np.linspace(103.1, 70, 6)
strike_labels = [f"€{sp:.1f}" for sp in strike_prices]




# =============================================================================
# LOAD AND PREPARE DATA - FIXED FOR YYYY-MM FORMAT
# =============================================================================

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

# Load WITHOUT parse_dates - we'll handle the conversion manually
df_monthly = pd.read_csv(monthly_market_revenue_path)
df_ref = pd.read_csv(ref_price_path)

# Strip column names
df_monthly.columns = df_monthly.columns.str.strip()
df_ref.columns = df_ref.columns.str.strip()

# Rename columns
for col in df_monthly.columns:
    if 'Monthly_Market_Revenue' in col:
        df_monthly.rename(columns={col: 'Monthly_Market_Revenue'}, inplace=True)

for col in df_ref.columns:
    if 'tech_specific_ref_price' in col:
        df_ref.rename(columns={col: 'tech_specific_ref_price'}, inplace=True)
    if 'cap_sum' in col:
        df_ref.rename(columns={col: 'Capability_Generation_MWh'}, inplace=True)

print(f"✓ Loaded monthly market revenue: {len(df_monthly)} months")
print(f"✓ Loaded reference prices: {len(df_ref)} months")

# Convert Month to string format YYYY-MM (in case it's not already)
df_monthly['Month'] = df_monthly['Month'].astype(str).str[:7]  # Take first 7 chars: "2019-01"
df_ref['Month'] = df_ref['Month'].astype(str).str[:7]

print(f"\nMonth format after conversion:")
print(f"  df_monthly sample: {df_monthly['Month'].iloc[0]}")
print(f"  df_ref sample: {df_ref['Month'].iloc[0]}")
print(f"  Both dtypes: {df_monthly['Month'].dtype}, {df_ref['Month'].dtype}")

# Merge datasets
df = pd.merge(df_monthly[['Month', 'Monthly_Market_Revenue']], 
              df_ref[['Month', 'tech_specific_ref_price', 'Capability_Generation_MWh']], 
              on='Month', how='inner')

print(f"✓ Merged data: {len(df)} months available for analysis")

# =============================================================================
# STRIKE PRICE ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("ANALYZING STRIKE PRICE SCENARIOS")
print("="*80)

results = []

for i, strike in enumerate(strike_prices):
    # Calculate CfD payments (capability-based)
    df['cfd_payment'] = (strike - df['tech_specific_ref_price']) * df['Capability_Generation_MWh']
    
    # Calculate total revenue and cash flows
    df['total_revenue'] = df['Monthly_Market_Revenue'] + df['cfd_payment']
    df['monthly_cashflow'] = df['total_revenue'] - monthly_opex
    df['dscr'] = df['monthly_cashflow'] / monthly_debt_service
    
    # Generator metrics
    avg_dscr = df['dscr'].mean()
    min_dscr = df['dscr'].min()
    dscr_std = df['dscr'].std()
    
    # Fiscal analysis: Support vs Clawback
    support_payments = df[df['cfd_payment'] > 0]['cfd_payment'].sum()
    clawback_payments = abs(df[df['cfd_payment'] < 0]['cfd_payment'].sum())
    net_support = support_payments - clawback_payments
    
    # Fiscal balance metrics
    balance_ratio = min(support_payments, clawback_payments) / max(support_payments, clawback_payments) if max(support_payments, clawback_payments) > 0 else 0
    
    # Count support and clawback months
    support_months = (df['cfd_payment'] > 0).sum()
    clawback_months = (df['cfd_payment'] < 0).sum()
    
    # Consumer metrics
    total_support_contract = net_support * (contract_years * 12 / len(df))
    consumer_cost_volatility = df['cfd_payment'].std()
    
    # Revenue stability
    revenue_cv = df['total_revenue'].std() / df['total_revenue'].mean()
    
    results.append({
        'Strike_Price': f"€{strike:.1f}",
        'Label': strike_labels[i],
        'DSCR_Avg': avg_dscr,
        'DSCR_Min': min_dscr,
        'DSCR_StdDev': dscr_std,
        'Support_Paid_Million': support_payments / 1e6,
        'Clawback_Received_Million': clawback_payments / 1e6,
        'Net_Fiscal_Cost_Million': net_support / 1e6,
        'Balance_Ratio': balance_ratio,
        'Support_Months': support_months,
        'Clawback_Months': clawback_months,
        'Consumer_Net_Cost_Contract_Million': total_support_contract / 1e6,
        'Consumer_Cost_Volatility_Million': consumer_cost_volatility / 1e6,
        'Revenue_CV': revenue_cv,
        'Bankable': 'Yes' if avg_dscr >= 1.25 and min_dscr >= 1.0 else 'No'
    })
    
    print(f"\n{strike_labels[i]}:")
    print(f"  DSCR (avg/min):        {avg_dscr:.2f} / {min_dscr:.2f}")
    print(f"  Support paid:          €{support_payments/1e6:.1f}M")
    print(f"  Clawback received:     €{clawback_payments/1e6:.1f}M")
    print(f"  Net fiscal cost:       €{net_support/1e6:.1f}M")
    print(f"  Balance ratio:         {balance_ratio:.3f}")
    print(f"  Bankable:              {results[-1]['Bankable']}")

# =============================================================================
# RESULTS DATAFRAME
# =============================================================================

results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)

summary_display = results_df[[
    'Label', 'DSCR_Avg', 'DSCR_Min', 
    'Support_Paid_Million', 'Clawback_Received_Million', 
    'Net_Fiscal_Cost_Million', 'Balance_Ratio', 'Bankable'
]].copy()

summary_display.columns = [
    'Strike Price', 'Avg DSCR', 'Min DSCR',
    'Support (€M)', 'Clawback (€M)', 'Net Cost (€M)', 
    'Balance Ratio', 'Bankable'
]

print("\n" + summary_display.to_string(index=False))

# Find best balanced strike
best_balance_idx = results_df['Balance_Ratio'].idxmax()
best_balance_strike = results_df.loc[best_balance_idx, 'Label']

print(f"\n✓ Most Fiscally Balanced: {best_balance_strike}")
print(f"  Balance Ratio: {results_df.loc[best_balance_idx, 'Balance_Ratio']:.3f}")
print(f"  (1.0 = perfect symmetry, support = clawback)")

# =============================================================================
# VISUALIZATIONS
# =============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Prepare x-axis positions and labels
x_pos = np.arange(len(strike_prices))
colors = ['#D62828', '#F77F00', '#06A77D']

# Plot 1: DSCR Comparison
ax1 = axes[0, 0]
bars1 = ax1.bar(x_pos, results_df['DSCR_Avg'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.axhline(y=1.25, color='red', linestyle='--', linewidth=2, label='Bankability Threshold (1.25x)')
ax1.axhline(y=1.0, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='Minimum Threshold (1.0x)')
ax1.set_ylabel('Average DSCR', fontsize=12, fontweight='bold')
ax1.set_title('Generator Bankability (DSCR)', fontsize=13, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(strike_labels, rotation=15, ha='right')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

# Plot 2: Net Fiscal Cost
ax2 = axes[0, 1]
bars2 = ax2.bar(x_pos, results_df['Net_Fiscal_Cost_Million'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5, label='Zero Net Cost (Fiscal Neutrality)')
ax2.set_ylabel('Net Fiscal Cost (Million €)', fontsize=12, fontweight='bold')
ax2.set_title('Net Fiscal Cost (Support - Clawback)', fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(strike_labels, rotation=15, ha='right')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'€{height:.1f}M', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')

# Plot 3: Support vs Clawback
ax3 = axes[1, 0]
width = 0.35
x_support = x_pos - width/2
x_clawback = x_pos + width/2
bars3a = ax3.bar(x_support, results_df['Support_Paid_Million'], width, 
                 label='Support Paid to Generator', color='#F18F01', alpha=0.8, edgecolor='black')
bars3b = ax3.bar(x_clawback, results_df['Clawback_Received_Million'], width, 
                 label='Clawback from Generator', color='#06A77D', alpha=0.8, edgecolor='black')
ax3.set_ylabel('Amount (Million €)', fontsize=12, fontweight='bold')
ax3.set_title('Fiscal Balance: Support vs Clawback', fontsize=13, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(strike_labels, rotation=15, ha='right')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Balance Ratio
ax4 = axes[1, 1]
bars4 = ax4.bar(x_pos, results_df['Balance_Ratio'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Perfect Symmetry (1.0)')
ax4.set_ylabel('Balance Ratio', fontsize=12, fontweight='bold')
ax4.set_title('Fiscal Symmetry (Support/Clawback Balance)', fontsize=13, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(strike_labels, rotation=15, ha='right')
ax4.set_ylim(0, 1.1)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')
for bar in bars4:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(plot_output_path + 'strike_price_analysis_DK_high.png', dpi=300, bbox_inches='tight')
print(f"✓ Plots saved: {plot_output_path}strike_price_analysis_DK_high.png")

# =============================================================================
# SAVE RESULTS
# =============================================================================

results_df.to_csv(output_path, index=False)
print(f"✓ Results saved: {output_path}")

print("\n" + "="*80)
print("KEY FINDINGS FOR THESIS")
print("="*80)


print(f"   - Historical strike (€{strike_prices[0]:.1f}) → Net cost: €{results_df.loc[0, 'Net_Fiscal_Cost_Million']:.1f}M")
print(f"   - Adjusted strike (€{strike_prices[2]:.1f}) → Net cost: €{results_df.loc[2, 'Net_Fiscal_Cost_Million']:.1f}M")

print(f"\n2. Fiscal Balance Improvement:")
print(f"   - Historical balance ratio: {results_df.loc[0, 'Balance_Ratio']:.3f}")
print(f"   - Adjusted balance ratio:  {results_df.loc[2, 'Balance_Ratio']:.3f}")
print(f"   - Closer to 1.0 = more symmetric (support ≈ clawback)")

print(f"\n3. Generator Bankability:")
all_bankable = all(results_df['Bankable'] == 'Yes')
if all_bankable:
    print(f"   ✓ All strike prices maintain bankability (DSCR ≥ 1.25)")
else:
    print(f"   ⚠ Some strike prices fail bankability requirements")

print(f"\n✅ CONCLUSION:")

print(f"   to achieve superior fiscal symmetry while maintaining bankability")

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETE")
print("="*80)