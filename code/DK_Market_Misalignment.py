import pandas as pd
import numpy as np

# ==============================================================================
# DENMARK MARKET MISALIGNMENT ANALYSIS
# Using actual monthly revenue data
# ==============================================================================

print("="*80)
print("DENMARK MARKET MISALIGNMENT ANALYSIS")
print("="*80)
print("Comparing: Current FiP vs Capability CfD")
print("Focus: Monthly strategic behavior and payment structure differences")
print("="*80)

# ==============================================================================
# STEP 1: LOAD DATA
# ==============================================================================

#File path 
input_path_fip = "D:/Thesis_Project/thesis_data/results/DK/FiP/high/Total_Revenue_monthly_FiP_DK_high.csv"
input_path_capcfd = "D:/Thesis_Project/thesis_data/results/DK/Cap_Cfd/Tech_specific/high/Total_Revenue_monthly_Cap_Cfd_DK_high_techref.csv"

# Load data
fip_df = pd.read_csv(input_path_fip)
capcfd_df = pd.read_csv(input_path_capcfd)

print(f"\n✓ Data loaded:")
print(f"  FiP months: {len(fip_df)}")
print(f"  CapCfD months: {len(capcfd_df)}")

# Clean column names
fip_df.columns = fip_df.columns.str.strip()
capcfd_df.columns = capcfd_df.columns.str.strip()

# Rename columns for clarity
for col in fip_df.columns:
    if 'TotalRevenue' in col:
        fip_df.rename(columns={col: 'TotalRevenue_EUR_FiP'}, inplace=True)
    if 'SupportPayment' in col:
        fip_df.rename(columns={col: 'SupportPayment_EUR_FiP'}, inplace=True)
    if 'Actual_Generation_MWh' in col:
        fip_df.rename(columns={col: 'Actual_Generation_MWh_FiP'}, inplace=True)
    if 'Market_Revenue' in col:
        fip_df.rename(columns={col: 'Market_Revenue_EUR_FiP'}, inplace=True)
    if 'ReferencePrice' in col:
        fip_df.rename(columns={col: 'ReferencePrice_EUR_per_MWh_FiP'}, inplace=True)
for col in capcfd_df.columns:
    if 'TotalRevenue' in col:
        capcfd_df.rename(columns={col: 'TotalRevenue_EUR_CapCfD'}, inplace=True)
    if 'SupportPayment' in col:
        capcfd_df.rename(columns={col: 'SupportPayment_EUR_CapCfD'}, inplace=True)
    if 'Capability_Generation_MWh' in col:
        capcfd_df.rename(columns={col: 'CapabilityGeneration_MWh'}, inplace=True)
    if 'Actual_Generation_MWh' in col:
        capcfd_df.rename(columns={col: 'Actual_Generation_MWh_CapCfD'}, inplace=True)
    if 'Market_Revenue' in col:
        capcfd_df.rename(columns={col: 'Market_Revenue_EUR_CapCfD'}, inplace=True)
    if 'ReferencePrice' in col:
        capcfd_df.rename(columns={col: 'ReferencePrice_EUR_per_MWh_CapCfD'}, inplace=True)

# Convert Month to datetime
fip_df['Month'] = pd.to_datetime(fip_df['Month'])
capcfd_df['Month'] = pd.to_datetime(capcfd_df['Month'])

# ==============================================================================
# STEP 2: MERGE AND VALIDATE
# ==============================================================================

print("\n" + "="*80)
print("STEP 1: DATA VALIDATION")
print("="*80)

# Merge datasets
merged = pd.merge(fip_df, capcfd_df, on='Month')

# Verify curtailment behavior is identical
gen_difference = abs(merged['Actual_Generation_MWh_FiP'] - merged['Actual_Generation_MWh_CapCfD']).sum()
print(f"\nActual generation difference: {gen_difference:,.0f} MWh")
print("  ✓ Confirmed: Both mechanisms have same curtailment behavior" if gen_difference < 100 else "  ⚠ Warning: Check data consistency")

# ==============================================================================
# STEP 3: PAYMENT BASIS COMPARISON
# ==============================================================================

print("\n" + "="*80)
print("STEP 2: PAYMENT BASIS COMPARISON")
print("="*80)

avg_actual_gen = merged['Actual_Generation_MWh_FiP'].mean()
avg_capability_gen = merged['CapabilityGeneration_MWh'].mean()
avg_curtailment = avg_capability_gen - avg_actual_gen

print(f"\nMonthly Average Generation:")
print(f"  Potential: {avg_capability_gen:,.0f} MWh")
print(f"  Actual: {avg_actual_gen:,.0f} MWh")
print(f"  Curtailed: {avg_curtailment:,.0f} MWh ({avg_curtailment/avg_capability_gen*100:.1f}%)")

print(f"\nPayment Basis Difference:")
print(f"  FiP: Paid on ACTUAL generation ({avg_actual_gen:,.0f} MWh)")
print(f"  CapCfD: Paid on POTENTIAL ({avg_capability_gen:,.0f} MWh)")
print(f"  → CapCfD counts {avg_curtailment:,.0f} MWh more per month")

# ==============================================================================
# STEP 4: IDENTIFY MONTH TYPES
# ==============================================================================

print("\n" + "="*80)
print("STEP 3: MARKET CONDITIONS")
print("="*80)

# Identify month types based on CapCfD (two-sided mechanism shows true market)
merged['market_support_month'] = merged['SupportPayment_EUR_CapCfD'] > 0
merged['market_clawback_month'] = merged['SupportPayment_EUR_CapCfD'] < 0

# FiP behavior (one-sided)
merged['FiP_receives_support'] = merged['SupportPayment_EUR_FiP'] > 0
merged['FiP_no_support'] = merged['SupportPayment_EUR_FiP'] <= 0

# CapCfD behavior (two-sided)
merged['CapCfD_receives_support'] = merged['SupportPayment_EUR_CapCfD'] > 0
merged['CapCfD_pays_clawback'] = merged['SupportPayment_EUR_CapCfD'] < 0

support_months = merged['market_support_month'].sum()
clawback_months = merged['market_clawback_month'].sum()

print(f"\nMarket Conditions (from CapCfD two-sided behavior):")
print(f"  Low-price months (spot < strike): {support_months}")
print(f"  High-price months (spot > strike): {clawback_months}")

print(f"\nFiP Behavior (One-sided):")
print(f"  Receives support: {merged['FiP_receives_support'].sum()} months")
print(f"  No support: {merged['FiP_no_support'].sum()} months")
print(f"  → Never pays clawback (one-sided mechanism)")

print(f"\nCapCfD Behavior (Two-sided):")
print(f"  Receives support: {merged['CapCfD_receives_support'].sum()} months")
print(f"  Pays clawback: {merged['CapCfD_pays_clawback'].sum()} months")
print(f"  → Balanced risk/reward structure")

# ==============================================================================
# STEP 5: FINANCIAL COMPARISON
# ==============================================================================

print("\n" + "="*80)
print("STEP 4: FINANCIAL COMPARISON")
print("="*80)

# Calculate support payments
fip_support_total = merged[merged['FiP_receives_support']]['SupportPayment_EUR_FiP'].sum()
capcfd_support_total = merged[merged['CapCfD_receives_support']]['SupportPayment_EUR_CapCfD'].sum()
capcfd_clawback_total = merged[merged['CapCfD_pays_clawback']]['SupportPayment_EUR_CapCfD'].sum()  # Already negative

print(f"\nFiP (One-sided):")
print(f"  Support paid: €{fip_support_total:,.0f}")
print(f"  Clawback received: €0")
print(f"  Net government cost: €{fip_support_total:,.0f}")

print(f"\nCapCfD (Two-sided):")
print(f"  Support paid: €{capcfd_support_total:,.0f}")
print(f"  Clawback received: €{abs(capcfd_clawback_total):,.0f}")
print(f"  Net government cost: €{capcfd_support_total + capcfd_clawback_total:,.0f}")

net_cost_difference = fip_support_total - (capcfd_support_total + capcfd_clawback_total)

print(f"\nNet Cost Difference:")
if net_cost_difference > 0:
    print(f"  → FiP costs €{net_cost_difference:,.0f} MORE ({net_cost_difference/fip_support_total*100:.1f}%)")
    print(f"  → CapCfD saves money through clawback mechanism")
else:
    print(f"  → CapCfD costs €{abs(net_cost_difference):,.0f} MORE ({abs(net_cost_difference)/fip_support_total*100:.1f}%)")

# ==============================================================================
# STEP 6: GAMING INCENTIVE ANALYSIS
# ==============================================================================

print("\n" + "="*80)
print("STEP 5: GAMING INCENTIVE ANALYSIS")
print("="*80)

# Calculate generation fraction
merged['gen_fraction'] = merged['Actual_Generation_MWh_FiP'] / merged['CapabilityGeneration_MWh']

support_month_df = merged[merged['market_support_month']].copy()
clawback_month_df = merged[merged['market_clawback_month']].copy()

print(f"\nFiP Gaming Incentive (One-sided):")
if len(support_month_df) > 0:
    avg_gen_support = support_month_df['gen_fraction'].mean()
    print(f"  In support months:")
    print(f"    → Payment depends on ACTUAL generation")
    print(f"    → Incentive: MAXIMIZE generation to maximize payment")
    print(f"    → Avg generation rate: {avg_gen_support*100:.1f}% of potential")

if len(clawback_month_df) > 0:
    avg_gen_clawback = clawback_month_df['gen_fraction'].mean()
    print(f"  In high-price months:")
    print(f"    → No support payment (FiP pays €0)")
    print(f"    → No gaming incentive")
    print(f"    → Avg generation rate: {avg_gen_clawback*100:.1f}% of potential")

print(f"\nCapCfD Gaming Prevention (Two-sided):")
print(f"  In ALL months:")
print(f"    → Payment based on POTENTIAL (fixed)")
print(f"    → Generation volume does NOT affect monthly payment")
print(f"    → No incentive to adjust volumes strategically")

# Quantify gaming potential for FiP
if len(support_month_df) > 0:
    avg_fip_rate = support_month_df['SupportPayment_EUR_FiP'].sum() / support_month_df['Actual_Generation_MWh_FiP'].sum()
    total_gen = merged['Actual_Generation_MWh_FiP'].sum()
    one_percent = total_gen * 0.01
    gaming_value = avg_fip_rate * one_percent
    
    print(f"\nFiP Gaming Potential:")
    print(f"  If generator increases generation by 1% in support months:")
    print(f"    Additional volume: {one_percent:,.0f} MWh")
    print(f"    Support rate: €{avg_fip_rate:.2f}/MWh")
    print(f"    Extra revenue: €{gaming_value:,.0f}")
    print(f"  → This volume gaming is economically rational under FiP")
    print(f"  → This gaming is IMPOSSIBLE under CapCfD (paid on potential)")

# ==============================================================================
# STEP 7: SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("SUMMARY & CONCLUSION")
print("="*80)

print(f"\n1. MECHANISM STRUCTURE:")
print(f"   • FiP: One-sided (support only, no clawback)")
print(f"   • CapCfD: Two-sided (support AND clawback)")

print(f"\n2. PAYMENT BASIS:")
print(f"   • FiP: Paid on {merged['Actual_Generation_MWh_FiP'].sum():,.0f} MWh (actual)")
print(f"   • CapCfD: Paid on {merged['CapabilityGeneration_MWh'].sum():,.0f} MWh (potential)")
print(f"   • Difference: {(merged['CapabilityGeneration_MWh'].sum() - merged['Actual_Generation_MWh_FiP'].sum()):,.0f} MWh")

print(f"\n3. GAMING INCENTIVES:")
print(f"   • FiP: Volume-dependent payment → gaming incentive in support months")
print(f"   • CapCfD: Fixed potential payment → no gaming possible")

print(f"\n4. FINANCIAL IMPACT:")
if net_cost_difference > 0:
    print(f"   • CapCfD saves €{net_cost_difference:,.0f} ({net_cost_difference/fip_support_total*100:.1f}% lower cost)")
    if clawback_months > 0:
        print(f"   • Two-sided structure recoups money in {clawback_months} high-price months")
else:
    print(f"   • CapCfD costs €{abs(net_cost_difference):,.0f} MORE")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\n✓ Capability CfD advantages over FiP in Denmark:")
print("\n  1. REMOVES VOLUME GAMING INCENTIVE")
print("     • FiP payment varies with actual generation")
print("     • CapCfD payment fixed on potential")
print("\n  2. TWO-SIDED RISK STRUCTURE")
print("     • FiP never recoups money (one-sided)")
print("     • CapCfD balances through clawback mechanism")
print("\n  3. BETTER MARKET ALIGNMENT")
print("     • Generators can respond purely to hourly prices")
print("     • No incentive to game monthly settlements")

if net_cost_difference > 0:
    print(f"\n  4. LOWER GOVERNMENT COST")
    print(f"     • Saves €{net_cost_difference:,.0f} through two-sided structure")

print("\n✓ Result: Better market alignment with improved risk-sharing")
print("="*80)

# ==============================================================================
# STEP 8: EXPORT RESULTS
# ==============================================================================

print("\n" + "="*80)
print("EXPORTING RESULTS")
print("="*80)

output_path = "D:/Thesis_Project/thesis_data/results/DK/Market_Misalignment_Analysis_DK_high.csv"
merged.to_csv(output_path, index=False)
print(f"\n✓ Results exported to: {output_path}")
print("\n✓ ANALYSIS COMPLETE!")
print("="*80)