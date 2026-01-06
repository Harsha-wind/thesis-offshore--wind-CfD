import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#File paths
input_path_capability = "D:/Thesis_Project/thesis_data/results/DK/Cap_Cfd/Tech_specific/baseline/Total_Revenue_monthly_Cap_Cfd_DK_baseline_techref.csv"
input_path_fip = "D:/Thesis_Project/thesis_data/results/DK/FiP/baseline/Total_Revenue_monthly_FiP_DK_baseline.csv"
df_capability = pd.read_csv(input_path_capability)
df_fip = pd.read_csv(input_path_fip)
df_capability.columns = df_capability.columns.str.strip()
df_fip.columns = df_fip.columns.str.strip()

#Rename columns for consistency
for col in df_capability.columns:
    if 'TotalRevenue' in col:
        df_capability.rename(columns={col: 'TotalRevenue_€'}, inplace=True)
    if 'SupportPayment' in col:
        df_capability.rename(columns={col: 'SupportPayment_€'}, inplace=True)
for col in df_fip.columns:
    if 'TotalRevenue' in col:
        df_fip.rename(columns={col: 'TotalRevenue_€_FIP'}, inplace=True)
for col in df_fip.columns:
    if 'SupportPayment' in col:
        df_fip.rename(columns={col: 'SupportPayment_€_FIP'}, inplace=True)
# Merge data
df = df_capability.copy()
df['FIP_Total_Revenue'] = df_fip['TotalRevenue_€_FIP']  # Or market revenue column
df['FIP_Support'] = df_fip['SupportPayment_€_FIP']
# Calculate support and clawback for Capability CfD
df['Support_Paid'] = np.where(df['SupportPayment_€'] > 0, df['SupportPayment_€'], 0)
df['Clawback_Received'] = np.where(df['SupportPayment_€'] < 0, -df['SupportPayment_€'], 0)

# Summary calculations
total_revenue_capability = df['TotalRevenue_€'].sum()
total_revenue_fip = df['FIP_Total_Revenue'].sum()

total_support = df['Support_Paid'].sum()
total_clawback = df['Clawback_Received'].sum()
net_gov_cost_capability = total_support - total_clawback
#net_gov_cost_sde = 0
total_fip_support = df['FIP_Support'].sum()
net_gov_cost_fip = total_fip_support  # FiP has no clawback

support_months = (df['Support_Paid'] > 0).sum()
clawback_months = (df['Clawback_Received'] > 0).sum()
fip_support_months = (df['FIP_Support'] > 0).sum()
# Simple summary
print("="*60)
print("CAPABILITY-BASED CFD vs FiP COMPARISON")
print("="*60)

print("\n1. GENERATOR REVENUE:")
print(f"   Capability CfD:  €{total_revenue_capability:,.0f}")
print(f"   FiP:            €{total_revenue_fip:,.0f}")
print(f"   Difference:      €{total_revenue_capability - total_revenue_fip:,.0f}")

print("\n2. CONSUMER/GOVERNMENT COST:")
print(f"   Capability CfD:")
print(f"      Support paid:     €{total_support:,.0f}")
print(f"      Clawback received: €{total_clawback:,.0f}")
print(f"      NET cost:         €{net_gov_cost_capability:,.0f}")
print(f"\n   FiP:")
#print(f"      NET cost:         €0 (no payments, ref > strike)")  #Baseline
print(f"      Support paid:     €{total_fip_support:,.0f}")
print(f"      Clawback received: €0 (one-way mechanism)")
print(f"      NET cost:         €{net_gov_cost_fip:,.0f}")




print("\n3. MECHANISM ACTIVITY:")
print(f"   Capability CfD:")
print(f"      Support months:   {support_months} / {len(df)}")
print(f"      Clawback months:  {clawback_months} / {len(df)}")
print(f"   FiP:")
#print(f"      Payment months:   0 / {len(df)}")      #Baseline
print(f"      Payment months:   {fip_support_months} / {len(df)} ")



# Simple visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Total Revenue Comparison
schemes = ['Capability\nCfD', 'FIP']
revenues = [total_revenue_capability/1e6, total_revenue_fip/1e6]
colors = ['#3498db', '#e74c3c']

axes[0].bar(schemes, revenues, color=colors, alpha=0.7)
axes[0].set_ylabel('Total Revenue (Million €)')
axes[0].set_title('Total Generator Revenue\n(15-year period)')
axes[0].grid(alpha=0.3, axis='y')

# Add value labels on bars
for i, v in enumerate(revenues):
    axes[0].text(i, v, f'€{v:.1f}M', ha='center', va='bottom')

# 2. Government Cost Comparison
cost_categories = ['Support\nPaid', 'Clawback\nReceived', 'Net\nCost']
capability_costs = [total_support/1e6, -total_clawback/1e6, net_gov_cost_capability/1e6]
#sde_costs = [0, 0, 0]
fip_costs = [total_fip_support/1e6, 0, net_gov_cost_fip/1e6]

x = np.arange(len(cost_categories))
width = 0.35

axes[1].bar(x - width/2, capability_costs, width, label='Capability CfD', alpha=0.7, color='#3498db')
axes[1].bar(x + width/2, fip_costs, width, label='FiP', alpha=0.7, color='#e74c3c')
axes[1].set_ylabel('Amount (Million €)')
axes[1].set_title('Government Cost Comparison')
axes[1].set_xticks(x)
axes[1].set_xticklabels(cost_categories)
axes[1].legend()

# 3. Monthly Payment Activity - ADD THIS CHART!
months_data = {
    'Capability CfD': [support_months, clawback_months],
    'FiP': [fip_support_months, 0]
}

x_pos = np.arange(2)
width = 0.35

axes[2].bar(x_pos - width/2, months_data['Capability CfD'], width, 
           label='Capability CfD', alpha=0.7, color='#3498db')
axes[2].bar(x_pos + width/2, months_data['SDE+'], width, 
           label='SDE+', alpha=0.7, color='#e74c3c')

axes[2].set_ylabel('Number of Months')
axes[2].set_title('Mechanism Activity\n(15-year period = 180 months)')
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(['Support\nMonths', 'Clawback\nMonths'])
axes[2].legend()
axes[2].grid(alpha=0.3, axis='y')

# Add value labels
for i, (cap_val, sde_val) in enumerate(zip(months_data['Capability CfD'], months_data['SDE+'])):
    axes[2].text(i - width/2, cap_val, str(cap_val), ha='center', va='bottom')
    if sde_val > 0:
        axes[2].text(i + width/2, sde_val, str(sde_val), ha='center', va='bottom')

plt.tight_layout()
#plot_path = "D:/Thesis_Project/thesis_data/results/DK/Comparison/Tech_specific/high/CONsumer_comparison_Ccfd_FiP_DK_high_techref.pdf"
#plt.savefig(plot_path)

plt.show()