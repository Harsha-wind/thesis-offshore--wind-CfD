# === Hedge Effectiveness (variance reduction) ===
import numpy as np
import pandas as pd

# --- Load CSVs ---
input_path_Ccfd = "C:/Thesis_Project/thesis_data/results/Financial_Quantification/Baseline/Total/monthly_cash_flows(capability)_total.csv"
df_ccfd = pd.read_csv(input_path_Ccfd)

input_path_fit = "C:/Thesis_Project/thesis_data/results/Financial_Quantification/FiP-dk/Total/monthly_cash_flows(FiP-dk)_total.csv"
df_fit = pd.read_csv(input_path_fit)

input_path_MM = "C:/Thesis_Project/thesis_data/results/Financial_Quantification/MM/Total/monthly_cash_flows(MM)_total.csv"
df_MM = pd.read_csv(input_path_MM)

# --- Clean column names ---
df_ccfd.columns = df_ccfd.columns.str.strip()
df_fit.columns  = df_fit.columns.str.strip()
df_MM.columns   = df_MM.columns.str.strip()

# --- Rename target columns ---
for col in df_ccfd.columns: 
    if 'MonthlyCashFlow' in col:
        df_ccfd.rename(columns={col: 'MonthlyCashFlow_Ccfd_€'}, inplace=True)

for col in df_fit.columns:
    if 'MonthlyCashFlow' in col:
        df_fit.rename(columns={col: 'MonthlyCashFlow_FiP_€'}, inplace=True)

for col in df_MM.columns:
    if 'MonthlyCashFlow' in col:
        df_MM.rename(columns={col: 'MonthlyCashFlow_MM_€'}, inplace=True)

# --- Merge into one DataFrame (on month index if available) ---
df_all = pd.concat([df_ccfd, df_fit, df_MM], axis=1)

# --- Define hedge effectiveness ---
def hedge_effectiveness(mech_revenue, merch_revenue):
    """
    Hedge Effectiveness (variance reduction) compared to merchant case.
    """
    var_mech = np.var(mech_revenue, ddof=1)
    var_merch = np.var(merch_revenue, ddof=1)
    return 1 - (var_mech / var_merch)

# --- Calculate HE for Cap-CfD and FiP ---
capcfd_HE = hedge_effectiveness(df_all["MonthlyCashFlow_Ccfd_€"], df_all["MonthlyCashFlow_MM_€"])
fip_HE    = hedge_effectiveness(df_all["MonthlyCashFlow_FiP_€"], df_all["MonthlyCashFlow_MM_€"])

print(f"Hedge Effectiveness (Cap-CfD): {capcfd_HE:.3f}")
print(f"Hedge Effectiveness (FiP): {fip_HE:.3f}")

import matplotlib.pyplot as plt

# Hedge effectiveness values (from your calculation)
he_values = {
    "Cap-CfD": capcfd_HE,
    "FiP": fip_HE,
    "Merchant": 0.0  # by definition, no hedge
}

# Convert to percentages
he_percent = {k: v * 100 for k, v in he_values.items()}

# Plot
plt.figure(figsize=(8,5))
bars = plt.bar(he_percent.keys(), he_percent.values(), color=["green","orange","blue"])

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1,
             f"{yval:.1f}%", ha='center', va='bottom', fontsize=10)

plt.ylabel("Hedge Effectiveness (%)")
plt.title("Revenue Variance Reduction vs Merchant")
plt.ylim(0, 100)  # fix axis scale to make % interpretation clear
plt.grid(axis="y", linestyle="--", alpha=0.6)
plot_path = "C:/Thesis_Project/thesis_data/results/Financial_Quantification/hedge_effectiveness_plot_DK.png"
print(f"Plot saved to: {plot_path}")
plt.savefig(plot_path)
plt.show()
