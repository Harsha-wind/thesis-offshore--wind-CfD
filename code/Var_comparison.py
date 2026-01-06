import pandas as pd
import numpy as np

# ==== inputs ====
sde_path   = "C:/Thesis_Project/thesis_data/results/NL/Current/Low_Scenario/Market_Revenue_SDE_NL_monthly_(LS).csv"
capcfd_path=  "C:/Thesis_Project/thesis_data/results/NL/Capability/Low_Scenario/Total_Revenue_monthly_Ccfd_NL_(LS).csv"
opex_kw = 69  # €/kW-year

installed_capacity_mw = 740      
alpha = 0.95   # confidence level for the "at least" statement (5% left tail)
# ===============

# --- load ---
df_sde = pd.read_csv(sde_path, parse_dates=["Month"])
df_cap = pd.read_csv(capcfd_path, parse_dates=["Month"])
df_sde.columns = df_sde.columns.str.strip()
df_cap.columns = df_cap.columns.str.strip()

# --- rename revenue columns (keep your style) ---
for col in df_sde.columns:
    if "Total_Market_Revenue" in col:
        df_sde.rename(columns={col: "TotalRevenue_SDE_€"}, inplace=True)

for col in df_cap.columns:
    if "TotalRevenue" in col:
        df_cap.rename(columns={col: "TotalRevenue_CapCfD_€"}, inplace=True)

#for col in df_fip.columns:
#    if 'Market_Revenue' in col:
#        df_fip.rename(columns={col: "TotalRevenue_Market_€"}, inplace=True)

# --- align to the same months ---
df_sde["Month"] = pd.to_datetime(df_sde["Month"], errors="coerce")
df_cap["Month"] = pd.to_datetime(df_cap["Month"], errors="coerce")
df_sde["Month"] = df_sde["Month"].dt.to_period("M")
df_cap["Month"] = df_cap["Month"].dt.to_period("M")

common = set(df_sde["Month"]).intersection(set(df_cap["Month"]))
df_sde = df_sde[df_sde["Month"].isin(common)].sort_values("Month").reset_index(drop=True)
df_cap = df_cap[df_cap["Month"].isin(common)].sort_values("Month").reset_index(drop=True)

# --- monthly OPEX and cash flow ---
monthly_opex = (opex_kw * 1000 * installed_capacity_mw) / 12
df_sde["CashFlow_SDE_€"]    = df_sde["TotalRevenue_SDE_€"]    - monthly_opex
df_cap["CashFlow_CapCfD_€"] = df_cap["TotalRevenue_CapCfD_€"] - monthly_opex
#df_sde["CashFlow_Market_€"] = df_sde["TotalRevenue_Market_€"] - monthly_opex
# --- std & CoV (your logic) ---
std_sde = df_sde["CashFlow_SDE_€"].std()
std_cap = df_cap["CashFlow_CapCfD_€"].std()
#std_market = df_sde["CashFlow_Market_€"].std()

cov_sde = std_sde / df_sde["CashFlow_SDE_€"].mean()
cov_cap = std_cap / df_cap["CashFlow_CapCfD_€"].mean()
#cov_market = std_market / df_fip["CashFlow_Market_€"].mean()

print(f"Std (SDE++)          : {std_sde:,.0f} €")
print(f"Std (Cap-CfD)      : {std_cap:,.0f} €")
#print(f"Std (Market)       : {std_market:,.0f} €")
print(f"CoV  (SDE++)       : {cov_sde:.3f}")
print(f"CoV  (Cap-CfD)     : {cov_cap:.3f}")
#print(f"CoV  (Market)      : {cov_market:.3f}")

# --- left-tail VaR/ CVaR of CASH FLOW (no helpers) ---
x_sde = df_sde["CashFlow_SDE_€"].to_numpy(dtype=float)
x_cap = df_cap["CashFlow_CapCfD_€"].to_numpy(dtype=float)
#x_market = df_sde["CashFlow_Market_€"].to_numpy(dtype=float)

# 5th percentile (cash-flow floor at 95% confidence)
try:
    lvar_sde = np.quantile(x_sde, 1-alpha, method="higher")
    lvar_cap = np.quantile(x_cap, 1-alpha, method="higher")
    #lvar_market = np.quantile(x_market, 1-alpha, method="higher")
except TypeError:
    lvar_sde = np.quantile(x_sde, 1-alpha)
    lvar_cap = np.quantile(x_cap, 1-alpha)
    #lvar_market = np.quantile(x_market, 1-alpha)

# LCVaR = mean of the worst 5% months
tail_sde = x_sde[x_sde <= lvar_sde + 1e-12]
tail_cap = x_cap[x_cap <= lvar_cap + 1e-12]
#tail_market = x_market[x_market <= lvar_market + 1e-12]
lcvar_sde = tail_sde.mean() if tail_sde.size else lvar_sde
lcvar_cap = tail_cap.mean() if tail_cap.size else lvar_cap
#lcvar_market = tail_market.mean() if tail_market.size else lvar_market

print(f"LVaR95 (SDE++)      : {lvar_sde:,.0f} €")
print(f"LCVaR95 (SDE++)     : {lcvar_sde:,.0f} €")
print(f"LVaR95 (Cap-CfD)    : {lvar_cap:,.0f} €")
print(f"LCVaR95 (Cap-CfD)   : {lcvar_cap:,.0f} €")
#print(f"LVaR95 (Market)     : {lvar_market:,.0f} €")
#print(f"LCVaR95 (Market)    : {lcvar_market:,.0f} €")
