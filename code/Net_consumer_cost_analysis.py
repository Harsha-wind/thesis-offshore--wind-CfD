" This consumer net cost analysis compares Technology-Specific Capability CfD and FiP schemes for the baseline scenario in Denmark.It can be used for other sceanrios as well by changing the input file and also the same implies for the NL case study "


import pandas as pd
import numpy as np

# =============================================================================
# FILE PATHS
# =============================================================================
input_path_capability = (
    "thesis_data/results/DK/Cap_Cfd/Tech_specific/baseline/"
    "Total_Revenue_monthly_Cap_Cfd_DK_baseline_techref.csv"
)

input_path_fip = (
    "thesis_data/results/DK/FiP/baseline/"
    "Total_Revenue_monthly_FiP_DK_baseline.csv"
)

output_path = (
    "thesis_data/results/DK/Comparison/"
    "summary_capcfd_vs_fip_baseline_techref.csv"
)

# =============================================================================
# LOAD DATA
# =============================================================================
df_capability = pd.read_csv(input_path_capability)
df_fip = pd.read_csv(input_path_fip)

df_capability.columns = df_capability.columns.str.strip()
df_fip.columns = df_fip.columns.str.strip()

# =============================================================================
# RENAME COLUMNS FOR CONSISTENCY
# =============================================================================
for col in df_capability.columns:
    if "TotalRevenue" in col:
        df_capability.rename(columns={col: "TotalRevenue_€"}, inplace=True)
    if "SupportPayment" in col:
        df_capability.rename(columns={col: "SupportPayment_€"}, inplace=True)

for col in df_fip.columns:
    if "TotalRevenue" in col:
        df_fip.rename(columns={col: "TotalRevenue_€_FIP"}, inplace=True)
    if "SupportPayment" in col:
        df_fip.rename(columns={col: "SupportPayment_€_FIP"}, inplace=True)

# =============================================================================
# ALIGN AND MERGE
# =============================================================================
n_months = min(len(df_capability), len(df_fip))

df = df_capability.iloc[:n_months].copy()
df["FIP_Total_Revenue"] = df_fip.loc[: n_months - 1, "TotalRevenue_€_FIP"].values
df["FIP_Support"] = df_fip.loc[: n_months - 1, "SupportPayment_€_FIP"].values

# =============================================================================
# CAPABILITY CFD: SUPPORT VS CLAWBACK
# =============================================================================
df["Support_Paid"] = np.where(df["SupportPayment_€"] > 0, df["SupportPayment_€"], 0.0)
df["Clawback_Received"] = np.where(df["SupportPayment_€"] < 0, -df["SupportPayment_€"], 0.0)

# =============================================================================
# AGGREGATE METRICS
# =============================================================================
summary = pd.DataFrame(
    [{
        "months_total": n_months,

        "generator_revenue_capability_€": df["TotalRevenue_€"].sum(),
        "generator_revenue_fip_€": df["FIP_Total_Revenue"].sum(),
        "generator_revenue_diff_cap_minus_fip_€":
            df["TotalRevenue_€"].sum() - df["FIP_Total_Revenue"].sum(),

        "support_paid_capability_€": df["Support_Paid"].sum(),
        "clawback_received_capability_€": df["Clawback_Received"].sum(),
        "net_gov_cost_capability_€":
            df["Support_Paid"].sum() - df["Clawback_Received"].sum(),

        "support_paid_fip_€": df["FIP_Support"].sum(),
        "clawback_received_fip_€": 0.0,
        "net_gov_cost_fip_€": df["FIP_Support"].sum(),

        "support_months_capability": (df["Support_Paid"] > 0).sum(),
        "clawback_months_capability": (df["Clawback_Received"] > 0).sum(),
        "support_months_fip": (df["FIP_Support"] > 0).sum(),
    }]
)

# =============================================================================
# SAVE
# =============================================================================
summary.to_csv(output_path, index=False)
