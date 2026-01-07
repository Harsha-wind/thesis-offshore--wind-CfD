import pandas as pd
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

DATA_DIR = Path("thesis_data/results/DK")

baseline_cfd_path = DATA_DIR / "Cap_Cfd/high/Total_Revenue_monthly_Cap_Cfd_DK_high.csv"
techspecific_cfd_path = DATA_DIR / "Cap_Cfd/Tech_specific/high/Total_Revenue_monthly_Cap_Cfd_DK_high_techref.csv"
fip_path = DATA_DIR / "FiP/high/Total_Revenue_monthly_FiP_DK_high.csv"

output_path = DATA_DIR / "postprocessed/monthly_cashflow_comparison_DK_high.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================

df_baseline = pd.read_csv(baseline_cfd_path)
df_baseline.columns = df_baseline.columns.str.strip()

df_techspecific = pd.read_csv(techspecific_cfd_path)
df_techspecific.columns = df_techspecific.columns.str.strip()

df_fip = pd.read_csv(fip_path)
df_fip.columns = df_fip.columns.str.strip()

# =============================================================================
# STANDARDISE COLUMN NAMES
# =============================================================================

for col in df_baseline.columns:
    if "TotalRevenue" in col:
        df_baseline.rename(columns={col: "TotalRevenue_Capability_€"}, inplace=True)

for col in df_techspecific.columns:
    if "TotalRevenue" in col:
        df_techspecific.rename(columns={col: "TotalRevenue_TechSpecific_€"}, inplace=True)

for col in df_fip.columns:
    if "TotalRevenue" in col:
        df_fip.rename(columns={col: "TotalRevenue_FiP_€"}, inplace=True)
    if "Market_Revenue" in col:
        df_fip.rename(columns={col: "Market_Revenue_DK_€"}, inplace=True)

# =============================================================================
# ECONOMIC ASSUMPTIONS
# =============================================================================

OPEX_EUR_PER_KW_YEAR = 60
INSTALLED_CAPACITY_MW = 740

annual_opex = OPEX_EUR_PER_KW_YEAR * 1000 * INSTALLED_CAPACITY_MW
monthly_opex = annual_opex / 12

# =============================================================================
# MONTHLY CASH FLOW CALCULATION
# =============================================================================

df_cashflow = pd.DataFrame({
    "Month": pd.to_datetime(df_baseline["Month"]),
    "CashFlow_Capability_CfD_€":
        df_baseline["TotalRevenue_Capability_€"] - monthly_opex,
    "CashFlow_TechSpecific_CfD_€":
        df_techspecific["TotalRevenue_TechSpecific_€"] - monthly_opex,
    "CashFlow_FiP_€":
        df_fip["TotalRevenue_FiP_€"] - monthly_opex,
    "CashFlow_Market_€":
        df_fip["Market_Revenue_DK_€"] - monthly_opex
})

# =============================================================================
# EXPORT RESULTS
# =============================================================================

df_cashflow.to_csv(output_path, index=False)
print(f"Monthly cash flow file saved to: {output_path}")
