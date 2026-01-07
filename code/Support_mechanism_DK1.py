"""Market isolation analysis comparing Market Merchant and the Support Mechanism used in DK
This can be used for other scenarios and the support mechanisms used in the DK case study
Just by changing the input file paths and the output file paths accordingly"""



import pandas as pd
import numpy as np

# INPUT FILE PATHS
merchant_npv_path = (
    "thesis_data/results/DK/Market_Merchant/high/"
    "NPV_summary_Market_Merchant_DK_high.csv"
)

cfd_npv_path = (
    "thesis_data/results/DK/Cap_Cfd/Tech_specific/high/"
    "NPV_summary_CapCfd_DK_high_techref_tax_adjusted.csv"
)

merchant_dscr_path = (
    "thesis_data/results/DK/Market_Merchant/high/"
    "dscr_monthly_analysis_Market_Merchant_high.csv"
)

cfd_dscr_path = (
    "thesis_data/results/DK/Cap_Cfd/Tech_specific/high/"
    "dscr_monthly_analysis_Cap_Cfd_DK_high_techref.csv"
)

output_path = "thesis_data/results/DK/Support_Mechanism_ISO_summary_DK_high_Cap_techref.csv"
comparison_path = "thesis_data/results/DK/Support_Mechanism_ISO_monthly_dscr_DK_high_Cap_techref.csv"

# Load NPV values (from your NPV summary CSVs)

df_npv_merchant = pd.read_csv(merchant_npv_path)
df_npv_cfd = pd.read_csv(cfd_npv_path)

df_npv_merchant.columns = df_npv_merchant.columns.str.strip()
df_npv_cfd.columns = df_npv_cfd.columns.str.strip()

# Extract NPV values
npv_col_merchant = None
for col in df_npv_merchant.columns:
    if "NPV" in col:
        npv_col_merchant = col
        break

npv_col_cfd = None
for col in df_npv_cfd.columns:
    if "NPV" in col:
        npv_col_cfd = col
        break

if npv_col_merchant is None or npv_col_cfd is None:
    raise ValueError("NPV column not found in one of the NPV summary CSVs.")

npv_merchant = df_npv_merchant[npv_col_merchant].iloc[0]
npv_cfd = df_npv_cfd[npv_col_cfd].iloc[0]
npv_improvement = npv_cfd - npv_merchant

# Load data
df_merchant = pd.read_csv(merchant_dscr_path)
df_cfd = pd.read_csv(cfd_dscr_path)

df_merchant.columns = df_merchant.columns.str.strip()
df_cfd.columns = df_cfd.columns.str.strip()


# CALCULATE ALL METRICS 


avg_dscr_merchant = df_merchant["dscr"].mean()
avg_dscr_cfd = df_cfd["dscr"].mean()
dscr_improvement = avg_dscr_cfd - avg_dscr_merchant
dscr_improvement_pct = (dscr_improvement / avg_dscr_merchant) * 100

min_dscr_merchant = df_merchant["dscr"].min()
min_dscr_cfd = df_cfd["dscr"].min()
min_dscr_improvement = min_dscr_cfd - min_dscr_merchant
min_dscr_improvement_pct = (min_dscr_improvement / min_dscr_merchant) * 100

months_below_1_merchant = (df_merchant["dscr"] < 1.25).sum()
months_below_1_cfd = (df_cfd["dscr"] < 1.25).sum()
months_reduction = months_below_1_merchant - months_below_1_cfd
months_reduction_pct = (months_reduction / months_below_1_merchant * 100) if months_below_1_merchant > 0 else 0


# SAVE RESULTS TO CSV 

summary = pd.DataFrame({
    "Metric": [
        "NPV (€M)",
        "Average DSCR",
        "Minimum DSCR",
        "Months DSCR < 1.25"
    ],
    "Merchant": [
        npv_merchant / 1e6,
        avg_dscr_merchant,
        min_dscr_merchant,
        months_below_1_merchant
    ],
    "Capability_CfD": [
        npv_cfd / 1e6,
        avg_dscr_cfd,
        min_dscr_cfd,
        months_below_1_cfd
    ],
    "Absolute_Change": [
        npv_improvement / 1e6,
        dscr_improvement,
        min_dscr_improvement,
        -months_reduction
    ],
    "Percentage_Change": [
        "Negative_to_Positive",
        dscr_improvement_pct,
        min_dscr_improvement_pct,
        -months_reduction_pct
    ]
})

summary.to_csv(output_path, index=False)

comparison = pd.DataFrame({
    "Month": df_merchant["month"],
    "DSCR_Merchant": df_merchant["dscr"],
    "DSCR_CfD": df_cfd["dscr"],
    "DSCR_Difference": df_cfd["dscr"] - df_merchant["dscr"],
    "DSCR_Improvement_Pct": ((df_cfd["dscr"] - df_merchant["dscr"]) / df_merchant["dscr"] * 100)
})

comparison.to_csv(comparison_path, index=False)
