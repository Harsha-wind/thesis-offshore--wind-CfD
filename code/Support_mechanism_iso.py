import pandas as pd
import numpy as np

# Replace these with your actual NPV values from your separate analysis
npv_merchant = -594516179  # Merchant - negative NPV
npv_cfd = 976125254

        # Capability CfD - positive NPV

npv_improvement = npv_cfd - npv_merchant

# Load DSCR data
merchant_dscr = "D:/Thesis_Project/thesis_data/results/DK/Market_Merchant/high/dscr_monthly_analysis_Market_Merchant_high.csv"
df_merchant = pd.read_csv(merchant_dscr)

cfd_dscr = "D:/Thesis_Project/thesis_data/results/DK/Cap_Cfd/Tech_specific/high/dscr_monthly_analysis_Cap_Cfd_DK_high_techref.csv"
df_cfd = pd.read_csv(cfd_dscr)

# ============================================================================
# CALCULATE ALL METRICS
# ============================================================================

# Average DSCR
avg_dscr_merchant = df_merchant['dscr'].mean()
avg_dscr_cfd = df_cfd['dscr'].mean()
dscr_improvement = avg_dscr_cfd - avg_dscr_merchant
dscr_improvement_pct = (dscr_improvement / avg_dscr_merchant) * 100

# Minimum DSCR
min_dscr_merchant = df_merchant['dscr'].min()
min_dscr_cfd = df_cfd['dscr'].min()
min_dscr_improvement = min_dscr_cfd - min_dscr_merchant
min_dscr_improvement_pct = (min_dscr_improvement / min_dscr_merchant) * 100

# Months below DSCR = 1.25
months_below_1_merchant = (df_merchant['dscr'] < 1.25).sum()
months_below_1_cfd = (df_cfd['dscr'] < 1.25).sum()
months_reduction = months_below_1_merchant - months_below_1_cfd
months_reduction_pct = (months_reduction / months_below_1_merchant * 100) if months_below_1_merchant > 0 else 0

# ============================================================================
# PRINT RESULTS
# ============================================================================

print("\nNetherlands SUPPORT MECHANISM ANALYSIS")
print("="*60)

print("\nNPV (Million EUR):")
print(f"  Merchant:          {npv_merchant/1e6:.2f}")
print(f"  Capability CfD:    {npv_cfd/1e6:.2f}")
print(f"  Improvement:       {npv_improvement/1e6:.2f}")

print("\nAverage DSCR:")
print(f"  Merchant:          {avg_dscr_merchant:.3f}")
print(f"  Capability CfD:    {avg_dscr_cfd:.3f}")
print(f"  Improvement:       {dscr_improvement:.3f} ({dscr_improvement_pct:.1f}%)")

print("\nMinimum DSCR:")
print(f"  Merchant:          {min_dscr_merchant:.3f}")
print(f"  Capability CfD:    {min_dscr_cfd:.3f}")
print(f"  Improvement:       {min_dscr_improvement:.3f} ({min_dscr_improvement_pct:.1f}%)")

print("\nMonths with DSCR < 1.25:")
print(f"  Merchant:          {months_below_1_merchant}")
print(f"  Capability CfD:    {months_below_1_cfd}")
print(f"  Reduction:         {months_reduction} ({months_reduction_pct:.1f}%)")

print("="*60)

# ============================================================================
# SAVE RESULTS TO CSV
# ============================================================================

# Summary metrics
summary = pd.DataFrame({
    'Metric': [
        'NPV (€M)',
        'Average DSCR',
        'Minimum DSCR',
        'Months DSCR < 1.25'
    ],
    'Merchant': [
        npv_merchant / 1e6,
        avg_dscr_merchant,
        min_dscr_merchant,
        months_below_1_merchant
    ],
    'Capability_CfD': [
        npv_cfd / 1e6,
        avg_dscr_cfd,
        min_dscr_cfd,
        months_below_1_cfd
    ],
    'Absolute_Change': [
        npv_improvement / 1e6,
        dscr_improvement,
        min_dscr_improvement,
        -months_reduction
    ],
    'Percentage_Change': [
        'Negative_to_Positive',
        dscr_improvement_pct,
        min_dscr_improvement_pct,
        -months_reduction_pct
    ]
})

output_path = "D:/Thesis_Project/thesis_data/results/DK/Support_Mechanism_ISO_summary_DK_high_Cap_techref.csv"
summary.to_csv(output_path, index=False)

# Monthly DSCR comparison
comparison = pd.DataFrame({
    'Month': df_merchant['month'],
    'DSCR_Merchant': df_merchant['dscr'],
    'DSCR_CfD': df_cfd['dscr'],
    'DSCR_Difference': df_cfd['dscr'] - df_merchant['dscr'],
    'DSCR_Improvement_Pct': ((df_cfd['dscr'] - df_merchant['dscr']) / df_merchant['dscr'] * 100)
})

comparison_path = "D:/Thesis_Project/thesis_data/results/DK/Support_Mechanism_ISO_monthly_dscr_DK_high_Cap_techref.csv"
comparison.to_csv(comparison_path, index=False)

print(f"\nFiles saved:")
print(f"  {output_path}")
print(f"  {comparison_path}\n")
