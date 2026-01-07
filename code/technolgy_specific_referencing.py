""" Technology specific reference price calculation for Capability CfD, it is volume weighted average based on generational potential
Just change the input file path for different scenarios and case studies"""


import pandas as pd
import numpy as np


# LOAD AND CLEAN DATA

csv_path = "thesis_data/results/DK/Cap_Cfd/high/Capability_generation_DK_high.csv"
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

df["hours"] = pd.to_datetime(df["hours"], errors="coerce")
df = df.dropna(subset=["hours"])


for col in df.columns:
    if "SpotPrice/MWh" in col:
        df.rename(columns={col: "SpotPrice_€/MWh"}, inplace=True)
    if "TotalCapability_MWh_withwake" in col:
        df.rename(columns={col: "TotalCapability_MWh"}, inplace=True)

required_cols = {"hours", "SpotPrice_€/MWh", "TotalCapability_MWh"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")


df["Month"] = df["hours"].dt.to_period("M")

df["weighted_spot"] = df["SpotPrice_€/MWh"] * df["TotalCapability_MWh"]

monthly_tech_ref = (
    df.groupby("Month")
      .agg(
          weighted_sum=("weighted_spot", "sum"),
          cap_sum=("TotalCapability_MWh", "sum"),
          avg_spot=("SpotPrice_€/MWh", "mean"),
          hours_count=("hours", "count"),
      )
      .reset_index()
)

monthly_tech_ref["Month"] = monthly_tech_ref["Month"].astype(str)

monthly_tech_ref["tech_specific_ref_price"] = np.where(
    monthly_tech_ref["cap_sum"] > 0,
    monthly_tech_ref["weighted_sum"] / monthly_tech_ref["cap_sum"],
    np.nan
)

monthly_tech_ref["price_difference"] = (
    monthly_tech_ref["tech_specific_ref_price"] - monthly_tech_ref["avg_spot"]
)

# SAVE RESULTS
output_path = (
    "thesis_data/results/DK/Cap_Cfd/Tech_specific/high/"
    "monthly_tech_specific_ref_price_high_dk.csv"
)

monthly_tech_ref.to_csv(output_path, index=False)
