"""Capability cfd support payment calculation. This code can be used for both the reference methods (flat and tech specific)
Just by changing the input file path with respect to the price scenarios and the case studies, the code can be used."""

import pandas as pd
import numpy as np


# INPUTS
strike_price = 103.1  # €/MWh ( change based on the case study)

input_path = (
    "thesis_data/results/DK/Cap_Cfd/Tech_specific/high/"
    "monthly_tech_specific_ref_price_high_dk.csv"
)

output_path = (
    "thesis_data/results/DK/Cap_Cfd/Tech_specific/high/"
    "Capability_generation_DK_CfD_support_payment_high_techref.csv"
)

# LOAD DATA
df = pd.read_csv(input_path)
df.columns = df.columns.str.strip()

# Rename to consistent column names (based on  current file naming)
for col in df.columns:
    if "tech_specific_ref_price" in col:
        df.rename(columns={col: "AvgSpotPrice_techref_€/MWh"}, inplace=True)
    if "cap_sum" in col:
        df.rename(columns={col: "Total_Capability_MWh"}, inplace=True)

required_cols = {"Month", "AvgSpotPrice_techref_€/MWh", "Total_Capability_MWh"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in input CSV: {sorted(missing)}")


# SUPPORT PAYMENT CALCULATION

df_out = df[["Month", "AvgSpotPrice_techref_€/MWh", "Total_Capability_MWh"]].copy()

df_out["StrikePrice_€/MWh"] = strike_price
df_out["PriceDiff"] = df_out["StrikePrice_€/MWh"] - df_out["AvgSpotPrice_techref_€/MWh"]
df_out["SupportPayment_€"] = df_out["PriceDiff"] * df_out["Total_Capability_MWh"]


df_out = df_out[["Month", "AvgSpotPrice_techref_€/MWh", "Total_Capability_MWh", "SupportPayment_€"]]

# SAVE
df_out.to_csv(output_path, index=False)
