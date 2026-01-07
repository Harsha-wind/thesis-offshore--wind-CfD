"""The Current represents the Netherlands SDE+ support mechanism scenario.
This script is used for the High scenario, just replacing the inputs baseline and moderate scenario can also be calculated."""


import pandas as pd
from pathlib import Path


# PATHS
DATA_DIR = Path("thesis_data/results/NL")
SCENARIO_DIR = DATA_DIR / "Current/high"
REF_DIR = DATA_DIR / "EP/High"

revenue_path = SCENARIO_DIR / "Market_revenue_NL_(High_NL).csv"
reference_path = REF_DIR / "Annual_Reference_Price_NL_High.csv"

output_path = SCENARIO_DIR / "Total_Revenue_NL_(High_NL)_SDE_payment.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)


# LOAD DATA
df_ref = pd.read_csv(reference_path)
df_rev = pd.read_csv(revenue_path, parse_dates=["hour"])

df_ref.columns = df_ref.columns.str.strip()
df_rev.columns = df_rev.columns.str.strip()

# CLEAN AND RENAME COLUMNS
df_ref.rename(columns={
    col: "Annual_Reference_Price_€/MWh"
    for col in df_ref.columns
    if "Annual_Reference_Price" in col
}, inplace=True)

df_ref.rename(columns={
    col: "year"
    for col in df_ref.columns
    if "Year" in col
}, inplace=True)

df_rev.rename(columns={
    col: "SpotPrice_€/MWh"
    for col in df_rev.columns
    if "SpotPrice" in col
}, inplace=True)

df_rev.rename(columns={
    col: "ActualGeneration_MWh"
    for col in df_rev.columns
    if "Dispatched_MWh" in col
}, inplace=True)

df_rev.rename(columns={
    col: "Market_Revenue_€"
    for col in df_rev.columns
    if "Market_Revenue" in col
}, inplace=True)

df_rev = df_rev[
    ["hour", "SpotPrice_€/MWh", "ActualGeneration_MWh", "Market_Revenue_€"]
].copy()

# AGGREGATE TO ANNUAL LEVEL

df_rev = df_rev[
    (df_rev["hour"].dt.year >= 2021) &
    (df_rev["hour"].dt.year <= 2035)
].copy()

df_rev["year"] = df_rev["hour"].dt.year

df_annual = df_rev.groupby("year").agg({
    "Market_Revenue_€": "sum",
    "SpotPrice_€/MWh": "mean",
    "ActualGeneration_MWh": "sum"
}).reset_index()

df_output = df_annual.merge(
    df_ref[["year", "Annual_Reference_Price_€/MWh"]],
    on="year",
    how="left"
)


# SDE++ SUPPORT PAYMENT CALCULATION

SDE_STRIKE_PRICE = 54.5  # €/MWh

df_output["SDE_SupportPayment_€"] = (
    (SDE_STRIKE_PRICE - df_output["Annual_Reference_Price_€/MWh"])
    * df_output["ActualGeneration_MWh"]
).clip(lower=0)

df_output["TotalRevenue_SDE_€"] = (
    df_output["Market_Revenue_€"] + df_output["SDE_SupportPayment_€"]
)

# SAVE OUTPUT
df_output.to_csv(output_path, index=False)
