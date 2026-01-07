import pandas as pd
from pathlib import Path


# USER INPUTS

STRIKE_PRICE = 103.1  # €/MWh # strike price for FiP scheme DK

DATA_DIR = Path("thesis_data/results/DK/FiP/high")

INPUT_FILE = DATA_DIR / "hourly_actual_generation_wake_DK_high.csv"
OUTPUT_FILE = DATA_DIR / "support_payment_FiP_DK_high.csv"


# LOAD DATA

df = pd.read_csv(INPUT_FILE, parse_dates=["hours"])
df["hours"] = pd.to_datetime(df["hours"], dayfirst=True, errors="coerce")
df.set_index("hours", inplace=True)

df.columns = df.columns.str.strip()

df.rename(
    columns={
        col: "ActualGeneration_MWh"
        for col in df.columns if "Actual_Generation_MWh" in col
    },
    inplace=True
)

df.rename(
    columns={
        col: "SpotPrice_€/MWh"
        for col in df.columns if "SpotPrice/MWh" in col
    },
    inplace=True
)


# FiP SUPPORT PAYMENT CALCULATION

df["SupportPayment_€"] = (
    STRIKE_PRICE - df["SpotPrice_€/MWh"]
) * df["ActualGeneration_MWh"]

df.loc[df["SpotPrice_€/MWh"] < 0, "SupportPayment_€"] = 0
df.loc[df["SpotPrice_€/MWh"] > STRIKE_PRICE, "SupportPayment_€"] = 0

#output
df.to_csv(OUTPUT_FILE)
