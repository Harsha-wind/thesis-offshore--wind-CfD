"""
Scenario 3: High - Aggressive Renewable Penetration (Denmark)
Mehta Method variant: Mean, CV, and Correlation evolve (as in your code)
Generates hourly future prices for 2025-2032 and saves:
1) Hourly time series (wind + price)
2) Yearly stats (target vs realized)
"""

import pandas as pd
import numpy as np

# =============================================================================
# CONFIG (repo-friendly paths: use relative paths inside your repo)
# =============================================================================

INPUT_FILE = "data/merged_wind_speed_direction_price_DK.csv"

OUTPUT_PRICES = "results/DK/EP/high/future_prices_scenario3_high_DK.csv"
OUTPUT_STATS = "results/DK/EP/high/scenario3_high_stats_DK.csv"

START_DATE = "2025-04-01"
END_DATE = "2032-03-31 23:00"

EXCLUDE_YEARS = [2022]
RANDOM_SEED = 42

# =============================================================================
# SCENARIO PARAMETERS (High scenario: Mean declines, CV increases, rho becomes more negative)
# =============================================================================

# Starting point (your base)
HISTORICAL_MEAN = 63.5
HISTORICAL_CV = 0.81
HISTORICAL_CORRELATION = -0.299

# Target by 2032 (high renewable scenario)
TARGET_MEAN_2032 = 40.0
TARGET_CV_2032 = 1.0
TARGET_CORRELATION_2032 = -0.75

# =============================================================================
# STEP 1: LOAD HISTORICAL DATA
# =============================================================================

df = pd.read_csv(INPUT_FILE)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.set_index("datetime").sort_index()
df = df.dropna()

if EXCLUDE_YEARS:
    df = df[~df.index.year.isin(EXCLUDE_YEARS)]

required_cols = {"wind_speed_hub", "wind_dir_deg"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in input CSV: {sorted(missing)}")

# =============================================================================
# STEP 2: PREPARE FUTURE TIME SERIES (tile historical wind data)
# =============================================================================

future_dates = pd.date_range(START_DATE, END_DATE, freq="h")
n_hours = len(future_dates)

hist_wind = df["wind_speed_hub"].values
hist_wind_dir = df["wind_dir_deg"].values

n_repeat = int(np.ceil(n_hours / len(hist_wind)))
future_wind = np.tile(hist_wind, n_repeat)[:n_hours]
future_wind_dir = np.tile(hist_wind_dir, n_repeat)[:n_hours]

results = pd.DataFrame(index=future_dates)
results["wind_speed_hub"] = future_wind
results["wind_dir_deg"] = future_wind_dir
results["spot_price_eur_per_mwh"] = np.nan

# =============================================================================
# STEP 3: YEARLY PARAMETER EVOLUTION (linear from 2025 to 2032)
# =============================================================================

years = range(future_dates.year.min(), future_dates.year.max() + 1)
yearly_params = []

for year in years:
    progress = (year - 2025) / (2032 - 2025)

    mean = HISTORICAL_MEAN + (TARGET_MEAN_2032 - HISTORICAL_MEAN) * progress

    # CV evolves (as in your "NEW (correct)" line)
    cv = HISTORICAL_CV + (TARGET_CV_2032 - HISTORICAL_CV) * progress

    std = mean * cv

    rho = HISTORICAL_CORRELATION + (TARGET_CORRELATION_2032 - HISTORICAL_CORRELATION) * progress

    yearly_params.append({"year": year, "mean": mean, "cv": cv, "std": std, "rho": rho})

# =============================================================================
# STEP 4: GENERATE PRICES (Mehta method)
# =============================================================================

np.random.seed(RANDOM_SEED)
yearly_stats = []

for params in yearly_params:
    year = params["year"]
    mean = params["mean"]
    std = params["std"]
    rho = params["rho"]

    year_mask = results.index.year == year
    wind = results.loc[year_mask, "wind_speed_hub"].values
    n = len(wind)
    if n == 0:
        continue

    wind_mean = wind.mean()
    wind_std = wind.std()
    wind_norm = (wind - wind_mean) / wind_std if wind_std > 0 else np.zeros(n)

    noise = np.random.randn(n)

    if np.dot(wind_norm, wind_norm) > 0:
        proj = np.dot(noise, wind_norm) / np.dot(wind_norm, wind_norm)
        noise_orth = noise - proj * wind_norm
        if noise_orth.std() > 0:
            noise_orth = noise_orth / noise_orth.std()
    else:
        noise_orth = noise

    driver = rho * wind_norm + np.sqrt(max(0, 1 - rho**2)) * noise_orth
    prices = mean + std * driver

    results.loc[year_mask, "spot_price_eur_per_mwh"] = prices

    real_mean = prices.mean()
    real_std = prices.std()
    real_cv = real_std / abs(real_mean) if real_mean != 0 else 0
    real_rho = np.corrcoef(prices, wind)[0, 1]

    neg_hours = (prices < 0).sum()
    neg_pct = 100 * neg_hours / n

    yearly_stats.append(
        {
            "Year": year,
            "Target_Mean": mean,
            "Realized_Mean": real_mean,
            "Target_CV": params["cv"],
            "Realized_CV": real_cv,
            "Target_Rho": rho,
            "Realized_Rho": real_rho,
            "Negative_Hours": neg_hours,
            "Negative_Pct": neg_pct,
            "Min": prices.min(),
            "Max": prices.max(),
        }
    )

stats_df = pd.DataFrame(yearly_stats)

# =============================================================================
# STEP 5: SAVE RESULTS
# =============================================================================

results.index.name = "datetime"
results.to_csv(OUTPUT_PRICES)

stats_df.to_csv(OUTPUT_STATS, index=False)

print("Done.")
print("Saved prices to:", OUTPUT_PRICES)
print("Saved stats to:", OUTPUT_STATS)
