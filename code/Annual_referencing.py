import pandas as pd
from pathlib import Path



BASE_RESULTS = Path("results")


input_path = BASE_RESULTS / "NL/EP/High/merged_wind_price_full_high_NL.csv"
output_path = BASE_RESULTS / "NL/EP/High/Annual_Reference_Price_NL_High.csv"


df = pd.read_csv(input_path)

# Strip column names
df.columns = df.columns.str.strip()

# Rename columns
for col in df.columns:
    if 'hours' in col:
        df.rename(columns={col: 'hour'}, inplace=True)
    if 'SpotPrice' in col:
        df.rename(columns={col: 'SpotPrice/MWh'}, inplace=True)
    if 'Wind_Direction' in col:
        df.rename(columns={col: 'Wind_Direction'}, inplace=True)


# ANNUAL REFERENCING for NL

df['year'] = pd.to_datetime(df['hour'], errors='coerce').dt.year

Annual_ref_spot = df.groupby('year')['SpotPrice/MWh'].mean().reset_index()
Annual_ref_spot = Annual_ref_spot[Annual_ref_spot['year'] < 2036]

Annual_ref_spot.rename(
    columns={'SpotPrice/MWh': 'Annual_Reference_Price_€/MWh_raw'},
    inplace=True
)

# Offshore correction factor
correction_factor = 0.9
Annual_ref_spot['Annual_Reference_Price_€/MWh'] = (
    Annual_ref_spot['Annual_Reference_Price_€/MWh_raw'] * correction_factor
)

Annual_ref_spot = Annual_ref_spot[['year', 'Annual_Reference_Price_€/MWh']]
Annual_ref_spot.rename(columns={'year': 'Year'}, inplace=True)


output_path.parent.mkdir(parents=True, exist_ok=True)
Annual_ref_spot.to_csv(output_path, index=False)

print(Annual_ref_spot)