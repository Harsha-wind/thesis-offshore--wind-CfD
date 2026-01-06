import pandas as pd
import numpy as np

# Load your CSV file (put your CSV file in the same folder as this script)
input_path = "D:/Thesis_Project/thesis_data/results/NL/EP/High/merged_wind_price_full_high_NL.csv"
df = pd.read_csv(input_path)
#Strip column names
df.columns = df.columns.str.strip()
#Rename columns
for col in df.columns:
    if 'hours' in col:
        df.rename(columns={col: 'hour'}, inplace=True)
    if 'SpotPrice' in col:
        df.rename(columns={col: 'SpotPrice/MWh'}, inplace=True)
    if 'Wind_Speed_m/s' in col:
        df.rename(columns={col: 'Wind_Speed_m/s'}, inplace=True)
    if 'Wind_Direction_deg' in col:
        df.rename(columns={col: 'Wind_Direction'}, inplace=True)

# Year referencing
df['year'] = pd.to_datetime(df['hour'], errors='coerce').dt.year
#ignore 2025 data

Annual_ref_spot = df.groupby('year')['SpotPrice/MWh'].mean().reset_index()
Annual_ref_spot =Annual_ref_spot[Annual_ref_spot['year'] < 2036]
Annual_ref_spot.rename(columns={'SpotPrice/MWh': 'Annual_Reference_Price_€/MWh_raw'}, inplace=True)
#Include the offshore correction factor
correction_factor = 0.9
Annual_ref_spot['Annual_Reference_Price_€/MWh'] = Annual_ref_spot['Annual_Reference_Price_€/MWh_raw'] * correction_factor
Annual_ref_spot = Annual_ref_spot[['year', 'Annual_Reference_Price_€/MWh']]
Annual_ref_spot.rename(columns={'year': 'Year'}, inplace=True)
# Save the combined DataFrame to a new CSV file
#View the DataFrame
output_path = "D:/Thesis_Project/thesis_data/results/NL/EP/High/Annual_Reference_Price_NL_High.csv"
Annual_ref_spot.to_csv(output_path, index=False)
print(Annual_ref_spot)
