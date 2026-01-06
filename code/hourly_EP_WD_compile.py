import pandas as pd
import numpy as np

# ======= paths =======
price_path  = "C:/Thesis_Project/thesis_data/results/NL/EP/SpotPrices_NL_merged_cleaned.csv"
wind_path   = "C:/Thesis_Project/thesis_data/data/NL_wind/hourly_wind_speed_hub_height_NL.csv"

#Load the data
df_price = pd.read_csv(price_path)
df_wind = pd.read_csv(wind_path)

#Strip column names
df_price.columns = df_price.columns.str.strip()
df_wind.columns = df_wind.columns.str.strip()

#Convert datetime to datetime type
df_price['datetime'] = pd.to_datetime(df_price['datetime'], utc=True, errors='coerce')
df_wind['datetime'] = pd.to_datetime(df_wind['datetime'], utc=True, errors='coerce')

#Merge on datetime
df_merged = pd.merge(df_price, df_wind, on='datetime', how='inner')
df_merged = df_merged.sort_values('datetime')
output_path = "C:/Thesis_Project/thesis_data/results/NL/EP/merged_wind_SP_NL_cleaned.csv"
df_merged.to_csv(output_path, index=False)
print("df_merged saved to:", output_path)
print(df_merged.head())
