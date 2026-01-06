import pandas as pd
import numpy as np

input_path = "O:/Thesis_Project/thesis_data/data/10_metre/ERA5_2019_2025dk_10m.csv"
df_dir = pd.read_csv(input_path)
df_dir.columns = df_dir.columns.str.strip()

#input_path_ws = "C:/Thesis_Project/thesis_data/results/hourly_wind_price_merged.csv"
#df_ws = pd.read_csv(input_path_ws)
#df_ws.columns = df_ws.columns.str.strip()

# --- Parse as UTC-aware datetimes; DO NOT convert to strings ---
df_dir['datetime'] = pd.to_datetime(df_dir['datetime'], utc=True, errors='coerce')
#df_ws['datetime']  = pd.to_datetime(df_ws['datetime'], dayfirst=True, errors='coerce').dt.tz_localize('UTC')

# --- Compute wind direction (meteo convention: 0°=N, clockwise) ---
df_dir['wind_dir_deg'] = (270.0 - np.degrees(np.arctan2(df_dir['v10'], df_dir['u10']))) % 360.0

# --- Fix 2019 duplicates in df_dir ---
def circular_mean_deg(angles):
    rad = np.deg2rad(angles)
    return (np.degrees(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))) + 360) % 360

is_2022 = df_dir['datetime'].dt.year == 2022

# Option A: circular mean (recommended if direction matters)
#df_dir_2022 = (df_dir.loc[is_2022, ['datetime', 'wind_dir_deg', 'wind_speed_100m']]
#               .groupby('datetime', as_index=False)
#               .agg({'wind_dir_deg': circular_mean_deg,
#                     'wind_speed_100m': 'mean'}))

# Option B: keep first duplicate (uncomment to use instead of A)
# df_dir_2022 = df_dir.loc[is_2022, ['datetime','wind_dir_deg']].drop_duplicates('datetime', keep='first')

# 2020+ already unique; ensure no dupes just in case
#df_dir_rest = (df_dir.loc[~is_2022, ['datetime', 'wind_dir_deg', 'wind_speed_100m']]
#               .drop_duplicates('datetime'))

#f_dir_clean = (pd.concat([df_dir_2022, df_dir_rest], ignore_index=True)
#                  .sort_values('datetime'))

# --- Ensure df_ws has one row per hour ---
#df_ws = df_ws.drop_duplicates('datetime').sort_values('datetime')


df_dir_out = df_dir[['datetime', 'wind_dir_deg', 'wind_speed_10m']].copy()
df_dir_clean = df_dir_out.drop_duplicates('datetime').sort_values('datetime')




# --- Merge ---
#df_merged = pd.merge(df_ws, df_dir_clean, on='datetime', how='inner')
df_out = df_dir_clean[['datetime', 'wind_dir_deg', 'wind_speed_10m']].copy()
# --- Save ---
output_path = "O:/Thesis_Project/thesis_data/data/DK_EP/hourly_wind_speed_direction_price_DK.csv"
df_out.to_csv(output_path, index=False)

# --- Sanity checks ---
#print("df_ws rows:", len(df_ws))
print("df_dir_clean rows:", len(df_dir_clean))
print("merged rows:", len(df_dir_clean))
print(df_dir_clean.head())
