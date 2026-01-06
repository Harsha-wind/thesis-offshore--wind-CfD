
import pandas as pd
import numpy as np
input_path = "O:/Thesis_Project/thesis_data/data/DK_EP/hourly_wind_speed_direction_price_DK.csv"
df = pd.read_csv(input_path)
df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
df = df.sort_values('datetime')

#Extrapolate wind_speed_100m to hub height
def extrapolate_wind_speed(ws_10m, z_10m=10, z_hub=119, alpha=0.11):
    return ws_10m * (z_hub / z_10m) ** alpha


# Apply the extrapolation function to the wind speed column
df['wind_speed_hub'] = extrapolate_wind_speed(df['wind_speed_10m'])
df_out = df[['datetime', 'wind_dir_deg', 'wind_speed_hub']].copy()
# Save the updated DataFrame to a new CSV file
out_path = "O:/Thesis_Project/thesis_data/data/DK_EP/hourly_wind_speed_hub_height_DK.csv"
df_out.to_csv(out_path, index=False)
print("df_out saved to:", out_path)
print(df_out.head())
