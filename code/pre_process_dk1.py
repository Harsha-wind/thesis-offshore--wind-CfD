import pandas as pd
import pandas as pd

# Load your merged file
df = pd.read_csv("C:/Thesis_Project/thesis_data/data/NL_wind/hourly_wind_speed_hub_height_NL.csv")
df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
df = df.sort_values('datetime')

# 1. Check for missing datetimes
expected = pd.date_range(start=df['datetime'].min(),
                         end=df['datetime'].max(),
                         freq='H', tz='UTC')
missing_times = expected.difference(df['datetime'])

print(f"Missing timestamps: {len(missing_times)}")
if len(missing_times) > 0:
    print(missing_times[:10])  # show first 10

# 2. Check NaNs
print("\nNaN counts per column:")
print(df.isna().sum())

# 3. Check for abnormal values
print("\nWind speed min/max:")
print(df['wind_speed_hub'].min(), df['wind_speed_hub'].max())

print("\nWind direction min/max:")
print(df['wind_dir_deg'].min(), df['wind_dir_deg'].max())
