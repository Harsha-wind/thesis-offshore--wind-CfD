import pandas as pd
import numpy as np

input_path = "D:/Thesis_Project/thesis_data/results/DK/Cap_Cfd/high/Capability_generation_DK_high.csv"
df = pd.read_csv(input_path)
df.columns = df.columns.str.strip()

#Rename column
for col in df.columns:
    if 'hours' in col:
        df.rename(columns={col: 'Month'}, inplace=True)
for col in df.columns:
    if 'SpotPrice/MWh' in col:
        df.rename(columns={col: 'AvgSpotPrice_€/MWh'}, inplace=True)
for col in df.columns:
    if 'WindSpeed_m/s' in col:
        df.rename(columns={col: 'AvgWindSpeed_m/s'}, inplace=True)
for col in df.columns:
    if 'TotalCapability_MWh_withwake' in col:
        df.rename(columns={col: 'CapabilityGeneration_MWh'}, inplace=True)


# Convert 'Month' to datetime and set as index
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Monthly total Market Revenue (sum of hourly values)

wind_speed = df['AvgWindSpeed_m/s'].resample('M').mean().reset_index()
wind_speed.columns = ['Month', 'AvgWindSpeed_m/s']
# Monthly monthly actual generation (sum of hourly values)

#monthly_revenue = df['CapabilityGeneration_MWh'].resample('M').sum().reset_index()
#monthly_revenue.columns = ['Month', 'CapabilityGeneration_MWh']
# Monthly spot price based on baseline average
monthly_spot_price = df['AvgSpotPrice_€/MWh'].resample('M').mean().reset_index()
monthly_spot_price.columns = ['Month', 'AvgSpotPrice_€/MWh']

# Monthly actual generation (sum of hourly values)
monthly_generation = df['CapabilityGeneration_MWh'].resample('M').sum().reset_index()
monthly_generation.columns = ['Month', 'CapabilityGeneration_MWh']

# Combine if needed
#monthly_df = pd.merge(monthly_revenue, monthly_spot_price, on='Month')
monthly_df = pd.merge(monthly_spot_price, wind_speed, on='Month')
monthly_df = pd.merge(monthly_df, monthly_generation, on='Month')
# Convert 'Month' column from end-of-month date to 'YYYY-MM' format
monthly_df['Month'] = monthly_df['Month'].dt.strftime('%Y-%m')
#view the result
print(monthly_df)

output_path = "D:/Thesis_Project/thesis_data/results/DK/Cap_Cfd/high/Monthly_Capability_Generation_DK_(High).csv"
monthly_df.to_csv(output_path, index=False)
print(f"Monthly capability generation saved to {output_path}")
print(monthly_df.head())
