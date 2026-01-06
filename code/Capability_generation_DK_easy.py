import pandas as pd
import numpy as np

import yaml
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# === 1. Load YAML with turbine performance ===
yaml_path = "D:/Thesis_Project/thesis_data/code/IEA37_10MW_turbine.yaml"  # adjust to your absolute path
with open(yaml_path, 'r') as file:
    data = yaml.safe_load(file)

# Access power curve data
wind_speeds = np.array(data['performance']['power_curve']['power_wind_speeds'])  # m/s
power_outputs = np.array(data['performance']['power_curve']['power_values'])     # Watts

# Convert power from W to MW for ease of use
power_outputs_MW = power_outputs / 1000000.0

# Create interpolation function
power_curve_interp = interp1d(wind_speeds, power_outputs_MW, bounds_error=False, fill_value=0)

# === 2. Load your monthly wind speed data ===
csv_path = "D:/Thesis_Project/thesis_data/results/DK/EP/high/merged_wind_price_full_high.csv"  # adjust to your file
df = pd.read_csv(csv_path)

# Expected columns: 'month', 'windSpeed'
df['hours'] = pd.to_datetime(df['hours'])
df['Wind_Speed_m/s'] = df['Wind_Speed_m/s'].astype(float)

#Rename datetime and WindSpeed_mps columns for consistency
for col in df.columns:
    if 'hours' in col:
        df.rename(columns={col: 'hours'}, inplace=True)
    if 'Wind_Speed_m/s' in col:
        df.rename(columns={col: 'WindSpeed_m/s'}, inplace=True)
    if 'SpotPrice' in col:
        df.rename(columns={col: 'SpotPrice/MWh'}, inplace=True)
    if 'Wind_Direction_deg' in col:
        df.rename(columns={col: 'Wind_Direction_deg'}, inplace=True)
# === 3. Estimate power per turbine for each month ===
df['Power_MW'] = df['WindSpeed_m/s'].apply(lambda ws: power_curve_interp(ws))

# Convert Power_MW to MWh for energy estimation
df['Energy_MWh'] = df['Power_MW'] * 1 # One hour wind speed interval 

# Total Energy production of the wind farm
n = 74 # Number of turbines
df['TotalCapability_MWh'] = df['Energy_MWh'] * n  # Assuming 74 turbines

# Apply the Wake
Wake_effect = 0.1075 # Wake = 10.75%
df['TotalCapability_MWh_withwake'] = df['TotalCapability_MWh'] * (1 - Wake_effect)

# Save
#Save hours, WindSpeed, Power_MW, Energy_MWh, TotalCapability_MWh_withwake

output_path = "D:/Thesis_Project/thesis_data/results/DK/Cap_Cfd/high/Capability_generation_DK_high.csv"
df[['hours', 'WindSpeed_m/s', 'SpotPrice/MWh', 'Wind_Direction_deg', 'TotalCapability_MWh_withwake']].to_csv(output_path, index=False)
print(f"Capability generation with wake effect saved to {output_path}")