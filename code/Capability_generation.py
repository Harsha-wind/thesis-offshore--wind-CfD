import pandas as pd
import numpy as np
import yaml
from scipy.interpolate import interp1d

print("="*60)
print("WIND FARM CAPABILITY ESTIMATION")
print("="*60)

# ==============================================================================
# 1. Load Turbine Power Curve (IEA 10MW Reference Turbine)
# ==============================================================================

with open("IEA37_10MW_turbine.yaml", 'r') as file:
    turbine_data = yaml.safe_load(file)

# Extract power curve: wind speed (m/s) → power output (MW)
wind_speeds = np.array(turbine_data['performance']['power_curve']['power_wind_speeds'])
power_watts = np.array(turbine_data['performance']['power_curve']['power_values'])
power_MW = power_watts / 1_000_000  # Convert W to MW

# Create interpolation function for any wind speed
power_curve = interp1d(wind_speeds, power_MW, bounds_error=False, fill_value=0)

print(f"✓ Loaded turbine power curve ({len(wind_speeds)} data points)")

# ==============================================================================
# 2. Load Hourly Wind Speed Data
# ==============================================================================

df = pd.read_csv("merged_wind_price_full_high.csv", parse_dates=['hours'])

# Standardize column names
df.rename(columns={
    'Wind_Speed_m/s': 'WindSpeed',
    'SpotPrice': 'SpotPrice',
    'Wind_Direction_deg': 'WindDirection'
}, inplace=True)

print(f"✓ Loaded {len(df)} hours of wind data")

# ==============================================================================
# 3. Calculate Capability (Theoretical Maximum Generation)
# ==============================================================================

# Single turbine output per hour (MW for 1 hour = MWh)
df['SingleTurbine_MWh'] = df['WindSpeed'].apply(power_curve)

# Wind farm parameters
NUM_TURBINES = 74
WAKE_LOSS = 0.1075  # 10.75% energy loss from wake effects

# Total wind farm capability with wake losses
df['Capability_MWh'] = (
    df['SingleTurbine_MWh'] * NUM_TURBINES * (1 - WAKE_LOSS)
)

print(f"\n✓ Calculated capability for {NUM_TURBINES} turbines")
print(f"  Wake loss applied: {WAKE_LOSS*100:.2f}%")
print(f"  Average hourly capability: {df['Capability_MWh'].mean():.2f} MWh")

# ==============================================================================
# 4. Save Results
# ==============================================================================

output = df[['hours', 'WindSpeed', 'SpotPrice', 'WindDirection', 'Capability_MWh']]
output.to_csv("capability_generation_DK_high.csv", index=False)

print(f"\n✓ Saved to: capability_generation_DK_high.csv")
print("="*60)