import pandas as pd
import matplotlib.pyplot as plt

# Load data
input_path = "D:/Thesis_Project/thesis_data/results/DK/FiP/high/hourly_actual_generation_withwake_NL_(moderated_dk).csv"
df_input = pd.read_csv(input_path, parse_dates=['hours'])
#spot_path = "C:/Thesis_Project/thesis_data/results/SpotPriceSimulated/capability/capability_generation_hourly_with_wake_sim.csv"
#df_spot = pd.read_csv(spot_path, parse_dates=['hour'])
#output save
#output_path = "C:/Thesis_Project/thesis_data/results/Wake/hourly_actual_generation_simulated.csv"
#plot_path = "C:/Thesis_Project/thesis_data/results/Wake/hourly_actual_generation_simulated.png"

# The datetime column is in this format 01-04-2025  00:00:00
# Make sure all all datetime values are there in the datetime column
df_input['hours'] = pd.to_datetime(df_input['hours'], dayfirst=True, errors='coerce')

df_input = df_input.dropna(subset=['hours'])

#Rename datetime columns for clarity
df_input.rename(columns={'hours': 'hours'}, inplace=True)
#Rename column
for col in df_input.columns:
    if 'Actual_MWh' in col:
        df_input.rename(columns={col: 'ActualGeneration_MWh'}, inplace=True)

for col in df_input.columns:
    if 'SpotPrice' in col:
        df_input.rename(columns={col: 'SpotPrice/MWh'}, inplace=True)

for col in df_input.columns:
    if 'Wind_Speed_m/s' in col:
        df_input.rename(columns={col: 'WindSpeed_m/s'}, inplace=True)
for col in df_input.columns:
    if 'Wind_Direction_deg' in col:
        df_input.rename(columns={col: 'WindDirection_deg'}, inplace=True)
df = df_input.copy()
# === Merge on timestamp ===
#df = pd.merge(df_input[['hour','WindSpeed','ActualGeneration_MWh']],
#              df_spot[['hour','SpotPrice','WindSpeed']],
#              on='hour',
#              how='inner',
#              suffixes=('_cap','_spot'))

# keep just one wind speed (they look identical between files)
#df['WindSpeed'] = df['WindSpeed_cap']
#df.drop(columns=['WindSpeed_cap','WindSpeed_spot'], inplace=True)

# === Apply losses ===
#Denmark 
electrical_loss = 0.02
other_loss      = 0.01
availability    = 0.9532
loss_factor = (1 - electrical_loss) * (1 - other_loss) * availability

df['Actual_Generation_MWh'] = df['ActualGeneration_MWh'] * loss_factor

# === Curtailment ===
#df.loc[df['SpotPrice/MWh'] <= 0, 'Actual_Generation_MWh'] = 0.0 #This is for DK market


# === Final tidy columns ===
df_out = df[['hours','SpotPrice/MWh','WindSpeed_m/s','Actual_Generation_MWh']]

print(df_out.head())
df_out.to_csv("D:/Thesis_Project/thesis_data/results/DK/FiP/high/hourly_actual_generation_wake_DK_high.csv", index=False)
print("Output saved to D:/Thesis_Project/thesis_data/results/DK/FiP/high/hourly_actual_generation_wake_DK_high.csv")
