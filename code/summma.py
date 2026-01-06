import pandas as pd

# Define the file paths
input_path1 = "E:/Thesis_Project/thesis_data/data/DK_EP/hourly_wind_speed_hub_height_DK.csv"
input_path2 = "E:/Thesis_Project/thesis_data/results/hourly_wind_price_merged.csv"

# Read the CSV files - NO QUOTES around variable names!
spot_price_df = pd.read_csv(input_path2)  # File with SpotPrice/MWh
wind_data_df = pd.read_csv(input_path1)   # File with wind_dir_deg and wind_speed_hub

# Convert datetime in the spot price file to match the wind data format
spot_price_df['datetime'] = pd.to_datetime(spot_price_df['datetime'], 
                                             format='%d-%m-%Y %H:%M')

# Add timezone to match wind_data format
spot_price_df['datetime'] = spot_price_df['datetime'].dt.tz_localize('UTC')

# Convert wind_data datetime to datetime object
wind_data_df['datetime'] = pd.to_datetime(wind_data_df['datetime'])

# Merge the dataframes on datetime column
merged_df = wind_data_df.merge(spot_price_df[['datetime', 'SpotPrice/MWh']], 
                                on='datetime', 
                                how='left')

# Display the first few rows to check the result
print(merged_df.head())

# Save the merged data - NO QUOTES around variable name!
output_path = "E:/Thesis_Project/thesis_data/results/merged_wind_speed_direction_price_DK.csv"
merged_df.to_csv(output_path, index=False)
print(f"\nMerged file saved as '{output_path}'")