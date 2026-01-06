import os
import pandas as pd

# Define file paths for each year's data
input_path3 = "C:/Thesis_Project/thesis_data/data/NL_wind/NL(100m)/ERA5_2021_NL_100m.csv"
input_path4 = "C:/Thesis_Project/thesis_data/data/NL_wind/NL(100m)/ERA5_2022_NL_100m.csv"
input_path5 = "C:/Thesis_Project/thesis_data/data/NL_wind/NL(100m)/ERA5_2023_NL_100m.csv"
input_path6 = "C:/Thesis_Project/thesis_data/data/NL_wind/NL(100m)/ERA5_2024_NL_100m.csv"
input_path7 = "C:/Thesis_Project/thesis_data/data/NL_wind/NL(100m)/ERA5_2025_NL_100m.csv"
# Load the data its a xlsx file

df3 = pd.read_csv(input_path3)
df4 = pd.read_csv(input_path4)
df5 = pd.read_csv(input_path5)
df6 = pd.read_csv(input_path6)
df7 = pd.read_csv(input_path7)

df = pd.concat([df3, df4, df5, df6, df7], ignore_index=True)
df.columns = df.columns.str.strip()


# Convert the correct time column to datetime (assumes it's 'time')
df['time'] = pd.to_datetime(df['time'], utc=True)

#Rename column
for col in df.columns:
    if 'time' in col:
        df.rename(columns={col: 'datetime'}, inplace=True)
# Set time as index
df.set_index('datetime', inplace=True)
#convert xlsx to csv
output_path = "C:/Thesis_Project/thesis_data/data/NL_wind/NL(100m)/ERA5_2021_2025_NL_100m.csv"
df.to_csv(output_path)
print(f"Data saved to: {output_path}")
#Show the first few rows to verify
print(df.head())
