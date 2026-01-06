import pandas as pd
import numpy as np

# ======= paths =======
input_path1 = "O:/Thesis_Project/thesis_data/data/EP/DK1_EP_2019.csv"
input_path2 = "O:/Thesis_Project/thesis_data/data/EP/DK1_EP_2020.csv"
input_path3 = "O:/Thesis_Project/thesis_data/data/EP/DK1_EP_2021.csv"
input_path4 = "O:/Thesis_Project/thesis_data/data/EP/DK1_EP_2022.csv"
input_path5 = "O:/Thesis_Project/thesis_data/data/EP/DK1_EP_2023.csv"
input_path6 = "O:/Thesis_Project/thesis_data/data/EP/DK1_EP_2024.csv"
input_path7 = "O:/Thesis_Project/thesis_data/data/EP/DK1_EP_2025.csv"

# Load and process each file
dfs = []
for path in [input_path1, input_path2, input_path3, input_path4, input_path5, input_path6, input_path7]:
    try:
        df = pd.read_csv(path)
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Find the datetime column (MTU or datetime)
        datetime_col = None
        for col in df.columns:
            if 'MTU' in col or 'datetime' in col.lower():
                datetime_col = col
                break
        
        # Find the price column
        price_col = None
        for col in df.columns:
            if 'Day-ahead Price' in col or 'price' in col.lower():
                price_col = col
                break
        
        # Extract only the needed columns
        if datetime_col and price_col:
            df = df[[datetime_col, price_col]].copy()
            
            # Extract the START datetime from the range (before the " - ")
            # Example: "01/01/2019 00:00:00 - 01/01/2019 01:00:00" becomes "01/01/2019 00:00:00"
            df[datetime_col] = df[datetime_col].str.split(' - ').str[0]
            
            # Convert to datetime object (DD/MM/YYYY format)
            df[datetime_col] = pd.to_datetime(df[datetime_col], format='%d/%m/%Y %H:%M:%S')
            
            # Convert to UTC timezone and format as ISO string with timezone
            # Format: YYYY-MM-DD HH:MM:SS+00:00
            df[datetime_col] = df[datetime_col].dt.tz_localize('UTC').dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
            
            # Rename columns to standard names
            df.columns = ['datetime', 'spot_price_eur_per_mwh']
            
            dfs.append(df)
            print(f"Processed: {path}")
    except Exception as e:
        print(f"Error processing {path}: {e}")

# Concatenate all DataFrames
df_merged = pd.concat(dfs, ignore_index=True)

# Remove any duplicate rows
df_merged = df_merged.drop_duplicates()

# Sort by datetime (need to convert back to datetime temporarily for sorting)
df_merged['datetime_temp'] = pd.to_datetime(df_merged['datetime'])
df_merged = df_merged.sort_values('datetime_temp')
df_merged = df_merged.drop('datetime_temp', axis=1)
df_merged = df_merged.reset_index(drop=True)

# Save the combined DataFrame to a new CSV file
output_path = "O:/Thesis_Project/thesis_data/data/DK_EP/SpotPrices_DK_merged.csv"
df_merged.to_csv(output_path, index=False)

print(f"\nData saved to: {output_path}")
print(f"Total rows: {len(df_merged)}")
print(f"\nFirst few rows:")
print(df_merged.head(10))
print(f"\nLast few rows:")
print(df_merged.tail(10))