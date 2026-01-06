import pandas as pd

def preprocess_electricity_prices(input_path: str, output_path: str):
    # Load data
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()
    
    # Convert datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    #Rename columns
    for col in df.columns:
        if 'SpotPrice' in col:
            df.rename(columns={col: 'spot_price_eur_per_mwh'}, inplace=True)
    
    # Keep only needed columns
    df = df[['datetime', 'spot_price_eur_per_mwh']].copy()
    
    # Sort by datetime
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Remove duplicates (keep first)
    df = df.drop_duplicates(subset=['datetime'], keep='first')
    
    # Create complete hourly range
    full_range = pd.date_range(
        start=df['datetime'].min(),
        end=df['datetime'].max(),
        freq='h'
    )
    
    existing_hours = set(df['datetime'])
    missing_hours = sorted(set(full_range) - existing_hours)
    
    if missing_hours:
        print(f"⚠️ WARNING: {len(missing_hours)} hours missing in input file!")
        print(f"First 10 missing hours: {missing_hours[:10]}")
    else:
        print(f"✓ No missing hours in input file")
    print(f"✓ All hours covered: {len(existing_hours)}")






    # Reindex to include all hours
    df = df.set_index('datetime').reindex(full_range).reset_index()
    df = df.rename(columns={'index': 'datetime'})
    
    # Forward fill missing prices (use previous hour's price)
    df['spot_price_eur_per_mwh'] = df['spot_price_eur_per_mwh'].fillna(method='ffill')
    
    # Save
    df.to_csv(output_path, index=False)
    
    print(f"Processed {len(df)} hours")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Missing values: {df['spot_price_eur_per_mwh'].isna().sum()}")
    
    return df

# Run
input_file = "E:/Thesis_Project/thesis_data/results/merged_wind_speed_direction_price_DK.csv"
output_file = "E:/Thesis_Project/thesis_data/results/DK/EP/SpotPrices_DK_merged_cleaned.csv"

df_clean = preprocess_electricity_prices(input_file, output_file)
print("\nFirst 10 rows:")
print(df_clean.head(10))