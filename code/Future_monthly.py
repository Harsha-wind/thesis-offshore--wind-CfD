import pandas as pd

# Load CSV with datetime column
cpa_path = "C:/Thesis_Project/thesis_data/results/hourly_market_revenue_ccfd_DK1_sim.csv"  # Adjust to your file path
df = pd.read_csv(cpa_path, parse_dates=['hours'])

# Ensure datetime is properly parsed
df['hours'] = pd.to_datetime(df['hours'])
df.set_index('hours', inplace=True)

# Strip column names
df.columns = df.columns.str.strip()
# Rename columns for clarity
for col in df.columns:
    if 'SpotPrice/MWh' in col:
        df.rename(columns={col: 'SpotPrice_€/MWh'}, inplace=True)

for col in df.columns:
    if 'ActualGeneration_MWh' in col:
        df.rename(columns={col: 'Actual_Generation_MWh'}, inplace=True)

for col in df.columns:
    if 'WindSpeed_m_s' in col:
        df.rename(columns={col: 'WindSpeed_m_s'}, inplace=True)

for col in df.columns:
    if 'Market_Revenue' in col:
        df.rename(columns={col: 'Market_Revenue_€'}, inplace=True)

# Monthly total Market Revenue (sum of hourly values)

market_revenue = df['Market_Revenue_€'].resample('M').sum().reset_index()
market_revenue.columns = ['Month', 'Total_Market_Revenue_€']
# Optional: Monthly average wind speed
monthly_wind = df['WindSpeed_m_s'].resample('M').mean().reset_index()
monthly_wind.columns = ['Month', 'AvgWindSpeed_m_s']
monthly_generation = df['Actual_Generation_MWh'].resample('M').sum().reset_index()
monthly_generation.columns = ['Month', 'Total_Actual_Generation_MWh']
#Monthly spot price based on baseline average
monthly_spot_price = df['SpotPrice_€/MWh'].resample('M').mean().reset_index()
monthly_spot_price.columns = ['Month', 'AvgSpotPrice_€/MWh']


# Combine if needed
monthly_df = pd.merge(market_revenue, monthly_wind, on='Month')
monthly_df = pd.merge(monthly_df, monthly_spot_price, on='Month')
monthly_df = pd.merge(monthly_df, monthly_generation, on='Month')
# Convert 'Month' column from end-of-month date to 'YYYY-MM' format
monthly_df['Month'] = monthly_df['Month'].dt.strftime('%Y-%m')
#view the result
print(monthly_df)

# Save the monthly output to a new CSV
output_path = "C:/Thesis_Project/thesis_data/results/market_revenue_monthly_Ccfd_sim.csv"
monthly_df.to_csv(output_path, index=False)

# Optional: preview first few rows
print(monthly_df.head())
