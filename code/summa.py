#This is for monthly aggregation

import pandas as pd
input_path = "C:/Thesis_Project/thesis_data/results/SpotPriceSimulated/support_payment_FiP_DK_sim.csv"
df = pd.read_csv(input_path, parse_dates=['hour'])
# Strip whitespace from column names
df.columns = df.columns.str.strip()

#Rename columns for clarity

for col in df.columns:
    if 'WindSpeed' in col:
        df.rename(columns={col: 'WindSpeed_m_s'}, inplace=True)

for col in df.columns:
    if 'ActualGeneration_MWh' in col:
        df.rename(columns={col: 'Total_Actual_Generation_MWh'}, inplace=True)

for col in df.columns:
    if 'SpotPrice' in col:
        df.rename(columns={col: 'Spot_Price_€/MWh'}, inplace=True)
for col in df.columns:
    if 'SupportPayment' in col:
        df.rename(columns={col: 'SupportPayment_€'}, inplace=True)
#Monthly Aggregation
df['month'] = pd.to_datetime(df['hour']).dt.to_period('M').astype(str)
df_monthly = df.groupby('month').agg({
    'Spot_Price_€/MWh': 'mean',
    'WindSpeed_m_s': 'mean',
    'SupportPayment_€': 'sum',
    'Total_Actual_Generation_MWh': 'sum'
}).reset_index()

#Print the output
print(df_monthly)
output_path = "C:/Thesis_Project/thesis_data/results/SpotPriceSimulated/monthly_support_revenue_FiP_DK_sim.csv"
df_monthly.to_csv(output_path, index=False)
print(f"Monthly market revenue saved to {output_path}")
