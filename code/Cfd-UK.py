#Support Payment for the Feed in Premium
import pandas as pd
import matplotlib.pyplot as plt

# === USER CONFIGURATION ===
strike_price = 103.1  # €/MWh
input_path = "C:/Thesis_Project/thesis_data/results/Capability_Cfd/hourly_actual_generation_wake.csv"

# === Load data ===
df = pd.read_csv(input_path, parse_dates=['hour'])
df['hour'] = pd.to_datetime(df['hour'], dayfirst=True, errors='coerce')

df.set_index('hour', inplace=True)

#Strip column names
df.columns = df.columns.str.strip()
# Rename columns for clarity
for col in df.columns:
    if 'Actual_Generation_MWh' in col:
        df.rename(columns={col: 'ActualGeneration_MWh'}, inplace=True)
for col in df.columns:
    if 'SpotPrice' in col:
        df.rename(columns={col: 'SpotPrice_€/MWh'}, inplace=True)


# Support Payment Calculation

df_out = df.copy()
# Now compute payment AFTER merge
df_out['StrikePrice_€/MWh'] = strike_price
df_out['PriceDiff'] = df_out['StrikePrice_€/MWh'] - df_out['SpotPrice_€/MWh']
df_out['SupportPayment_€'] = df_out['PriceDiff'] * df_out['ActualGeneration_MWh']



# View the index values 26021 to 26030, the rows values 26021 to 26030, and the columns

print(df.iloc[2:10])
print(df.columns)

#output_path = "C:/Thesis_Project/thesis_data/results/Current_Support/support_payment_FiP_updated.csv"
# Save the DataFrame to a new CSV file
#the columns are datetime, ActualGeneration_MWh, SpotPrice_€/MWh, SupportPayment_€
#df.to_csv(output_path, index = True)   
# plot
import matplotlib.pyplot as plt
#focus only one day, like 24hrs

df_day = df_out.loc['2025-01-05']
#plot
plt.figure(figsize=(12, 6))
plt.plot(df_day.index, df_day['SupportPayment_€'], marker='o', label='Support Payment (CfD)')
plt.title('Support Payment (CfD) on 2025-01-05')
plt.xlabel('Hour')
plt.ylabel('Support Payment (CfD) in €')
plt.xticks(rotation=45)

plt.legend()
plt.tight_layout()
output_path = "C:/Thesis_Project/thesis_data/results/support_payment_CfD_2025-01-05_UK.png"
plt.savefig(output_path)
plt.show()

#plot spotprice vs the hour
plt.figure(figsize=(12, 6))
plt.plot(df_day.index, df_day['SpotPrice_€/MWh'], marker='o', label='Spot Price (€/MWh)')
plt.title('Spot Price (€/MWh) on 2025-01-05')
plt.xlabel('Hour')
plt.ylabel('Spot Price (€/MWh)')
plt.xticks(rotation=45)

plt.legend()
plt.tight_layout()
output_path1 = "C:/Thesis_Project/thesis_data/results/spot_price_CfD_2025-01-05_UK.png"
plt.savefig(output_path1)
plt.show()