#Support Payment for the Feed in Premium
import pandas as pd
import matplotlib.pyplot as plt

# === USER CONFIGURATION ===
strike_price = 103.1  # €/MWh
input_path = "D:/Thesis_Project/thesis_data/results/DK/FiP/high/hourly_actual_generation_wake_DK_high.csv"

# === Load data ===
df = pd.read_csv(input_path, parse_dates=['hours'])
df['hours'] = pd.to_datetime(df['hours'], dayfirst=True, errors='coerce')

df.set_index('hours', inplace=True)

#Strip column names
df.columns = df.columns.str.strip()
# Rename columns for clarity
for col in df.columns:
    if 'Actual_Generation_MWh' in col:
        df.rename(columns={col: 'ActualGeneration_MWh'}, inplace=True)
for col in df.columns:
    if 'SpotPrice/MWh' in col:
        df.rename(columns={col: 'SpotPrice_€/MWh'}, inplace=True)


# Support Payment Calculation

df['SupportPayment_€'] = (strike_price - df['SpotPrice_€/MWh']) * df['ActualGeneration_MWh']
# Spot price is negative, set support payment to zero
df.loc[df['SpotPrice_€/MWh'] < 0, 'SupportPayment_€'] = 0
#Spot price is greater than strike price, set support payment to zero
df.loc[df['SpotPrice_€/MWh'] > strike_price, 'SupportPayment_€'] = 0

# View the index values 26021 to 26030, the rows values 26021 to 26030, and the columns

print(df.iloc[2:10])
print(df.columns)

output_path = "D:/Thesis_Project/thesis_data/results/DK/FiP/high/support_payment_FiP_DK_high.csv"
# Save the DataFrame to a new CSV file
#the columns are datetime, ActualGeneration_MWh, SpotPrice_€/MWh, SupportPayment_€
df.to_csv(output_path, index = True)      




