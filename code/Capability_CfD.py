#Ithu than Capbility-based Contract for Difference (CfD) support payment model for wind energy in Denmark
import pandas as pd
import os
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.dates as mdates


#. Input Configuration ===
strike_price = 103.1  # €/MWh (Set your fixed strike price here)

# Input file paths (update as needed)
input_path = "D:/Thesis_Project/thesis_data/results/DK/Cap_Cfd/Tech_specific/high/monthly_tech_specific_ref_price_high_dk.csv"  # Market revenue data

# Output file path
output_path = "D:/Thesis_Project/thesis_data/results/DK/Cap_Cfd/Tech_specific/high/Capability_generation_DK_CfD_support_payment_high_techref.csv"
#plot_path = "C:/Thesis_Project/thesis_data/results/yearly_cfd_support_payment_corrected_techref(DK).png"

# Load Data
# Assumes each CSV has a 'Month' column (e.g., "01-01-2019") and one data column

df_in = pd.read_csv(input_path)

print("Input DataFrame columns:", df_in.columns.tolist())

# Ensure column names are stripped
df_in.columns = df_in.columns.str.strip()

# === Rename for consistency ===
#for col in df_in.columns:
#    if 'Month' in col:
#        df_in.rename(columns={col: 'month'}, inplace=True)
for col in df_in.columns:
    if 'tech_specific_ref_price' in col:
        df_in.rename(columns={col: 'AvgSpotPrice_techref_€/MWh'}, inplace=True)
for col in df_in.columns:
    if 'cap_sum' in col:
        df_in.rename(columns={col: 'Total_Capability_MWh'}, inplace=True)


df_out = df_in.copy()
# Now compute payment AFTER merge
df_out['StrikePrice_€/MWh'] = strike_price
df_out['PriceDiff'] = df_out['StrikePrice_€/MWh'] - df_out['AvgSpotPrice_techref_€/MWh']
df_out['SupportPayment_€'] = df_out['PriceDiff'] * df_out['Total_Capability_MWh']

#Print first few rows of the DataFrame
print(df_out[['Month', 'AvgSpotPrice_techref_€/MWh', 'Total_Capability_MWh', 'SupportPayment_€']].head())

#Save only required columns
df_out = df_out[['Month', 'AvgSpotPrice_techref_€/MWh', 'Total_Capability_MWh', 'SupportPayment_€']]
# Save the DataFrame to a new CSV file
df_out.to_csv(output_path, index=False)

print(f"Output DataFrame saved to: {output_path}")

#Plotting
#df_out['month'] = pd.to_datetime(df_out['month'], format='%Y-%m')

# Now plot yearly support payment
#df_out['year'] = df_out['month'].dt.year
#df_yearly = df_out.groupby('year')['SupportPayment_€'].sum().reset_index()
#plt.figure(figsize=(14, 6))  # Wider and slightly shorter for a zoomed-in look

#plt.plot(df_yearly['year'], df_yearly['SupportPayment_€'], marker='o', linestyle='-', color='blue', label='Support Payment')
#plt.axhline(y=0, color='black', linewidth=1, linestyle='--')



# Labels and styling
#plt.title('Total Support Payment Over Years', pad=15)
#plt.xlabel('Month')
#plt.ylabel('Total Support Payment (€)')
#plt.xticks(rotation=45, ha='right')
#plt.grid(True, which='major', linestyle='-', linewidth=0.5)
#plt.grid(True, which='minor', linestyle=':', linewidth=0.3)
#plt.legend()
#plt.tight_layout()

#plt.savefig(plot_path, dpi=300)
#print(f"Plot saved to: {plot_path}")
#plt.show()




#strike price = 0.1031 # €/kWh (103.1 €/MWh)
# Convert to MWh for consistency 
# 
# Plotting
# Convert Month to datetime if needed
#df['Month_dt'] = pd.to_datetime(df['month'])
#Spot Price vs Strike Price Plot
#plt.figure(figsize=(12,6))
#plt.plot(df['Month_dt'], df['SpotPrice_€/MWh'], label='Monthly Spot Price', marker='o')
#plt.axhline(y=103.1, color='r', linestyle='--', label='Strike Price (€103.1/MWh)')

# Fill area where spot > strike
#above_strike = df[df['SpotPrice_€/MWh'] > 103.1]
#plt.fill_between(
#    above_strike['Month_dt'], 103.1, above_strike['SpotPrice_€/MWh'],
#    color='orange', alpha=0.3, label='Payment from Generator'
#)

#plt.title('Monthly Spot Price vs Strike Price')
#plt.xlabel('Month')
##plt.ylabel('€/MWh')
#plt.xticks(rotation=45)
#plt.legend()
#plt.grid(True)
#plt.tight_layout()

#plt.savefig("C:/Thesis_Project/thesis_data/results/spot_vs_strike_visual.png", dpi=300)
#print(f"Plot saved to: {plot_path}")
#plt.show()




