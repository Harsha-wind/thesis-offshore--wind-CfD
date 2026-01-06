import matplotlib.pyplot as plt
import pandas as pd
input_path = "C:/Thesis_Project/thesis_data/results/support_payment_FiP.csv"
# === Load data ===
df = pd.read_csv(input_path, parse_dates=['datetime'])
df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')
df.set_index('datetime', inplace=True)
# Strip column names
df.columns = df.columns.str.strip()
# Rename columns for clarity
for col in df.columns:
    if 'SupportPayment' in col:
        df.rename(columns={col: 'SupportPayment_€'}, inplace=True)
# Resample the hourly Support Payment and its respective hourly datetime to monthly
monthly_support_payment = df.resample('M').sum().reset_index()
monthly_support_payment['Month'] = monthly_support_payment['datetime'].dt.to_period('M').astype(str)
monthly_support_payment = monthly_support_payment[['Month', 'SupportPayment_€']]



#Yearly Aggregation
#Yearly aggregation, Just aggregate till 2024, 2025 has only 3 months (2019-2024)
# values from 2019, 2020, 2021, 2022, 2023, and 2024
monthly_support_payment = monthly_support_payment.rename(columns={'SupportPayment_€': 'SupportPayment_€'})
monthly_support_payment = monthly_support_payment[['Month', 'SupportPayment_€']]   
# Filter for years 2019 to 2025
# Assuming 'Month' is in 'YYYY-MM' format, we can filter based on the year part of the string
# This will keep only the months that start with the specified years
monthly_support_payment = monthly_support_payment[monthly_support_payment['Month'].str.startswith(('2019', '2020', '2021', '2022', '2023', '2024'))]
monthly_support_payment['Year'] = pd.to_datetime(monthly_support_payment['Month']).dt.year
monthly_support_payment = monthly_support_payment.groupby('Year')['SupportPayment_€'].sum().reset_index()

# View all the columns and the values
print(monthly_support_payment.head())


plt.figure(figsize=(14, 6))

monthly_support_payment['Year'] = pd.to_datetime(monthly_support_payment['Year'], format='%Y')
plt.xticks(rotation=45, ha='right')
plt.plot(monthly_support_payment['Year'], monthly_support_payment['SupportPayment_€'], marker='o', linestyle='-', color='green', label='Annual Support Payment (€)')
plt.axhline(y=0, color='black', linewidth=1, linestyle='--')
plt.title('Annual Support Payment (FiP)', pad=15) 
plt.xlabel('Year')
plt.ylabel('Annual Support Payment (€)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.legend()
plot_path = "C:/Thesis_Project/thesis_data/results/support_payment_FiP_annual.png"
plt.savefig(plot_path, dpi=300)
plt.show()
print(f"Plot saved to: {plot_path}")