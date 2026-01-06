import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# === User Inputs ===

#support_path = "C:/Thesis_Project/thesis_data/results/monthly_cfd_support_payment_Ccfd_sim_new.csv"  # support payment data
input_path = "D:/Thesis_Project/thesis_data/results/DK/FiP/high/hourly_actual_generation_wake_DK_high.csv"  # input data
output_path = "D:/Thesis_Project/thesis_data/results/DK/FiP/high/hourly_market_revenue_FiP_DK_high.csv"  # output file path
# === Load Merged Data ===
df_input = pd.read_csv(input_path, parse_dates=['hours'])
#df_support = pd.read_csv(support_path, parse_dates=['hour'])

# Strip column names
df_input.columns = df_input.columns.str.strip()
#df_support.columns = df_support.columns.str.strip()

# Rename support column for clarity
# === Rename columns with fuzzy handling for spot price ===
for col in df_input.columns:
    if 'hours' in col:
        df_input.rename(columns={col: 'hour'}, inplace=True)
        print(f"Renaming '{col}' to 'hour'")
for col in df_input.columns:
    if 'SpotPrice/MWh' in col:
        df_input.rename(columns={col: 'SpotPrice_€/MWh'}, inplace=True)
        print(f"Renaming '{col}' to 'SpotPrice_€/MWh'")
for col in df_input.columns:
    if 'WindSpeed_m/s' in col:
        df_input.rename(columns={col: 'WindSpeed_m_s'}, inplace=True)
        print(f"Renaming '{col}' to 'WindSpeed_m_s'")
for col in df_input.columns:
    if 'Actual_Generation_MWh' in col:
        df_input.rename(columns={col: 'Actual_Generation_MWh'}, inplace=True)
        print(f"Renaming '{col}' to 'Actual_Generation_MWh'")



# When spot price is zero and negative, curtailment is assumed, so generation is zero, in other case its the no issue
df_input.loc[df_input['SpotPrice_€/MWh'] <= 0, 'Actual_Generation_MWh'] = 0

# === Calculate Market Revenue (€/h) ===
df_input['Market_Revenue_€'] = df_input['SpotPrice_€/MWh'] * df_input['Actual_Generation_MWh']

# Print first few rows to check
print("Market Revenue Calculation:")
#Index values and columns from 15 to 20
print(df_input.iloc[15:20][['SpotPrice_€/MWh', 'Actual_Generation_MWh', 'Market_Revenue_€']])
print(df_input[['SpotPrice_€/MWh','WindSpeed_m_s','Actual_Generation_MWh', 'Market_Revenue_€']].index)
#only keep relevant columns
df_input = df_input[['hour', 'SpotPrice_€/MWh','WindSpeed_m_s','Actual_Generation_MWh', 'Market_Revenue_€']]
# === Save the updated DataFrame to a new CSV file ===
df_input.to_csv(output_path, index=False)
print(f"Updated market revenue file saved at:\n{output_path}")
