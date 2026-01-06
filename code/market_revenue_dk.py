import pandas as pd
import matplotlib.pyplot as plt
from plot_style import set_style
set_style()

# === File paths ===
SP_path = "D:/Thesis_Project/thesis_data/results/DK/Cap_Cfd/Tech_specific/high/Capability_generation_DK_CfD_support_payment_high_techref.csv"
MR_path = "D:/Thesis_Project/thesis_data/results/DK/FiP/high/Monthly_Market_Revenue_DK_(High).csv"


# === Load CSVs ===
df_MR = pd.read_csv(MR_path)
df_SP = pd.read_csv(SP_path)

# === Clean column names ===
df_MR.columns = df_MR.columns.str.strip()
df_SP.columns = df_SP.columns.str.strip()

# === Rename spot price column to a clean name
for col in df_SP.columns:
    if 'AvgSpotPrice_techref' in col:
        df_SP.rename(columns={col: 'ReferencePrice_€/MWh'}, inplace=True)
        print(f"Renaming '{col}' to 'ReferencePrice_€/MWh'")
for col in df_SP.columns:
    if 'SupportPayment' in col:
        df_SP.rename(columns={col: 'SupportPayment_€'}, inplace=True)
        print(f"Renaming '{col}' to 'SupportPayment_€'")
for col in df_SP.columns:
    if 'Total_Capability_MWh' in col:
        df_SP.rename(columns={col: 'Capability_Generation_MWh'}, inplace=True)
        print(f"Renaming '{col}' to 'Capability_Generation_MWh'")
for col in df_MR.columns:
    if 'Monthly_Market_Revenue' in col:
        df_MR.rename(columns={col: 'Market_Revenue_€'}, inplace=True)
        print(f"Renaming '{col}' to 'Market_Revenue_€'")
for col in df_MR.columns:
    if 'ActualGeneration_MWh' in col:
        df_MR.rename(columns={col: 'Actual_Generation_MWh'}, inplace=True)
        print(f"Renaming '{col}' to 'Actual_Generation_MWh'")

keep_MR = ['Month', 'Market_Revenue_€', 'Actual_Generation_MWh']
df_MR = df_MR[keep_MR]

keep_SU = ['Month', 'SupportPayment_€', 'ReferencePrice_€/MWh', 'Capability_Generation_MWh']
df_SP = df_SP[keep_SU]
# Need to make sure both the hour column are in string type
#df_MR["month"] = pd.to_datetime(df_MR["month"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")
#df_SP["month"] = pd.to_datetime(df_SP["month"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")

# === Merge DataFrames on 'Month' column ===
df_output = df_MR.merge(df_SP, on='Month', how='left')
df_output['TotalRevenue_€'] = df_output['Market_Revenue_€'] + df_output['SupportPayment_€']
#Group by month and sum the TotalRevenue_€
#df_output['hour'] = pd.to_datetime(df_output['hour'], errors="coerce")
#df_output['month'] = df_output['hour'].dt.to_period('M').dt.to_timestamp()
#df_output = df_output.groupby('month').agg({
#    'Market_Revenue_€': 'sum',
#    'SupportPayment_€': 'sum',
#   'TotalRevenue_€': 'sum',
#    'SpotPrice_€/MWh': 'mean',
#   'ActualGeneration_MWh': 'sum'
#}).reset_index()

# Save the DataFrame to a new CSV file
output_path = "D:/Thesis_Project/thesis_data/results/DK/Cap_Cfd/Tech_specific/high/Total_Revenue_monthly_Cap_Cfd_DK_high_techref.csv"
df_output.to_csv(output_path, index=False)
print(f"Monthly total revenue data saved to {output_path}")

print(df_output.head())


# Group by year and sum the market revenue
#df_output['month'] = pd.to_datetime(df_output['month'], dayfirst=True, errors='coerce')
#df_output['year'] = df_output['month'].dt.year
#yearly_revenue = df_output.groupby('year')['TotalRevenue_€'].sum().reset_index()
#ignore the 2025 data
#yearly_revenue = yearly_revenue[yearly_revenue['year'] < 2025]
#View the few results
#print(yearly_revenue)

#Plot
# Match your Overleaf body size: 10 / 11 / 12
#set_style(body_pt=10, theme="latex_like")
#df_output['year'] = pd.to_datetime(df_output['year'], errors='coerce')
# Now you can pick colors per plot
#plt.plot(df_output["year"], df_output["TotalRevenue_€"], 
#        label="Capability-CfD", marker="o", color="darkblue")

#plt.title("Annual Revenue Flow - Capability-CfD-NL (€)")
#plt.xlabel("Year")
#plt.ylabel("Annual Total Revenue (€)")
#plt.legend()
  # Set y-axis limits for better visibility
#plt.tight_layout()

# === Save as PDF for Overleaf ===
#output_plot_path = "C:/Thesis_Project/thesis_data/results/SpotPriceSimulated/Current_DK/Monthly_Total_Revenue_Flow_Sliding_FiP_sim.pdf"
#plt.savefig(output_plot_path, bbox_inches='tight')
#print(f"PDF plot saved to: {output_plot_path}")
#
# plt.show()








#plt.figure(figsize=(12, 6))
#plt.plot(df_output['month'], df_output['TotalRevenue_€'], marker='o', label='Total Revenue (€/MWh)')
#plt.title('Monthly Total Revenue (€/MWh) - FiP_DK')
#plt.xlabel('Month')
#plt.ylabel('Total Revenue (€/MWh)')
#plt.xticks(rotation=45)

#plt.legend()
#plt.tight_layout()
#plot_path = "C:/Thesis_Project/thesis_data/results/SpotPriceSimulated/Current_DK/monthly_total_revenue_FiP_sim.png"
#plt.savefig(plot_path)
#plt.show()

# === Plotting ===
# Match your Overleaf body size: 10 / 11 / 12

