import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from plot_style import set_style
set_style()
ref_path = "C:/Thesis_Project/thesis_data/results/NL/EP/Low_Scenario/Annual_Reference_Price_NL_LC.csv"
df_ref = pd.read_csv(ref_path)
df_ref.columns = df_ref.columns.str.strip()
Strike_Price = 54.5  # EUR/MWh
#Rename columns
for col in df_ref.columns:
    if 'Annual_Reference_Price' in col:
        df_ref.rename(columns={col: 'Annual_Reference_Price_€/MWh'}, inplace=True)

#Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df_ref['Year'], df_ref['Annual_Reference_Price_€/MWh'], marker='o', label='Annual Reference Price', color='blue')
ax.axhline(y=Strike_Price, color='red', linestyle='--', label='Strike Price (54.5 €/MWh)')
#Annotations    
for i, row in df_ref.iterrows():
    ax.annotate(f"{row['Annual_Reference_Price_€/MWh']:.1f}", (row['Year'], row['Annual_Reference_Price_€/MWh']),
                textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
ax.set_xlabel('Year')
ax.set_ylabel('Price (€/MWh)')
ax.set_title('Annual Reference Price vs Strike Price')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.legend()
plt.tight_layout()
#Save
plot_path = "C:/Thesis_Project/thesis_data/results/NL/EP/Low_Scenario/Annual_Reference_Price_vs_Strike_Price_NL_LC.png"
plt.savefig(plot_path, bbox_inches='tight')
print(f"Plot saved to: {plot_path}")
plt.show()