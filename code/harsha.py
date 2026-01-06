import pandas as pd
import matplotlib.pyplot as plt
from plot_style import set_style
set_style()
basis_path = "C:/Thesis_Project/thesis_data/results/Capability_Cfd/monthly_total_revenue_Ccfd_dk1.csv" 
ref_path = "C:/Thesis_Project/thesis_data/results/Tech_ref/monthly_total_revenue_Ccfd_techref_dk1.csv"
# === Load your dataset ===
df_base = pd.read_csv(basis_path, parse_dates=['month'], dayfirst=True)
df_ref = pd.read_csv(ref_path, parse_dates=['month'], dayfirst=True)
df_base['month'] = pd.to_datetime(df_base['month'], format="%Y-%d-%m", errors="coerce")

# Now reformat to YYYY-MM-DD string
df_base['month'] = df_base['month'].dt.strftime("%Y-%m-%d")

# For consistency, make sure tech-ref is also parsed and formatted
df_ref['month'] = pd.to_datetime(df_ref['month'], format="%Y-%m-%d", errors="coerce")

for col in df_ref.columns:
    if 'Techspecific' in col:
        df_ref.rename(columns={col: 'SpotPrice(techref)_€/MWh'}, inplace=True)
for col in df_base.columns:
    if 'SpotPrice' in col:
        df_base.rename(columns={col: 'SpotPrice(baseline)_€/MWh'}, inplace=True)
#Make sure both month columns are in period format
df_base['month'] = pd.to_datetime(df_base['month'], dayfirst=True, errors='coerce')
df_ref['month'] = pd.to_datetime(df_ref['month'], dayfirst=True, errors='coerce')
#Keep only month and SpotPrice columns
monthly_revenue_baseline = df_base[['month', 'SpotPrice(baseline)_€/MWh']]
monthly_revenue_techref = df_ref[['month', 'SpotPrice(techref)_€/MWh']]
#show values
print(monthly_revenue_baseline)
print(monthly_revenue_techref)
#Yearly aggregation
#df_base['month'] = pd.to_datetime(df_base['month'], dayfirst=True, errors='coerce')
#df_base['year'] = df_base['month'].dt.year
#yearly_revenue_baseline = df_base.groupby('year')['TotalRevenue(baseline)_€'].sum().reset_index()
#ignore the 2025 data
#yearly_revenue_baseline = yearly_revenue_baseline[yearly_revenue_baseline['year'] < 2025]
#View the few results
#print(yearly_revenue_baseline)

#df_ref['month'] = pd.to_datetime(df_ref['month'], dayfirst=True, errors='coerce')
#df_ref['year'] = df_ref['month'].dt.year
#yearly_revenue_techref = df_ref.groupby('year')['TotalRevenue(techref)_€'].sum().reset_index()
#ignore the 2025 data
#yearly_revenue_techref = yearly_revenue_techref[yearly_revenue_techref['year'] < 2025]
#View the few results
#print(yearly_revenue_techref)

#Plotting
set_style(body_pt=10, theme="latex_like")
plt.plot(monthly_revenue_baseline["month"], monthly_revenue_baseline["SpotPrice(baseline)_€/MWh"], 
         label="Baseline-Referencing(SpotPrice)", marker="o", color="darkblue")
plt.plot(monthly_revenue_techref["month"], monthly_revenue_techref["SpotPrice(techref)_€/MWh"], 
         label="Tech specific- Referencing(SpotPrice)", marker="o", color="violet")

plt.title("Spot Price Referencing Comparison")
plt.xlabel("Month")
plt.ylabel("Spot Price (€)")
plt.legend()
plt.tight_layout()
output_plot_path = "C:/Thesis_Project/thesis_data/results/Tech_ref/SpotPrice_comparison_dk1_zoom.pdf"
plt.savefig(output_plot_path, bbox_inches='tight')
print(f"PDF plot saved to: {output_plot_path}")
plt.show()