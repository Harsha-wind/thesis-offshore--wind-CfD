import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from py_wake.site import XRSite, UniformSite
from py_wake import NOJ
from py_wake.rotor_avg_models import RotorCenter
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
import yaml
import warnings
import time

warnings.filterwarnings("ignore", message=".*NOJ model is not representative.*")

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths
INPUT_CSV = r"E:/Thesis_Project/thesis_data/results/NL/EP/High/merged_wind_price_full_high_NL.csv"
YAML_PATH = r"E:/Thesis_Project/thesis_data/code/IEA37_10MW_turbine.yaml"
OUTPUT_CSV = r"E:/Thesis_Project/thesis_data/results/NL/Current/high/hourly_actual_generation_withwake_NL_(High_NL).csv"

# Output plot paths
PLOT_DIR = "E:/Thesis_Project/thesis_data/results/NL/Wind_Direction/"
QUICK_WAKE_PLOT = PLOT_DIR + "quick_wake_test_wd270(NL)_high.png"
FINGERPRINT_PLOT = PLOT_DIR + "wake_fingerprint_(NL)_high.png"
WAKE_VS_WS_PLOT = PLOT_DIR + "wake_loss_vs_ws(NL)_high.png"
ACTUAL_FINGERPRINT_PLOT = PLOT_DIR + "wake_fingerprint_actual_(NL)_high.png"

# Wake model parameters
USE_K_FROM_TI = True  # Set False to use fixed k
FIXED_K = 0.05

# TI bounds
TI_MIN = 0.04
TI_MAX = 0.25

# Validation settings
ENABLE_VALIDATION_PLOTS = True
TEST_WIND_DIRECTION = 270.0  # Wind from West
TEST_WIND_SPEED = 10.0
TEST_TURBULENCE = 0.06


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ti_from_speed(ws, ti_min=TI_MIN, ti_max=TI_MAX):
    """Calculate turbulence intensity from wind speed using empirical formula."""
    if ws <= 0:
        return ti_max
    return float(np.clip(0.03 + 0.455 / ws, ti_min, ti_max))


def k_from_ti(ti):
    """Calculate wake decay constant from turbulence intensity (tuned for offshore)."""
    return float(np.clip(0.8 * ti, 0.03, 0.08))


def rotate_about_centroid(x, y, deg):
    """Rotate coordinates about their centroid by specified degrees."""
    theta = np.deg2rad(deg)
    cx, cy = x.mean(), y.mean()
    X, Y = x - cx, y - cy
    x_rot = X * np.cos(theta) - Y * np.sin(theta) + cx
    y_rot = X * np.sin(theta) + Y * np.cos(theta) + cy
    return x_rot, y_rot


def single_turbine_power_W(turbine, ws, TI):
    """Calculate power of a single turbine without wake effects."""
    site_no_wake = XRSite(xr.Dataset(
        data_vars={
            "TI": (("wd", "ws"), np.array([[TI]])),
            "Sector_frequency": (("wd",), np.array([1.0])),
            "Weibull_A": (("wd",), np.array([8.0])),
            "Weibull_k": (("wd",), np.array([2.0]))
        },
        coords={"wd": np.array([0.0]), "ws": np.array([ws])}
    ))
    sim = NOJ(site_no_wake, turbine, k=0.05, rotorAvgModel=RotorCenter())(
        [0.0], [0.0], wd=[0.0], ws=[ws], TI=[TI]
    )
    return float(sim.Power.values[:, 0, 0].sum())


# ============================================================================
# LOAD AND PREPARE INPUT DATA
# ============================================================================

print("=" * 80)
print("LOADING INPUT DATA")
print("=" * 80)

df = pd.read_csv(INPUT_CSV)
df['hours'] = pd.to_datetime(df['hours'], utc=True, errors='coerce')
df['Wind_Speed_m/s'] = pd.to_numeric(df['Wind_Speed_m/s'], errors='coerce')
df['Wind_Direction_deg'] = pd.to_numeric(df['Wind_Direction_deg'], errors='coerce') % 360.0
df = df.dropna(subset=['hours']).reset_index(drop=True)

# Data quality summary
n_rows = len(df)
n_nan_ws = df['Wind_Speed_m/s'].isna().sum()
n_nan_wd = df['Wind_Direction_deg'].isna().sum()

print(f"Total rows: {n_rows:,}")
print(f"Time span: {df['hours'].min()} → {df['hours'].max()}")
print(f"Missing data - Wind Speed: {n_nan_ws:,}, Wind Direction: {n_nan_wd:,}")


# ============================================================================
# LOAD TURBINE SPECIFICATIONS
# ============================================================================

print("\n" + "=" * 80)
print("LOADING TURBINE SPECIFICATIONS")
print("=" * 80)

with open(YAML_PATH, "r") as f:
    turbine_yaml = yaml.safe_load(f)

# Extract turbine parameters
performance = turbine_yaml['performance']
rotor_diameter = float(turbine_yaml['rotor_diameter'])
hub_height = float(turbine_yaml['hub_height'])
turbine_name = turbine_yaml.get('name', 'Turbine')

# Power curve
pc_wind_speeds = np.array(performance['power_curve']['power_wind_speeds'], dtype=float)
pc_power_values = np.array(performance['power_curve']['power_values'], dtype=float)

# Thrust coefficient curve
ct_wind_speeds = np.array(performance['Ct_curve']['Ct_wind_speeds'], dtype=float)
ct_values = np.array(performance['Ct_curve']['Ct_values'], dtype=float)

# Operating limits
cut_in = float(performance.get('cutin_wind_speed', performance.get('cut_in_wind_speed')))
cut_out = float(performance.get('cutout_wind_speed', performance.get('cut_out_wind_speed')))

# Unify wind speed grids for power and Ct curves
if np.allclose(pc_wind_speeds, ct_wind_speeds):
    ws_grid = pc_wind_speeds
    ct_grid = ct_values
else:
    ws_grid = pc_wind_speeds
    ct_grid = np.interp(pc_wind_speeds, ct_wind_speeds, ct_values)

# Extend curves to include cut-in/cut-out with zero power
epsilon = 1e-6
ws_extended = np.r_[[max(0.0, cut_in - epsilon)], ws_grid, [cut_out + epsilon]]
power_extended = np.r_[[0.0], pc_power_values, [0.0]]
ct_extended = np.r_[[0.0], ct_grid, [0.0]]

# Create WindTurbine object
Turbine = WindTurbine(
    name=turbine_name,
    diameter=rotor_diameter,
    hub_height=hub_height,
    powerCtFunction=PowerCtTabular(
        ws=ws_extended,
        power=power_extended,
        ct=ct_extended,
        power_unit="W"
    )
)

print(f"Turbine: {turbine_name}")
print(f"Rotor diameter: {rotor_diameter} m")
print(f"Hub height: {hub_height} m")
print(f"Cut-in: {cut_in} m/s, Cut-out: {cut_out} m/s")
print(f"Rated power: {np.max(power_extended) / 1e6:.1f} MW")


# ============================================================================
# LOAD WIND FARM LAYOUT
# ============================================================================

print("\n" + "=" * 80)
print("LOADING WIND FARM LAYOUT")
print("=" * 80)

from Layoutirregular_IEA10mw import layout_x, layout_y

layout_x_array = np.asarray(layout_x, float)
layout_y_array = np.asarray(layout_y, float)

# Optional: Apply rotation if needed (currently disabled)
# layout_x_rot, layout_y_rot = rotate_about_centroid(layout_x_array, layout_y_array, -121.29)
layout_x_rot = layout_x_array
layout_y_rot = layout_y_array

n_turbines = len(layout_x_rot)
nameplate_capacity_MW = n_turbines * np.max(power_extended) / 1e6

print(f"Number of turbines: {n_turbines}")
print(f"Farm nameplate capacity: {nameplate_capacity_MW:.1f} MW")


# ============================================================================
# CREATE SITE MODEL
# ============================================================================

print("\n" + "=" * 80)
print("CREATING SITE MODEL")
print("=" * 80)

# Define wind direction and speed grids for site
wd_grid = np.arange(0, 360, 30.0)
ws_grid_site = np.linspace(max(0.5, cut_in * 0.5), cut_out, 50)

# Create xarray dataset for site
site_dataset = xr.Dataset(
    data_vars={
        "TI": (("wd", "ws"), np.full((len(wd_grid), len(ws_grid_site)), 0.08, dtype=float)),
        "Sector_frequency": (("wd",), np.full(len(wd_grid), 1 / len(wd_grid), dtype=float)),
        "Weibull_A": (("wd",), np.full(len(wd_grid), 8.0, dtype=float)),
        "Weibull_k": (("wd",), np.full(len(wd_grid), 2.0, dtype=float)),
    },
    coords={"wd": wd_grid, "ws": ws_grid_site}
)

site = XRSite(site_dataset)
site.interp_method = "linear"

print("Site model created with linear interpolation")


# ============================================================================
# VALIDATION: POWER CURVE CHECK
# ============================================================================

print("\n" + "=" * 80)
print("VALIDATION: POWER CURVE CHECK")
print("=" * 80)

# Verify single turbine power matches turbine specifications
wind_speeds_to_check = [3, 5, 8, 11, 15, 20, 25]
print("\nWind Speed (m/s) | PyWake Power (MW) | Curve Power (MW) | Match")
print("-" * 70)

for ws_check in wind_speeds_to_check:
    TI_check = ti_from_speed(ws_check)
    power_pywake = single_turbine_power_W(Turbine, ws_check, TI_check) / 1e6
    power_curve = float(np.interp(ws_check, ws_extended, power_extended)) / 1e6
    
    match = "✓" if np.isclose(power_pywake, power_curve, rtol=1e-3, atol=1e-6) else "✗"
    print(f"{ws_check:16.1f} | {power_pywake:18.2f} | {power_curve:16.2f} | {match:^5}")
    
    assert np.isclose(power_pywake, power_curve, rtol=1e-3, atol=1.0), \
        f"Power mismatch at {ws_check} m/s: {power_pywake:.2f} vs {power_curve:.2f}"

print("\n✓ Power curve validation passed!")


# ============================================================================
# VALIDATION PLOTS (OPTIONAL)
# ============================================================================

if ENABLE_VALIDATION_PLOTS:
    
    print("\n" + "=" * 80)
    print("VALIDATION: QUICK WAKE TEST")
    print("=" * 80)
    
    # Test wake propagation at 270° (wind from West)
    site_test = UniformSite(p_wd=[1.0], ti=TEST_TURBULENCE)
    noj_test = NOJ(site_test, Turbine, rotorAvgModel=RotorCenter())
    sim_test = noj_test(
        layout_x_rot, layout_y_rot,
        wd=[TEST_WIND_DIRECTION],
        ws=[TEST_WIND_SPEED]
    )
    
    # Generate flow map
    flow_map = sim_test.flow_map(wd=TEST_WIND_DIRECTION, ws=TEST_WIND_SPEED)
    
    plt.figure(figsize=(8, 4))
    flow_map.plot_wake_map()
    plt.scatter(layout_x_rot, layout_y_rot, c='r', s=15, label='Turbines')
    plt.title(f"Quick wake test: wd={TEST_WIND_DIRECTION:.0f}° (from West), ws={TEST_WIND_SPEED} m/s")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(QUICK_WAKE_PLOT, dpi=150)
    plt.close()
    print(f"✓ Quick wake test plot saved to: {QUICK_WAKE_PLOT}")
    
    
    print("\n" + "=" * 80)
    print("VALIDATION: WAKE FINGERPRINT (ALL DIRECTIONS)")
    print("=" * 80)
    
    # Calculate wake efficiency across all wind directions at 10 m/s
    ws_fingerprint = 10.0
    TI_fingerprint = ti_from_speed(ws_fingerprint)
    wd_fingerprint = np.arange(0, 360, 5.0)
    
    site_fingerprint = XRSite(xr.Dataset(
        data_vars={
            "TI": (("wd", "ws"), np.full((len(wd_fingerprint), 1), TI_fingerprint)),
            "Sector_frequency": (("wd",), np.full(len(wd_fingerprint), 1 / len(wd_fingerprint))),
            "Weibull_A": (("wd",), np.full(len(wd_fingerprint), 8.0)),
            "Weibull_k": (("wd",), np.full(len(wd_fingerprint), 2.0))
        },
        coords={"wd": wd_fingerprint, "ws": np.array([ws_fingerprint])}
    ))
    
    noj_fingerprint = NOJ(
        site_fingerprint, Turbine,
        k=k_from_ti(TI_fingerprint),
        rotorAvgModel=RotorCenter()
    )
    sim_fingerprint = noj_fingerprint(
        layout_x_rot, layout_y_rot,
        wd=wd_fingerprint,
        ws=[ws_fingerprint]
    )
    
    # Calculate efficiency
    power_single = single_turbine_power_W(Turbine, ws_fingerprint, TI_fingerprint)
    efficiency_fingerprint = sim_fingerprint.Power.values[:, :, 0].sum(axis=0) / (n_turbines * power_single)
    
    plt.figure(figsize=(9, 4))
    plt.plot(wd_fingerprint, efficiency_fingerprint)
    plt.axhline(1.0, ls='--', lw=0.8, color='gray', label='No wake loss')
    plt.title(f"Wake fingerprint at {ws_fingerprint:.0f} m/s (NL)")
    plt.xlabel("Wind direction (°)")
    plt.ylabel("Farm efficiency")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FINGERPRINT_PLOT, dpi=150)
    plt.close()
    print(f"✓ Wake fingerprint plot saved to: {FINGERPRINT_PLOT}")
    
    
    print("\n" + "=" * 80)
    print("VALIDATION: WAKE LOSS VS WIND SPEED")
    print("=" * 80)
    
    # Calculate efficiency at aligned direction across wind speeds
    wd_aligned = 120.0
    ws_range = np.arange(max(4.0, cut_in), min(cut_out, 16.0), 1.0)
    efficiency_vs_ws = []
    
    for ws_test in ws_range:
        TI_test = ti_from_speed(ws_test)
        site_test_ws = XRSite(xr.Dataset(
            data_vars={
                "TI": (("wd", "ws"), np.array([[TI_test]])),
                "Sector_frequency": (("wd",), np.array([1.0])),
                "Weibull_A": (("wd",), np.array([max(8.0, ws_test)])),
                "Weibull_k": (("wd",), np.array([2.0]))
            },
            coords={"wd": np.array([wd_aligned]), "ws": np.array([ws_test])}
        ))
        
        noj_test_ws = NOJ(
            site_test_ws, Turbine,
            k=k_from_ti(TI_test),
            rotorAvgModel=RotorCenter()
        )
        sim_test_ws = noj_test_ws(
            layout_x_rot, layout_y_rot,
            wd=[wd_aligned],
            ws=[ws_test],
            TI=[TI_test]
        )
        
        power_farm = sim_test_ws.Power.values[:, 0, 0].sum()
        power_single_no_wake = single_turbine_power_W(Turbine, ws_test, TI_test)
        
        if power_single_no_wake > 0:
            efficiency_vs_ws.append(power_farm / (n_turbines * power_single_no_wake))
        else:
            efficiency_vs_ws.append(np.nan)
    
    plt.figure(figsize=(8, 4))
    plt.plot(ws_range, np.array(efficiency_vs_ws), marker='o', lw=1.2)
    plt.axhline(1.0, ls='--', lw=0.8, color='gray', label='No wake loss')
    plt.title(f'Wake loss vs wind speed @ wd={wd_aligned:.0f}° (NL)')
    plt.xlabel("Wind speed (m/s)")
    plt.ylabel("Farm efficiency")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(WAKE_VS_WS_PLOT, dpi=150)
    plt.close()
    print(f"✓ Wake loss vs wind speed plot saved to: {WAKE_VS_WS_PLOT}")


# ============================================================================
# HOURLY GENERATION CALCULATION
# ============================================================================

print("\n" + "=" * 80)
print("HOURLY GENERATION CALCULATION")
print("=" * 80)

start_time = time.time()
actual_generation_MWh = np.zeros(len(df), dtype=float)

print(f"Processing {len(df):,} hourly records...")
print(f"Using dynamic k from TI: {USE_K_FROM_TI}")
if not USE_K_FROM_TI:
    print(f"Fixed k value: {FIXED_K}")

# Process each hour
for i, row in df.iterrows():
    # Progress updates
    if i % 200 == 0 and i > 0:
        elapsed = time.time() - start_time
        rate = i / elapsed
        remaining = (len(df) - i) / rate
        print(f"  Progress: {i:,}/{len(df):,} ({100*i/len(df):.1f}%) - "
              f"Est. time remaining: {remaining/60:.1f} min")
    
    # Extract wind conditions
    wind_speed = float(row['Wind_Speed_m/s'])
    wind_direction = float(row['Wind_Direction_deg'])
    
    # Check if conditions are valid and within operating range
    if (not np.isfinite(wind_speed) or 
        wind_speed < cut_in or 
        wind_speed >= cut_out or 
        not np.isfinite(wind_direction)):
        actual_generation_MWh[i] = 0.0
        continue
    
    # Calculate turbulence intensity and wake decay constant
    turbulence_intensity = ti_from_speed(wind_speed)
    wake_decay_k = k_from_ti(turbulence_intensity) if USE_K_FROM_TI else FIXED_K
    
    # Run wake simulation
    noj_hourly = NOJ(site, Turbine, k=wake_decay_k, rotorAvgModel=RotorCenter())
    sim_hourly = noj_hourly(
        layout_x_rot, layout_y_rot,
        wd=[wind_direction],
        ws=[wind_speed],
        TI=[turbulence_intensity]
    )
    
    # Extract total farm power and convert to MWh
    power_W = float(sim_hourly.Power.values[:, 0, 0].sum())
    actual_generation_MWh[i] = power_W / 1e6

elapsed_time = time.time() - start_time
print(f"\n✓ Calculation complete in {elapsed_time/60:.1f} minutes")
print(f"  Average: {elapsed_time/len(df):.3f} seconds per hour")


# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

output_df = df.copy()
output_df['Actual_MWh'] = actual_generation_MWh

output_df.to_csv(OUTPUT_CSV, index=False)

total_generation_GWh = actual_generation_MWh.sum() / 1000
avg_capacity_factor = (actual_generation_MWh.sum() / (nameplate_capacity_MW * len(df))) * 100

print(f"✓ Results saved to: {OUTPUT_CSV}")
print(f"\nGeneration Summary:")
print(f"  Total generation: {total_generation_GWh:.2f} GWh")
print(f"  Average capacity factor: {avg_capacity_factor:.2f}%")


# ============================================================================
# VALIDATION: ACTUAL DATA WAKE FINGERPRINT
# ============================================================================

print("\n" + "=" * 80)
print("VALIDATION: ACTUAL DATA WAKE FINGERPRINT")
print("=" * 80)

# Filter data to 9-11 m/s range for comparison with modeled fingerprint
df_filtered = output_df[
    (output_df['Wind_Speed_m/s'] >= 9) &
    (output_df['Wind_Speed_m/s'] <= 11) &
    (output_df['Actual_MWh'] > 0)
].copy()

if not df_filtered.empty:
    # Calculate no-wake reference power
    def reference_power_MWh(ws):
        TI = ti_from_speed(ws)
        return single_turbine_power_W(Turbine, ws, TI) / 1e6 * n_turbines
    
    df_filtered['Reference_MWh'] = df_filtered['Wind_Speed_m/s'].apply(reference_power_MWh)
    df_filtered['efficiency'] = df_filtered['Actual_MWh'] / df_filtered['Reference_MWh']
    
    # Group by wind direction (5° bins)
    df_filtered['wd_bin'] = (df_filtered['Wind_Direction_deg'] / 5).round() * 5
    efficiency_by_direction = df_filtered.groupby('wd_bin')['efficiency'].mean()
    
    print(f"\nActual data points analyzed: {len(df_filtered):,}")
    print(f"Wind directions covered: {len(efficiency_by_direction)}")
    print(f"Mean efficiency: {efficiency_by_direction.mean():.3f}")
    print(f"Min efficiency: {efficiency_by_direction.min():.3f} at {efficiency_by_direction.idxmin():.0f}°")
    print(f"Max efficiency: {efficiency_by_direction.max():.3f} at {efficiency_by_direction.idxmax():.0f}°")
    
    # Plot actual wake fingerprint
    plt.figure(figsize=(9, 4))
    plt.plot(efficiency_by_direction.index, efficiency_by_direction.values, marker='o')
    plt.axhline(1.0, ls='--', lw=0.8, color='gray', label='No wake loss')
    plt.title("Wake efficiency vs wind direction (9–11 m/s, actual data)")
    plt.xlabel("Wind direction (°)")
    plt.ylabel("Farm efficiency (with_wake / no_wake)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ACTUAL_FINGERPRINT_PLOT, dpi=150)
    plt.close()
    print(f"✓ Actual wake fingerprint plot saved to: {ACTUAL_FINGERPRINT_PLOT}")
else:
    print("⚠ Insufficient data in 9-11 m/s range for validation")

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)