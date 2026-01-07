"""
Hourly actual generation with wake effects (PyWake NOJ) for a wind farm layout.

"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from py_wake.site import XRSite
from py_wake import NOJ
from py_wake.rotor_avg_models import RotorCenter
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

warnings.filterwarnings("ignore", message=".*NOJ model is not representative.*")



# HELPER FUNCTIONS (same as analysis)


TI_MIN = 0.04
TI_MAX = 0.25


def ti_from_speed(ws, ti_min=TI_MIN, ti_max=TI_MAX):
    """Calculate turbulence intensity from wind speed using empirical formula."""
    if ws <= 0:
        return ti_max
    return float(np.clip(0.03 + 0.455 / ws, ti_min, ti_max))


def k_from_ti(ti):
    """Calculate wake decay constant from turbulence intensity (tuned for offshore)."""
    return float(np.clip(0.8 * ti, 0.03, 0.08))


# LOAD TURBINE SPECIFICATIONS 

def load_turbine_from_yaml(yaml_path: Path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        turbine_yaml = yaml.safe_load(f)

    performance = turbine_yaml["performance"]
    rotor_diameter = float(turbine_yaml["rotor_diameter"])
    hub_height = float(turbine_yaml["hub_height"])
    turbine_name = turbine_yaml.get("name", "Turbine")

    pc_wind_speeds = np.array(performance["power_curve"]["power_wind_speeds"], dtype=float)
    pc_power_values = np.array(performance["power_curve"]["power_values"], dtype=float)

    ct_wind_speeds = np.array(performance["Ct_curve"]["Ct_wind_speeds"], dtype=float)
    ct_values = np.array(performance["Ct_curve"]["Ct_values"], dtype=float)

    cut_in = float(performance.get("cutin_wind_speed", performance.get("cut_in_wind_speed")))
    cut_out = float(performance.get("cutout_wind_speed", performance.get("cut_out_wind_speed")))

    if np.allclose(pc_wind_speeds, ct_wind_speeds):
        ws_grid = pc_wind_speeds
        ct_grid = ct_values
    else:
        ws_grid = pc_wind_speeds
        ct_grid = np.interp(pc_wind_speeds, ct_wind_speeds, ct_values)

    epsilon = 1e-6
    ws_extended = np.r_[[max(0.0, cut_in - epsilon)], ws_grid, [cut_out + epsilon]]
    power_extended = np.r_[[0.0], pc_power_values, [0.0]]
    ct_extended = np.r_[[0.0], ct_grid, [0.0]]

    turbine = WindTurbine(
        name=turbine_name,
        diameter=rotor_diameter,
        hub_height=hub_height,
        powerCtFunction=PowerCtTabular(
            ws=ws_extended,
            power=power_extended,
            ct=ct_extended,
            power_unit="W",
        ),
    )

    meta = {
        "name": turbine_name,
        "rotor_diameter": rotor_diameter,
        "hub_height": hub_height,
        "cut_in": cut_in,
        "cut_out": cut_out,
        "rated_power_W": float(np.max(power_extended)),
    }
    return turbine, meta


# SITE MODEL (same as analysis)

def make_site(cut_in, cut_out):
    wd_grid = np.arange(0, 360, 30.0)
    ws_grid_site = np.linspace(max(0.5, cut_in * 0.5), cut_out, 50)

    site_dataset = xr.Dataset(
        data_vars={
            "TI": (("wd", "ws"), np.full((len(wd_grid), len(ws_grid_site)), 0.08, dtype=float)),
            "Sector_frequency": (("wd",), np.full(len(wd_grid), 1 / len(wd_grid), dtype=float)),
            "Weibull_A": (("wd",), np.full(len(wd_grid), 8.0, dtype=float)),
            "Weibull_k": (("wd",), np.full(len(wd_grid), 2.0, dtype=float)),
        },
        coords={"wd": wd_grid, "ws": ws_grid_site},
    )

    site = XRSite(site_dataset)
    site.interp_method = "linear"
    return site

# LAYOUT 

def load_layout_from_python(module_name: str):
    """
    Layout module must define layout_x and layout_y.
    Example: module file code/Layoutirregular_IEA10mw.py with layout_x, layout_y variables.
    """
    mod = __import__(module_name, fromlist=["layout_x", "layout_y"])
    layout_x = np.asarray(getattr(mod, "layout_x"), dtype=float)
    layout_y = np.asarray(getattr(mod, "layout_y"), dtype=float)
    return layout_x, layout_y



# MAIN

def parse_args():
    p = argparse.ArgumentParser(description="Hourly wake-based generation (NOJ) - analysis-consistent version")
    p.add_argument("--input-csv", type=Path, required=True, help="Input CSV containing hours, Wind_Speed_m/s, Wind_Direction_deg")
    p.add_argument("--turbine-yaml", type=Path, required=True, help="Turbine YAML file path")
    p.add_argument("--output-csv", type=Path, required=True, help="Output CSV path")
    p.add_argument("--layout-module", type=str, required=True, help="Python module name defining layout_x and layout_y")

    p.add_argument("--use-k-from-ti", action="store_true", help="Use dynamic k(TI). If not set, fixed-k is used.")
    p.add_argument("--fixed-k", type=float, default=0.05, help="Fixed k value (used if --use-k-from-ti not set)")
    p.add_argument("--progress-every", type=int, default=200, help="Print progress every N rows (0 disables)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load input
    df = pd.read_csv(args.input_csv)
    df["hours"] = pd.to_datetime(df["hours"], utc=True, errors="coerce")
    df["Wind_Speed_m/s"] = pd.to_numeric(df["Wind_Speed_m/s"], errors="coerce")
    df["Wind_Direction_deg"] = pd.to_numeric(df["Wind_Direction_deg"], errors="coerce") % 360.0
    df = df.dropna(subset=["hours"]).reset_index(drop=True)

    # Load turbine + site
    turbine, meta = load_turbine_from_yaml(args.turbine_yaml)
    cut_in = meta["cut_in"]
    cut_out = meta["cut_out"]
    site = make_site(cut_in=cut_in, cut_out=cut_out)

    # Load layout (python module)
    layout_x, layout_y = load_layout_from_python(args.layout_module)
    n_turbines = len(layout_x)
    nameplate_capacity_MW = n_turbines * meta["rated_power_W"] / 1e6

    print("=" * 80)
    print("HOURLY GENERATION CALCULATION (analysis-consistent)")
    print("=" * 80)
    print(f"Rows: {len(df):,}")
    print(f"Turbines: {n_turbines} | Nameplate: {nameplate_capacity_MW:.1f} MW")
    print(f"Dynamic k(TI): {bool(args.use_k_from_ti)} | Fixed k: {args.fixed_k:.3f}")

    start_time = time.time()
    actual_generation_MWh = np.zeros(len(df), dtype=float)

    # Hourly loop 
    for i, row in df.iterrows():
        if args.progress_every and i % args.progress_every == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else np.nan
            remaining = (len(df) - i) / rate if rate and np.isfinite(rate) else np.nan
            print(f"Progress: {i:,}/{len(df):,} ({100*i/len(df):.1f}%)  ETA: {remaining/60:.1f} min")

        wind_speed = float(row["Wind_Speed_m/s"])
        wind_direction = float(row["Wind_Direction_deg"])

        if (not np.isfinite(wind_speed) or
            wind_speed < cut_in or
            wind_speed >= cut_out or
            not np.isfinite(wind_direction)):
            actual_generation_MWh[i] = 0.0
            continue

        turbulence_intensity = ti_from_speed(wind_speed)
        wake_decay_k = k_from_ti(turbulence_intensity) if args.use_k_from_ti else args.fixed_k

        # NOJ instantiated inside loop 
        noj_hourly = NOJ(site, turbine, k=wake_decay_k, rotorAvgModel=RotorCenter())
        sim_hourly = noj_hourly(
            layout_x,
            layout_y,
            wd=[wind_direction],
            ws=[wind_speed],
            TI=[turbulence_intensity],
        )

        power_W = float(sim_hourly.Power.values[:, 0, 0].sum())
        actual_generation_MWh[i] = power_W / 1e6

    elapsed_time = time.time() - start_time

    # Save results
    output_df = df.copy()
    output_df["Actual_MWh"] = actual_generation_MWh

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_csv, index=False)

    total_generation_GWh = actual_generation_MWh.sum() / 1000.0
    avg_capacity_factor = (actual_generation_MWh.sum() / (nameplate_capacity_MW * len(df))) * 100.0

    print("\nComplete.")
    print(f"Runtime: {elapsed_time/60:.1f} min | Avg per hour: {elapsed_time/len(df):.3f} s")
    print(f"Total generation: {total_generation_GWh:.2f} GWh")
    print(f"Average capacity factor: {avg_capacity_factor:.2f}%")
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
