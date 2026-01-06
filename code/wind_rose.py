import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import windrose 
from windrose import WindroseAxes

FILES = [
    "C:/Thesis_Project/thesis_data/data/10_metre/ERA5_2019dk_10m.xlsx",
    "C:/Thesis_Project/thesis_data/data/10_metre/ERA5_2020dk_10m.xlsx",
    "C:/Thesis_Project/thesis_data/data/10_metre/ERA5_2021dk_10m.xlsx",
    "C:/Thesis_Project/thesis_data/data/10_metre/ERA5_2022dk_10m.xlsx",
    "C:/Thesis_Project/thesis_data/data/10_metre/ERA5_2023dk_10m.xlsx",
    "C:/Thesis_Project/thesis_data/data/10_metre/ERA5_2024dk_10m.xlsx",
    "C:/Thesis_Project/thesis_data/data/10_metre/ERA5_2025dk_10m.xlsx"
]

output_csv = "C:/Thesis_Project/thesis_data/results/Wind_direction_frequenciesdk.csv"
output_plot = "C:/Thesis_Project/thesis_data/results/Wind_direction_frequenciesdk.png"
sector_width_deg = 30

def met_direction_from_uv(u, v):
    """Meteorological direction (FROM), degrees in [0,360)."""
    return (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0

def load_frames(paths):
    if not paths:
        raise ValueError("No file paths provided.")
    frames = []
    for path in paths:
        print(f"Reading {path} ...")
        df = pd.read_excel(path)
        if not {"u10", "v10"}.issubset(df.columns):
            raise ValueError(f"{path} must contain 'u10' and 'v10'")
        u = df["u10"].astype(float).to_numpy()
        v = df["v10"].astype(float).to_numpy()
        ws = np.sqrt(u*u + v*v)
        wd = met_direction_from_uv(u, v)
        out = {"wind_speed": ws, "wind_direction": wd}
        if "time" in df.columns:
            out["time"] = pd.to_datetime(df["time"], errors="coerce")
        frames.append(pd.DataFrame(out))
    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.dropna(subset=["wind_speed", "wind_direction"]).reset_index(drop=True)
    return df_all

def compute_sector_frequencies(df, sector_width_deg=30):
    edges = np.arange(0, 360 + sector_width_deg, sector_width_deg)  # 0,30,...,360
    centers = (edges[:-1] + edges[1:]) / 2.0                        # 15,45,...,345
    sector_idx = pd.cut(
        df["wind_direction"], bins=edges, right=False, include_lowest=True, labels=False
    )
    freq_frac = sector_idx.value_counts(normalize=True).sort_index()
    freq_frac = freq_frac.reindex(range(len(centers)), fill_value=0.0)
    out = pd.DataFrame({"direction": centers.astype(float),
                        "proportion": (freq_frac.values * 100.0)})
    # Normalize exactly to 100%
    s = out["proportion"].sum()
    if s > 0:
        out["proportion"] *= (100.0 / s)
    return out

def plot_wind_rose(df_all, out_path):
    """Plot a wind rose diagram from the DataFrame."""
    ax = WindroseAxes.from_ax()
    ax.bar(df_all["wind_direction"].values,
           df_all["wind_speed"].values,
            normed=True, opening=0.8, edgecolor='black',
            bins=np.arange(0, 26, 5))
    ax.set_legend(title="Wind Speed (m/s)", loc="upper right")
    ax.set_title("Combined Wind Rose  (2019 to 2025)", pad=12)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved wind rose plot to: {out_path}")
    plt.show()

def main():
    df_all = load_frames(FILES)

    # Quick sanity
    print(f"Total records: {len(df_all):,}")
    print(f"Mean wind speed: {df_all['wind_speed'].mean():.2f} m/s")
    print(f"Calms (<0.5 m/s): {(df_all['wind_speed'] < 0.5).mean()*100:.2f}%")

    # Frequencies
    windrose_df = compute_sector_frequencies(df_all, sector_width_deg=sector_width_deg)
    assert np.isclose(windrose_df["proportion"].sum(), 100.0, atol=1e-6)

    # Save CSV schema expected by your wake model: direction (deg), proportion (%)
    windrose_df.to_csv(output_csv, index=False)
    print(f"Saved frequencies to: {output_csv}")

    # Plot
    plot_wind_rose(df_all, output_plot)

if __name__ == "__main__":
    main()