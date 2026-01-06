import xarray as xr
import numpy as np
import pandas as pd

files = {
    2021: ("C:/Thesis_Project/thesis_data/data/NL_wind/2021_NL_windspeed.nc",
           "C:/Thesis_Project/thesis_data/data/NL_wind/NL(100m)/ERA5_2021_NL_100m.csv"),
    2022: ("C:/Thesis_Project/thesis_data/data/NL_wind/2022_NL_windspeed.nc",
           "C:/Thesis_Project/thesis_data/data/NL_wind/NL(100m)/ERA5_2022_NL_100m.csv"),
    2023: ("C:/Thesis_Project/thesis_data/data/NL_wind/2023_NL_windspeed.nc",
           "C:/Thesis_Project/thesis_data/data/NL_wind/NL(100m)/ERA5_2023_NL_100m.csv"),
    2024: ("C:/Thesis_Project/thesis_data/data/NL_wind/2024_NL_windspeed.nc",
           "C:/Thesis_Project/thesis_data/data/NL_wind/NL(100m)/ERA5_2024_NL_100m.csv"),
    2025: ("C:/Thesis_Project/thesis_data/data/NL_wind/2025_NL_windspeed.nc",
           "C:/Thesis_Project/thesis_data/data/NL_wind/NL(100m)/ERA5_2025_NL_100m.csv"),
}

for year, (inp, outp) in files.items():
    ds = xr.open_dataset(inp)

    # take spatial mean across the domain -> one value per timestamp
    u100_ts = ds["u100"].mean(dim=("latitude", "longitude"))
    v100_ts = ds["v100"].mean(dim=("latitude", "longitude"))

    # merge into dataframe
    df = xr.merge([u100_ts, v100_ts]).to_dataframe().reset_index()

    # clean/rename
    if "valid_time" in df.columns:
        df.rename(columns={"valid_time": "time"}, inplace=True)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # add wind speed magnitude at 100 m
    df["wind_speed_100m"] = np.sqrt(df["u100"]**2 + df["v100"]**2)

    # reorder columns like your screenshot
    df = df[["time", "u100", "v100", "wind_speed_100m"]]

    df.to_csv(outp, index=False)
    print(f"{year}: saved -> {outp}")

