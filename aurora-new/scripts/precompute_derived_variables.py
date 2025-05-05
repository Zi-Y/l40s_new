"""Precompute derived variables for CERRA dataset."""
import argparse
import os

import numpy as np
from metpy.calc import dewpoint_from_relative_humidity, specific_humidity_from_dewpoint
from metpy.units import units
from numcodecs import Blosc
from tqdm import tqdm

from main import check_and_start_debugger
import xarray as xr


def compute_wind_components(wind_speed: np.ndarray, wind_direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wind_direction_rad = np.radians(wind_direction)
    v = -wind_speed * np.cos(wind_direction_rad)
    u = -wind_speed * np.sin(wind_direction_rad)

    return u, v


def compute_specific_humidity(temperature: np.ndarray, relative_humidity: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    pressure = np.expand_dims(pressure, axis=(-1, -2))  # Initial shape: (pressure_level, 1, 1)
    pressure = np.broadcast_to(pressure, temperature.shape)  # Final shape: (pressure_level, lat, lon)
    pressure = pressure * units.hPa
    temperature_k = temperature * units.kelvin
    relative_h = np.maximum(relative_humidity, np.finfo(np.float32).eps) * units.percent  # Avoid division by zero
    dewpoint = dewpoint_from_relative_humidity(temperature_k, relative_h * units.percent)
    specific_humidity = specific_humidity_from_dewpoint(pressure, dewpoint)

    return specific_humidity.m

def main(args: argparse.Namespace):
    """Compute 10m u/v wind components from 10m wind speed and direction.
    In addition, compute specific humidity from relative humidity and temperature and pressure.

    Args:
        args (argparse.Namespace): Command-line arguments
    """
    chunks = {"valid_time": 1, "pressure_level": 13, "y": 1069, "x": 1069}
    data = xr.open_zarr(args.src_dir, chunks=chunks)

    if args.start_date is not None and args.end_date is not None:
        data = data.sel(valid_time=slice(args.start_date, args.end_date))

    wind_u_values = []
    wind_v_values = []
    specific_humidity_values = []

    for i in tqdm(range(len(data["valid_time"]))):
        sample = data.isel(valid_time=i)

        wind_speed = sample["si10"].values
        wind_direction = sample["wdir10"].values

        u, v = compute_wind_components(wind_speed, wind_direction)

        temperature = sample["t"].values
        relative_humidity = sample["r"].values
        pressure = sample["pressure_level"].values

        specific_humidity = compute_specific_humidity(temperature, relative_humidity, pressure)

        wind_u_values.append(u)
        wind_v_values.append(v)
        specific_humidity_values.append(specific_humidity)

        if (i + 1) % args.save_freq == 0:
            data_subset = data.isel(valid_time=slice(i - args.save_freq + 1, i + 1))
            wind_dims = ["valid_time", "y", "x"]
            q_dims = ["valid_time", "pressure_level", "y", "x"]
            wind_coords = {"valid_time": data_subset["valid_time"]}
            q_coords = {"valid_time": data_subset["valid_time"], "pressure_level": data["pressure_level"]}
            wind_chunks = {"valid_time": 1, "y": 1069, "x": 1069}
            q_chunks = {"valid_time": 1, "pressure_level": 13, "y": 1069, "x": 1069}
            data_subset = data_subset.drop_vars(["si10", "wdir10", "r"])

            data_subset["10u"] = xr.DataArray(wind_u_values, dims=wind_dims, coords=wind_coords).chunk(wind_chunks)
            data_subset["10v"] = xr.DataArray(wind_v_values, dims=wind_dims, coords=wind_coords).chunk(wind_chunks)
            data_subset["q"] = xr.DataArray(specific_humidity_values, dims=q_dims, coords=q_coords).chunk(q_chunks)

            if not os.path.exists(args.dst_dir):
                compressor = Blosc(cname='lz4hc', clevel=9, shuffle=Blosc.SHUFFLE)
                encoding = {var: {"compressor": compressor} for var in data_subset.data_vars}

                data_subset.to_zarr(args.dst_dir, mode="w", encoding=encoding)
            else:
                data_subset.to_zarr(args.dst_dir, append_dim="valid_time")

            wind_u_values = []
            wind_v_values = []
            specific_humidity_values = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src_dir",
        type=str,
        help="Location of the CERRA dataset",
    )

    parser.add_argument(
        "--dst_dir",
        type=str,
        help="Location of the CERRA dataset",
    )

    parser.add_argument(
        "--save_freq",
        type=int,
        default=100,
        help="Frequency to save the precomputed variables. Lower values will consume less RAM",
    )

    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Start datetime for the precomputation",
    )

    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="End datetime for the precomputation",
    )

    args = parser.parse_args()

    check_and_start_debugger()
    main(args)
