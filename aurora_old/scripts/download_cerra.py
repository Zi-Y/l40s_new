import argparse
import os
from typing import Any

import cdsapi
import xarray as xr
import yaml
from numcodecs import Blosc


def download_month(config: dict[str, Any], year: str, month: str, outdir: str):
    """Download a month of CERRA data and save it to a zarr file.

    Args:
        config (dict): Template for the CDS API requests
        year (str): Year to download
        month (str): Month to download
        outdir (str): Directory to save the zarr file to
    """

    dataset_atmosphere = "reanalysis-cerra-pressure-levels"
    tempfile_atmosphere = f"./cerra_atmosphere_{year}_{month}.nc"
    request_atmosphere = config["cerra_atmosphere"].copy()
    request_atmosphere["year"] = [year]
    request_atmosphere["month"] = [month]

    # For the surface vars, we need two separate requests, because the 10m vars
    # (e.g. 10m_wind_direction) don't play nice with the 2m vars (e.g. 2m_temperature)
    dataset_surface = "reanalysis-cerra-single-levels"
    tempfile_surface_10m = f"./cerra_surface_10m_{year}_{month}.nc"
    request_surface_10m = config["cerra_surface_10m"].copy()
    request_surface_10m["year"] = [year]
    request_surface_10m["month"] = [month]

    tempfile_surface_2m = f"./cerra_surface_2m_{year}_{month}.nc"
    request_surface_2m = config["cerra_surface_2m"].copy()
    request_surface_2m["year"] = [year]
    request_surface_2m["month"] = [month]

    dataset_land = "reanalysis-cerra-land"
    tempfile_land = f"./cerra_land_{year}_{month}.nc"
    request_land = config["cerra_land"].copy()
    request_land["year"] = [year]
    request_land["month"] = [month]

    # This only works if you have the CDS API key set up
    client = cdsapi.Client()
    client.retrieve(dataset_atmosphere, request_atmosphere, target=tempfile_atmosphere)
    client.retrieve(dataset_surface, request_surface_10m, target=tempfile_surface_10m)
    client.retrieve(dataset_surface, request_surface_2m, target=tempfile_surface_2m)
    client.retrieve(dataset_land, request_land, target=tempfile_land)

    # Align and merge the datasets
    num_pressure_levels = len(request_atmosphere["pressure_level"])
    chunks_atmosphere = {"valid_time": 1, "y": 1069, "x": 1069, "pressure_level": num_pressure_levels}
    chunks_surface = {"valid_time": 1, "y": 1069, "x": 1069}
    chunks_land = {"time": 1, "y": 1069, "x": 1069}

    ds_atmosphere = xr.open_dataset(tempfile_atmosphere, chunks=chunks_atmosphere)
    ds_surface_10m = xr.open_dataset(tempfile_surface_10m, chunks=chunks_surface)
    ds_surface_2m = xr.open_dataset(tempfile_surface_2m, chunks=chunks_surface)
    ds_land = xr.open_dataset(tempfile_land, chunks=chunks_land)
    ds_atmosphere = ds_atmosphere.reset_coords(["latitude", "longitude", "expver"])
    ds_surface_10m = ds_surface_10m.reset_coords(["latitude", "longitude", "expver"])
    ds_surface_2m = ds_surface_2m.reset_coords(["latitude", "longitude", "expver"])
    ds_land = ds_land.reset_coords(["latitude", "longitude", "step", "surface", "valid_time"])
    ds_land = ds_land.drop_vars(["step", "surface", "valid_time"])
    ds_land = ds_land.rename({"time": "valid_time"})

    ds_merged = xr.merge([ds_atmosphere, ds_surface_10m, ds_surface_2m, ds_land])
    ds_merged = ds_merged.drop_vars(["expver"])

    if os.path.exists(outdir):
        ds_merged.to_zarr(outdir, append_dim="valid_time", mode="a")
    else:
        # We use lz4hc because it combines good compression with fast decompression
        # It's relatively slow to compress, but we only need to do that once
        compressor = Blosc(cname='lz4hc', clevel=9, shuffle=Blosc.SHUFFLE)
        encoding = {var: {"compressor": compressor} for var in ds_merged.data_vars}

        ds_merged.to_zarr(outdir, mode="w-", encoding=encoding)

    # Clean up
    os.remove(tempfile_atmosphere)
    os.remove(tempfile_surface_10m)
    os.remove(tempfile_surface_2m)
    os.remove(tempfile_land)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--request_template",
        type=str,
        help="Template file for the request",
    )

    parser.add_argument(
        "--start_year",
        type=int,
        default=2020,
        help="Year to start downloading"
    )

    parser.add_argument(
        "--start_month",
        type=int,
        default=1,
        help="Month in start_year to start downloading"
    )

    parser.add_argument(
        "--end_year",
        type=int,
        default=2020,
        help="Year to stop downloading"
    )

    parser.add_argument(
        "--end_month",
        type=int,
        default=12,
        help="Month in end_year to stop downloading"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        help="dir to write results to",
    )

    args = parser.parse_args()

    with open(args.request_template, "r") as f:
        request_config = yaml.load(f, Loader=yaml.FullLoader)

    current_year = args.start_year
    current_month = args.start_month
    end_year = args.end_year
    end_month = args.end_month

    while current_year < end_year or (current_year == end_year and current_month <= end_month):
        download_month(
            config=request_config,
            year=f"{current_year}",
            month=f"{current_month:02}",
            outdir=args.outdir
        )

        print(f"Downloaded CERRA data for {current_year}-{current_month:02}")

        if current_month == 12:
            current_month = 1
            current_year += 1
        else:
            current_month += 1
