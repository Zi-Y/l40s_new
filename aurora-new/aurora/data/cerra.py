import pickle

import numpy as np
from metpy.calc import dewpoint_from_relative_humidity, specific_humidity_from_dewpoint
from metpy.units import units
from torch.utils.data import Dataset, DataLoader
from aurora.batch import Batch, Metadata
import torch
import math
import xarray as xr
import zarr

from datetime import datetime, timezone
from aurora.data.collate import collate_fn

SURFACE_VAR_MAPPING = {
    "2t" : "t2m",
    "10u" : "10u",
    "10v" : "10v",
    "msl" : "msl",
    "tp" : "tp",
    "tciwv" : "tciwv"
}

STATIC_VAR_MAPPING = {
    "lsm" : "lsm",
    "slt" : "slt",
    "z" : "sp"
 }

ATMOS_VAR_MAPPING = {
    "t" : "t",
    "u" : "u",
    "v" : "v",
    "q" : "q",
    "z" : "z",
}

class CerraDataset(Dataset):
    """
    For every batch we have current time T and state at T-1. Time dimension is therefore 2.


    """
    def __init__(self,
                 data_path: str,
                 start_time: str,
                 end_time: str,
                 surf_vars: tuple[str, ...] = ("2t", "10u", "10v", "msl"),
                 atmos_vars: tuple[str, ...] = ("z", "u", "v", "t", "q"),
                 static_vars: tuple[str, ...] = ("lsm", "z", "slt"),
                 normalize: bool = False,
                 stats_path: str = None,
                 x_range: tuple[int, int] = (0, 1068),
                 y_range: tuple[int, int] = (0, 1068),
                 rollout_steps: int = 1,
                 step_size_hours: int = 24,
                 lead_time_hours: int = 24,
                 dataset_step_size: int = 24,
                 use_dummy_slt: bool = False,
                 no_xarray: bool = False,
                 ):
        """
        Args:
            data_path (str): Path to the dataset
            start_time (str): Start time of the dataset, currently only supports 6 AM start time
            end_time (str): End time of the dataset
            normalize (bool): Whether to normalize the dataset
            stats_path (str): Path to the statistics file. Expects a pickled dictionary with variable names as keys and
                tuples of mean and std as values. If None, the dataset will not be normalized!
            x_range (tuple[int, int]): Range of x values to consider
            y_range (tuple[int, int]): Range of y values to consider
            rollout_steps (int): Number of steps to consider in the rollout
            step_size_hours (int): Step size in hours
            lead_time_hours (int): Forecast lead time in hours. Influences the difference between current and previous time.
            dataset_step_size (int): Step size of the dataset
            use_dummy_slt (bool): Whether to use a dummy slt variable. This is useful when the dataset does not contain slt.
            no_xarray (bool): Whether to use xarray or not. If True, the dataset will be loaded without xarray.
        """
        super().__init__()

        self.x_range = x_range
        self.y_range = y_range

        self.ds_step_size_hours = dataset_step_size
        self.rollout_steps = rollout_steps
        self.lead_time_steps = lead_time_hours // self.ds_step_size_hours
        self.step_size = step_size_hours // self.ds_step_size_hours
        self.surf_vars = surf_vars
        self.atmos_vars = atmos_vars
        self.static_vars = static_vars
        self.normalize = normalize
        self.use_dummy_slt = use_dummy_slt
        self.no_xarray = no_xarray

        if self.normalize:
            if stats_path is None:
                self.stats = {}
            else:
                with open(stats_path, "rb") as f:
                    self.stats = pickle.load(f)

        if self.no_xarray:
            self._init_zarr(data_path, start_time, end_time)
        else:
            self._init_xarray(data_path, start_time, end_time)

    def _init_zarr(self, data_path: str, start_time: str, end_time: str):
        self.data = zarr.open(data_path)

        # The time dim doesn't always have the same name, so we accept multiple values
        if "time" in self.data:
            self.time_dim = "time"
        elif "valid_time" in self.data:
            self.time_dim = "valid_time"
        else:
            raise ValueError("No time dimension found for dataset.")

        # Convert longitude to the range [-180, 180] regardless of whether the dataset is in [0, 360] or [-180, 180]
        longitudes_neg = ((self.data["longitude"][0] % 360) + 180) % 360 - 180

        pressure_level_descending = np.all(self.data["pressure_level"][:][:-1] > self.data["pressure_level"][:][1:])
        latitude_ascending = np.all(self.data["latitude"][:, 0][:-1] <= self.data["latitude"][:, 0][1:])
        longitude_ascending = np.all(longitudes_neg[:-1] <= longitudes_neg[1:])
        assert pressure_level_descending, "Use xarray dataset if pressure levels are not descending!"
        assert latitude_ascending, "Use xarray dataset if latitude is not ascending!"
        assert longitude_ascending, "Use xarray dataset if longitude is not ascending!"

        self.longitude = self.data["longitude"][:][::-1]
        self.longitude = self.longitude[self.x_range[0]:self.x_range[1], self.y_range[0]:self.y_range[1]].copy()
        self.longitude = self.longitude % 360
        self.latitude = self.data["latitude"][:][::-1]
        self.latitude = self.latitude[self.x_range[0]:self.x_range[1], self.y_range[0]:self.y_range[1]].copy()
        self.pressure_level = self.data["pressure_level"][:][::-1]
        self.pressure_level = self.pressure_level.copy()

        assert np.all(self.longitude.min(axis=1) >= 0), "Datavar: longitude - Some longitudes are below 0"
        assert np.all(self.longitude.min(axis=1) < 360), "Datavar: longitude - Some longitudes are >= 360"
        assert np.all(self.latitude.min(axis=1) >= 0), "Datavar: latitude - Some latitudes are below 0"
        assert np.all(self.latitude.min(axis=1) < 360), "Datavar: latitude - Some latitudes are >= 360"

        start_timestamp = int(datetime.fromisoformat(start_time).replace(tzinfo=timezone.utc).timestamp())
        end_timestamp = int(datetime.fromisoformat(end_time).replace(tzinfo=timezone.utc).timestamp())
        self.start_idx = np.argmax(start_timestamp <= self.data[self.time_dim][:]).item()
        self.end_idx = np.argmax(end_timestamp < self.data[self.time_dim][:]).item()

        # If the end index is 0, it means that the end time is not in the dataset.
        if self.end_idx == 0:
            self.end_idx = len(self.data[self.time_dim][:])

    def _init_xarray(self, data_path: str, start_time: str, end_time: str):
        self.data = xr.open_zarr(data_path)

        # The time dim doesn't always have the same name, so we accept multiple values
        if "time" in self.data:
            self.time_dim = "time"
        elif "valid_time" in self.data:
            self.time_dim = "valid_time"
        else:
            raise ValueError("No time dimension found for dataset.")

        # Ensure the correct order of dimensions
        self.data = self.data.transpose(self.time_dim, "pressure_level", "y", "x")
        self.data = self.data.sortby([-self.data.y, self.data.x, self.data.pressure_level])
        self.data["longitude"].values = self.data["longitude"].values % 360

        assert (self.data["longitude"].min(axis=1) >= 0).all(), "Datavar: longitude - Some longitudes are below 0"
        assert (self.data["longitude"].min(axis=1) < 360).all(), "Datavar: longitude - Some longitudes are >= 360"
        assert (self.data["latitude"].min(axis=1) >= 0).all(), "Datavar: latitude - Some longitudes are below 0"
        assert (self.data["latitude"].min(axis=1) < 360).all(), "Datavar: latitude - Some longitudes are >= 360"

        selection = {self.time_dim: slice(start_time, end_time), "x": slice(*self.x_range), "y": slice(*self.y_range)}
        self.data = self.data.sel(selection)

    def _get_sample_zarr(self, id_range: list[int], var: str) -> torch.Tensor:
        # Compute derived variables online if not precomputed
        if (var == "10v" or var == "10u") and var not in self.data:
            wind_speed = self.data["si10"][id_range][:, ::-1]
            wdir = self.data["si10"][id_range][:, ::-1]
            wind_speed = wind_speed[:, self.x_range[0]:self.x_range[1], self.y_range[0]:self.y_range[1]][None]
            wdir = wdir[:, self.x_range[0]:self.x_range[1], self.y_range[0]:self.y_range[1]][None]

            wdir_rad = np.radians(wdir)

            if var == "10v":
                v = -wind_speed * np.cos(wdir_rad)
                return torch.from_numpy(v.copy())
            else:
                u = -wind_speed * np.sin(wdir_rad)
                return torch.from_numpy(u.copy())

        elif var == "q" and var not in self.data:
            temperature = self.data["t"][id_range][:, ::-1, ::-1]
            relative_h = self.data["r"][id_range][:, ::-1, ::-1]
            temperature = temperature[:, :, self.x_range[0]:self.x_range[1], self.y_range[0]:self.y_range[1]][None]
            relative_h = relative_h[:, :, self.x_range[0]:self.x_range[1], self.y_range[0]:self.y_range[1]][None]
            pressure_levels = self.pressure_level
            # Shape: (pressure_level,)
            pressure = np.expand_dims(pressure_levels, axis=(0, -1, -2))  # Initial shape: (1, pressure_level, 1, 1)
            pressure = np.broadcast_to(pressure, temperature.shape)  # Final shape: (1, pressure_level, lat, lon)
            pressure = pressure * units.hPa
            temperature_k = temperature * units.kelvin
            relative_h = np.maximum(relative_h, np.finfo(np.float32).eps) * units.percent  # Avoid division by zero
            dewpoint = dewpoint_from_relative_humidity(temperature_k, relative_h * units.percent)
            specific_humidity = specific_humidity_from_dewpoint(pressure, dewpoint)

            return torch.from_numpy(specific_humidity.m.copy())

        elif var == "slt" and self.use_dummy_slt:
            # We don't have slt in the dataset, so we just fill with zeros
            shape = (self.x_range[1] - self.x_range[0], self.y_range[1] - self.y_range[0])
            return torch.zeros(shape)

        elif var == "slt" and not self.use_dummy_slt:
            # The slt variable is added manually and may not have a time dimension (which it doesn't need since it is constant)
            if len(self.data[var].shape) == 2:
                data_point = self.data[var][:][::-1]
            else:
                data_point = self.data[var][id_range[0]][::-1]
            data_point = data_point[self.x_range[0]:self.x_range[1], self.y_range[0]:self.y_range[1]]
            data_point = torch.from_numpy(data_point.copy())
            return data_point

        elif var in STATIC_VAR_MAPPING.values():
            data_point = self.data[var][id_range[0]][::-1]
            data_point = data_point[self.x_range[0]:self.x_range[1], self.y_range[0]:self.y_range[1]]
            data_point = torch.from_numpy(data_point.copy())
            return data_point

        elif var in SURFACE_VAR_MAPPING.values():
            data_point = self.data[var][id_range][:, ::-1]
            data_point = data_point[:, self.x_range[0]:self.x_range[1], self.y_range[0]:self.y_range[1]][None]
            data_point = torch.from_numpy(data_point.copy())
            return data_point

        else:
            data_point = self.data[var][id_range][:, ::-1, ::-1]
            data_point = data_point[:, :, self.x_range[0]:self.x_range[1], self.y_range[0]:self.y_range[1]][None]
            data_point = torch.from_numpy(data_point.copy())
            return data_point

    def _get_metadata_zarr(self, id_range: list[int]) -> Metadata:
        time = self.data[self.time_dim][id_range]
        time_metadata = [datetime.fromtimestamp(t, timezone.utc) for t in time]
        time_metadata = tuple(time_metadata)

        levels = tuple(self.pressure_level.astype(int))

        return Metadata(
            lat=torch.from_numpy(self.latitude),
            lon=torch.from_numpy(self.longitude),
            time=time_metadata,
            atmos_levels=levels
        )

    def _get_sample_xarray(self, selection: xr.Dataset, var: str) -> torch.Tensor:
        # Compute derived variables online if not precomputed
        if (var == "10v" or var == "10u") and var not in self.data.data_vars:
            wind_speed = selection["si10"].values[None]
            wdir = selection["wdir10"].values[None]
            wdir_rad = np.radians(wdir)

            if var == "10v":
                v = -wind_speed * np.cos(wdir_rad)
                return torch.from_numpy(v)
            else:
                u = -wind_speed * np.sin(wdir_rad)
                return torch.from_numpy(u)

        elif var == "q" and var not in self.data.data_vars:
            temperature = selection["t"].values[None]
            relative_h = selection["r"].values[None]
            pressure_levels = self.data.coords["pressure_level"].values
            # Shape: (pressure_level,)
            pressure = np.expand_dims(pressure_levels, axis=(0, -1, -2))  # Initial shape: (1, pressure_level, 1, 1)
            pressure = np.broadcast_to(pressure, temperature.shape)  # Final shape: (1, pressure_level, lat, lon)
            pressure = pressure * units.hPa
            temperature_k = temperature * units.kelvin
            relative_h = np.maximum(relative_h, np.finfo(np.float32).eps) * units.percent  # Avoid division by zero
            dewpoint = dewpoint_from_relative_humidity(temperature_k, relative_h * units.percent)
            specific_humidity = specific_humidity_from_dewpoint(pressure, dewpoint)

            return torch.from_numpy(specific_humidity.m)

        elif var == "slt" and self.use_dummy_slt:
            # We don't have slt in the dataset, so we just fill with zeros
            shape = selection["lsm"].values.shape
            return torch.zeros(shape)

        elif var in STATIC_VAR_MAPPING.values():
            data_point = selection[var].values
            data_point = torch.from_numpy(data_point)
            return data_point

        else:
            data_point = selection[var].values[None]
            data_point = torch.from_numpy(data_point)

            return data_point

    def _get_metadata_xarray(self, selection: xr.Dataset) -> Metadata:
        time = selection[self.time_dim].values
        time_metadata = [datetime.fromtimestamp(t.astype("datetime64[s]").astype("int"), timezone.utc) for t in time]
        time_metadata = tuple(time_metadata)

        levels = tuple(self.data.pressure_level.values.astype(int))

        return Metadata(
            lat=torch.from_numpy(self.data.latitude.values),
            lon=torch.from_numpy(self.data.longitude.values),
            time=time_metadata,
            atmos_levels=levels
        )

    def __len__(self) -> int:
        # Since we return (2 + self.rollout_steps) values per index,
        # we have to subtract (1 + self.rollout_steps) * self.lead_time_steps from the number of timesteps
        if self.no_xarray:
            num_steps = (self.end_idx - self.start_idx) - (1 + self.rollout_steps) * self.lead_time_steps
        else:
            num_steps = len(self.data[self.time_dim]) - (1 + self.rollout_steps) * self.lead_time_steps
        length = math.ceil(num_steps / self.step_size)

        return length

    def __getitem__(self, idx: int) -> dict[str, Batch]:
        # step size -> time delta between two steps in the dataset
        # lead time is the time between prev_idx and current idx
        # it is also the diff between t+1 and t.
        internal_idx = idx + self.start_idx if self.no_xarray else idx
        prev_idx = internal_idx * self.step_size
        current_idx = prev_idx + self.lead_time_steps
        target_idx = current_idx + self.lead_time_steps

        if self.no_xarray:
            batch = self._create_batch_zarr([current_idx, prev_idx])
        else:
            batch = self._create_batch_xarray([current_idx, prev_idx])

        output = {"input": batch}

        for i in range(0, self.rollout_steps):
            if self.no_xarray:
                output[f"target_{i}"] = self._create_batch_zarr([target_idx])
            else:
                output[f"target_{i}"] = self._create_batch_xarray([target_idx])
            target_idx += self.lead_time_steps

        return output

    def _create_batch_zarr(self, id_range: list[int]) -> Batch:

        surf_vars = {k : self._get_sample_zarr(id_range, SURFACE_VAR_MAPPING[k]) for k in self.surf_vars}
        atmos_vars = {k : self._get_sample_zarr(id_range, ATMOS_VAR_MAPPING[k]) for k in self.atmos_vars}
        static_vars = {k : self._get_sample_zarr(id_range, STATIC_VAR_MAPPING[k]) for k in self.static_vars}
        meta = self._get_metadata_zarr(id_range)

        batch = Batch(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            metadata=meta
        )

        if self.normalize:
            batch = batch.normalise(stats=self.stats)

        return batch

    def _create_batch_xarray(self, id_range: list[int]) -> Batch:

        selection = self.data.isel({self.time_dim: id_range})
        selection_static = self.data.isel({self.time_dim: id_range[0]})

        surf_vars = {k : self._get_sample_xarray(selection, SURFACE_VAR_MAPPING[k]) for k in self.surf_vars}
        atmos_vars = {k : self._get_sample_xarray(selection, ATMOS_VAR_MAPPING[k]) for k in self.atmos_vars}
        static_vars = {k : self._get_sample_xarray(selection_static, STATIC_VAR_MAPPING[k]) for k in self.static_vars}
        meta = self._get_metadata_xarray(selection)

        batch = Batch(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            metadata=meta
        )

        if self.normalize:
            batch = batch.normalise(stats=self.stats)

        return batch


if __name__ == "__main__":
    # Ensure paths and dataset details are correct
    DATA_PATH = "/mnt/cache/data/datasets/cerra_v1.zarr"  # Replace with your dataset's path
    START_TIME = "2007-01-01T06:00:00"  # Modify as needed
    END_TIME = "2007-01-10T23:59:59"  # Modify as needed

    # Step size and lead time in hours
    STEP_SIZE_HOURS = 24
    LEAD_TIME_HOURS = 24

    # Ensure CerraDataset and custom_collate_fn are correctly implemented
    dataset = CerraDataset(
        data_path=DATA_PATH,
        start_time=START_TIME,
        end_time=END_TIME,
        step_size_hours=STEP_SIZE_HOURS,
        lead_time_hours=LEAD_TIME_HOURS,
    )

    # Print dataset length to confirm loading
    print(f"Dataset length: {len(dataset)}")

    # Create DataLoader with batch size and custom collation
    dataloader = DataLoader(
        dataset, batch_size=1, collate_fn=collate_fn, shuffle=False
    )

    # Iterate through the DataLoader
    for i, batch in enumerate(dataloader):
        print(f"Batch {i + 1}")  # Modify to inspect the content of `batch`

        # Break after 5 iterations to test quickly
        if i == 5:
            break
