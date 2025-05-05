import pickle
from typing import Optional

from torch.utils.data import Dataset, DataLoader
import xarray as xr
from aurora_old.batch import Batch, Metadata
import torch
import math

from datetime import datetime, timezone
from aurora_old.data.collate import collate_fn

SURFACE_VARS = {
    "2t" : "t2m",
    "10u" : "10u",
    "10v" : "10v",
    "msl" : "msl",
    "tp" : "tp"
}

STATIC_VARS = {
    "lsm" : "lsm",
    "slt" : "slt",  # As of now, we don't have slt in the dataset, so we just fill with zeros
    "z" : "sp"
 }

ATMOS_VARS = {
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
                 normalize: bool = False,
                 stats_path: str = None,
                 x_range: tuple[int, int] = (0, 1068),
                 y_range: tuple[int, int] = (0, 1068),
                 step_size_hours: int = 24,
                 lead_time_hours: int = 24,
                 dataset_step_size: int = 24
                 ):
        """
        Args:
            data_path (str): Path to the dataset
            start_time (str): Start time of the dataset, currently only supports 6 AM start time
            end_time (str): End time of the dataset
            normalize (bool): Whether to normalise the dataset
            stats_path (str): Path to the statistics file. Expects a pickled dictionary with variable names as keys and
                tuples of mean and std as values. If None, the dataset will not be normalised!
            x_range (tuple[int, int]): Range of x values to consider
            y_range (tuple[int, int]): Range of y values to consider
            step_size_hours (int): Step size in hours
            lead_time_hours (int): Forecast lead time in hours. Influences the difference between current and previous time.
            dataset_step_size (int): Step size of the dataset
        """
        super().__init__()

        time_obj = datetime.fromisoformat(start_time)

        # Assert that starting time is 06 AM
        assert time_obj.hour == 6, f"Expected start time to be 6 AM, but got {time_obj.strftime('%I:%M %p')}"
        assert time_obj.minute == 0, "Minutes should be 00"

        data = xr.open_zarr(data_path)
        data = data.transpose("valid_time", "pressure_level", "y", "x")  # Ensure the correct order of

        # dimensions
        data = data.sortby([-data.latitude[:, 0], data.longitude[0], data.pressure_level])
        data["longitude"].values = data["longitude"].values % 360

        assert (data["longitude"].min(axis=1) >= 0).all(), "Datavar: longitude - Some longitudes are below 0"
        assert (data["longitude"].min(axis=1) < 360).all(), "Datavar: longitude - Some longitudes are >= 360"
        assert (data["latitude"].min(axis=1) >= 0).all(), "Datavar: latitude - Some longitudes are below 0"
        assert (data["latitude"].min(axis=1) < 360).all(), "Datavar: latitude - Some longitudes are >= 360"

        x_slice = slice(*x_range)
        y_slice = slice(*y_range)
        data = data.sel(valid_time=slice(start_time, end_time), x=x_slice, y=y_slice)

        self.data = data

        self.ds_step_size_hours = dataset_step_size

        self.lead_time_steps = lead_time_hours // self.ds_step_size_hours
        self.step_size = step_size_hours // self.ds_step_size_hours
        self.normalize = normalize

        if self.normalize:
            if stats_path is None:
                self.stats = {}
            else:
                with open(stats_path, "rb") as f:
                    self.stats = pickle.load(f)

    def _get_sample(self, current_idx: int, previous_idx: int, var: str) -> torch.Tensor:

        if previous_idx is not None:
            date_range = [previous_idx, current_idx]
        else:
            date_range = [current_idx]

        if var == "slt":
            # We don't have slt in the dataset, so we just fill with zeros
            shape = self.data.isel(valid_time=date_range[0])["lsm"].values.shape
            return torch.zeros(shape)

        elif var in STATIC_VARS.values():
            data_point = self.data.isel(valid_time=date_range[0])[var].values
            data_point = torch.from_numpy(data_point)
            return data_point

        else:
            data_point = self.data.isel(valid_time=date_range)[var].values[None]
            data_point = torch.from_numpy(data_point)

            return data_point

    def _get_metadata(self, current_idx: int) -> Metadata:

        time = self.data.valid_time[current_idx].values
        time_dt = datetime.fromtimestamp(time.astype("datetime64[s]").astype("int"), timezone.utc)
        time_metadata = (time_dt,)

        levels = tuple(self.data.pressure_level.values.astype(int))

        return Metadata(
            lat=torch.from_numpy(self.data.latitude.values),
            lon=torch.from_numpy(self.data.longitude.values),
            time=time_metadata,
            atmos_levels=levels
        )

    def __len__(self) -> int:
        num_steps = len(self.data.valid_time) - 2 * self.lead_time_steps
        length = math.ceil(num_steps / self.step_size)

        return length

    def __getitem__(self, idx: int) -> dict[str, Batch]:
        # step size -> time delta between two steps in the dataset
        # lead time is the time between prev_idx and current idx
        # it is also the diff between t+1 and t.
        prev_idx = idx * self.step_size
        current_idx = prev_idx + self.lead_time_steps
        target_idx = current_idx + self.lead_time_steps

        batch = self._create_batch(current_idx, prev_idx)
        target_batch = self._create_batch(target_idx)

        return {"input": batch, "target": target_batch}

    def _create_batch(self, current_idx: int, prev_idx: Optional[int] = None) -> Batch:

        surf_vars = {k : self._get_sample(current_idx, prev_idx, v) for k, v in SURFACE_VARS.items() }
        atmos_vars = {k : self._get_sample(current_idx, prev_idx, v) for k, v in ATMOS_VARS.items() }
        static_vars = {k : self._get_sample(current_idx, prev_idx, v) for k, v in STATIC_VARS.items() }
        meta = self._get_metadata(current_idx)

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
    DATA_PATH = "/mnt/ssd/datasets/cerra_v1_derived.zarr"  # Replace with your dataset's path
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

        # batch['input'].metadata.time
        # (datetime.datetime(2007, 1, 2, 6, 0, tzinfo=datetime.timezone.utc),)
        # batch['input'].metadata.time[0].strftime('%Y-%m-%d %H:%M:%S')
        # '2007-01-02 06:00:00'

        # Break after 5 iterations to test quickly
        if i == 5:
            break
