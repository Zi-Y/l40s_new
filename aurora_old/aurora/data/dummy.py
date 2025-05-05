from torch.utils.data import Dataset
import torch
from datetime import datetime

from aurora_old import Batch, Metadata


class DummyDataset(Dataset):

    def __init__(self, num_samples, dim_lat=16, dim_lon=32):
        self.num_samples = num_samples
        self.dim_lat = dim_lat
        self.dim_lon = dim_lon

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate data with 3 dimensions for surf_vars and atmos_vars
        data = Batch(
            surf_vars={
                k: torch.randn(1, 2, self.dim_lat, self.dim_lon)  # (b, t, h, w)
                for k in ("2t", "10u", "10v", "msl", "tp")
            },
            static_vars={
                k: torch.randn(self.dim_lat, self.dim_lon)  # (h, w)
                for k in ("lsm", "z", "slt")
            },
            atmos_vars={
                k: torch.randn(1, 2, 13, self.dim_lat, self.dim_lon)  # (b, t, c, h, w)
                for k in ("z", "u", "v", "t", "q")
            },
            metadata=Metadata(
                lat=torch.linspace(90, -90, self.dim_lat),
                lon=torch.linspace(0, 360, self.dim_lon + 1)[:-1],
                time=(datetime(2020, 6, 1, 12, 0), ), # Single value for time
                atmos_levels=(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
            )
        )

        target = Batch(
            surf_vars={
                k: torch.randn(1, 1, self.dim_lat, self.dim_lon)  # (b, t, h, w)
                for k in ("2t", "10u", "10v", "msl", "tp")
            },
            static_vars={
                k: torch.randn(self.dim_lat, self.dim_lon)  # (h, w)
                for k in ("lsm", "z", "slt")
            },
            atmos_vars={
                k: torch.randn(1, 1, 13, self.dim_lat, self.dim_lon)  # (b, t, c, h, w)
                for k in ("z", "u", "v", "t", "q")
            },
            metadata=Metadata(
                lat=torch.linspace(90, -90, self.dim_lat),
                lon=torch.linspace(0, 360, self.dim_lon + 1)[:-1],
                time=(datetime(2020, 6, 1, 12, 0), ), # Single value for time
                atmos_levels=(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
            )
        )

        return {"input": data, "target": target}
