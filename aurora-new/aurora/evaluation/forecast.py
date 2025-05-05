import os
import torch.distributed as dist
import xarray as xr
import numpy as np
from numcodecs import Blosc
from tqdm import tqdm

from aurora import rollout
from aurora.training.checkpoint import load_checkpoint


def save_forecast(inputs, preds, cfg):
    """
    Saves the forecast batch as zarr file in the output directory.
    """

    # Remove timezone from the string because numpy doesn't support it
    time = [np.datetime64(str(t)[:-6]) for t in inputs.metadata.time]
    time = np.array(time).astype("datetime64[ns]")
    timedeltas = [np.timedelta64(lt, 'h').astype("timedelta64[ns]") for lt in cfg.task.lead_times]
    timedeltas = np.array(timedeltas)

    lat_coord = "latitude" if cfg.task.use_wb2_format else "y"
    lon_coord = "longitude" if cfg.task.use_wb2_format else "x"

    data_vars = {}

    for var in preds[0].atmos_vars:
        var_values = np.concatenate([pred.atmos_vars[var] for pred in preds], axis=1)
        data_vars[var] = (["time", "prediction_timedelta", "pressure_level", lat_coord, lon_coord], var_values)

    for var in preds[0].surf_vars:
        var_values = np.concatenate([pred.surf_vars[var] for pred in preds], axis=1)
        data_vars[var] = (["time", "prediction_timedelta", lat_coord, lon_coord], var_values)

    if cfg.task.use_wb2_format:
        data_vars["cerra_latitude"] = ([lat_coord, lon_coord], inputs.metadata.lat.cpu().numpy())
        data_vars["cerra_longitude"] = ([lat_coord, lon_coord], inputs.metadata.lon.cpu().numpy())

        ds = xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                latitude=inputs.metadata.lat.cpu().numpy()[:, 0],
                longitude=inputs.metadata.lat.cpu().numpy()[0],
                level=np.array(inputs.metadata.atmos_levels),
                time=time,
                prediction_timedelta=timedeltas,
            ),
            attrs=dict(description="Weather forecasts for Europe."),
        )

        num_timedeltas = len(timedeltas)
        num_pressure_levels = len(inputs.metadata.atmos_levels)
        num_y = ds[lat_coord].values.shape[0]
        num_x = ds[lon_coord].values.shape[0]
        ds = ds.chunk({
            "time": 1,
            "prediction_timedelta": num_timedeltas,
            "level": num_pressure_levels,
            lat_coord: num_y,
            lon_coord: num_x
        })
    else:
        ds = xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                latitude=(["y", "x"], inputs.metadata.lat.cpu().numpy()),
                longitude=(["y", "x"], inputs.metadata.lat.cpu().numpy()),
                level=np.array(inputs.metadata.atmos_levels),
                time=time,
                prediction_timedelta=timedeltas,
            ),
            attrs=dict(description="Weather forecasts for Europe."),
        )

        num_timedeltas = len(timedeltas)
        num_pressure_levels = len(inputs.metadata.atmos_levels)
        num_y = ds.y.values.shape[0]
        num_x = ds.x.values.shape[0]
        ds = ds.chunk({
            "time": 1,
            "prediction_timedelta": num_timedeltas,
            "level": num_pressure_levels,
            "y": num_y,
            "x": num_x
        })

    outdir = cfg.task.output_dir

    if os.path.exists(outdir):
        ds.to_zarr(outdir, append_dim="time", mode="a")
    else:
        # We use lz4hc because it combines good compression with fast decompression
        # It's relatively slow to compress, but we only need to do that once
        compressor = Blosc(cname='lz4hc', clevel=9, shuffle=Blosc.SHUFFLE)
        encoding = {var: {"compressor": compressor} for var in ds.data_vars}
        ds.to_zarr(outdir, mode="w-", encoding=encoding)


def forecast(model, dataloader, cfg, device):
    """
    Forecasts values using the trained model.

    Args:
        model (torch.nn.Module): The model to be trained.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the training dataset.
        cfg (DictConfig): Hydra configuration object containing hyperparameters.
        device (torch.device): The device to which the model and data is moved.
    Returns:
        None: The forecasted values are saved to the output directory.

    Example:
        >>> forecast(model, val_dataloader, cfg, torch.device("cuda:0"))
    """

    if cfg.task.distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank != 0:
            raise RuntimeError("Cannot use multiple GPUs for forecasting.")
    else:
        local_rank = 0

    ckpt_path = cfg.task.checkpoint_path
    try:
        _, _ = load_checkpoint(local_rank, model, None, None, None, ckpt_path)

        if cfg.task.distributed:
            dist.barrier()
    except RuntimeError as e:
        raise RuntimeError(f"Error loading checkpoint: {e}")

    lead_times = cfg.task.lead_times
    max_lead_time = max(lead_times)
    max_lead_time_steps = max_lead_time // cfg.model.lead_time_hours
    assert all([lead_time % cfg.model.lead_time_hours == 0 for lead_time in lead_times]), \
        "Lead times must be multiples of model lead time."

    # Disable gradients and set model to eval mode
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    stats_dict = dataloader.dataset.stats

    for batch in tqdm(dataloader):
        inputs = batch["input"].to(device)

        preds = [pred.to("cpu") for pred, _ in rollout(model, inputs, steps=max_lead_time_steps)]
        preds = [pred.unnormalise(stats=stats_dict) for pred in preds]

        # Save forecasted values
        save_forecast(inputs, preds, cfg)
