import os
import torch
import torch.distributed as dist
import xarray as xr
import numpy as np
from numcodecs import Blosc
from tqdm import tqdm

from aurora import rollout
from aurora.data.cerra import ATMOS_VAR_MAPPING, SURFACE_VAR_MAPPING
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

    lat_name = "latitude" if cfg.task.use_wb2_format else "y"
    lon_name = "longitude" if cfg.task.use_wb2_format else "x"
    level_name = "level" if cfg.task.use_wb2_format else "pressure_level"

    all_levels = inputs.metadata.atmos_levels
    kept_levels = [lvl for lvl in all_levels if lvl in cfg.task.save_levels]
    kept_levels_idx = [all_levels.index(lvl) for lvl in kept_levels]

    data_vars = {}

    for var in preds[0].atmos_vars:
        if var not in cfg.task.save_variables:
            continue
        var_mapped = ATMOS_VAR_MAPPING[var]
        var_values = np.concatenate([pred.atmos_vars[var] for pred in preds], axis=1)
        var_values = var_values[:, :, kept_levels_idx, :, :]
        data_vars[var_mapped] = (["time", "prediction_timedelta", level_name, lat_name, lon_name], var_values)

    for var in preds[0].surf_vars:
        if var not in cfg.task.save_variables:
            continue
        var_mapped = SURFACE_VAR_MAPPING[var]
        var_values = np.concatenate([pred.surf_vars[var] for pred in preds], axis=1)
        data_vars[var_mapped] = (["time", "prediction_timedelta", lat_name, lon_name], var_values)

    if cfg.task.use_wb2_format:
        ds = xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                latitude=inputs.metadata.lat.cpu().numpy()[:, 0],
                longitude=inputs.metadata.lon.cpu().numpy()[0, :],
                cerra_latitude=([lat_name, lon_name], inputs.metadata.lat.cpu().numpy()),
                cerra_longitude=([lat_name, lon_name], inputs.metadata.lon.cpu().numpy()),
                level=np.array(kept_levels),
                time=time,
                prediction_timedelta=timedeltas,
            ),
            attrs=dict(description="Weather forecasts for Europe."),
        )

    else:
        ds = xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                latitude=(["y", "x"], inputs.metadata.lat.cpu().numpy()),
                longitude=(["y", "x"], inputs.metadata.lon.cpu().numpy()),
                pressure_level=np.array(kept_levels),
                time=time,
                prediction_timedelta=timedeltas,
            ),
            attrs=dict(description="Weather forecasts for Europe."),
        )

    num_timedeltas = len(timedeltas)
    num_pressure_levels = len(kept_levels)
    num_y = ds[lat_name].values.shape[0]
    num_x = ds[lon_name].values.shape[0]
    ds = ds.chunk({
        "time": 1,
        "prediction_timedelta": num_timedeltas,
        level_name: num_pressure_levels,
        lat_name: num_y,
        lon_name: num_x
    })

    outdir = cfg.task.output_dir

    if os.path.exists(outdir):
        ds.to_zarr(outdir, append_dim="time", mode="a")
    else:
        # We use lz4hc because it combines good compression with fast decompression
        # It's relatively slow to compress, but we only need to do that once
        compressor = Blosc(cname='lz4hc', clevel=9, shuffle=Blosc.SHUFFLE)
        encoding = {var: {"compressor": compressor} for var in ds.data_vars}

        # It is important to set the encoding for the time variable explicitly, otherwise xarray might select
        # an encoding that is not compatible with the later values, resulting in wrong time values
        # See https://github.com/pydata/xarray/issues/5969 and https://github.com/pydata/xarray/issues/3942
        encoding.update({"time": {"units": "nanoseconds since 1970-01-01"}})

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

    with torch.no_grad(): 
        for batch in tqdm(dataloader):
            inputs = batch["input"].to(device)

            preds = [pred.to("cpu") for pred, _ in rollout(model, inputs, steps=max_lead_time_steps)]
            preds = [pred.unnormalise(stats=stats_dict) for pred in preds]

            # Save forecasted values
            save_forecast(inputs, preds, cfg)


def forecast_ensemble(model: torch.nn.Module,
                          dataloader: torch.utils.data.DataLoader,
                          cfg,  # Hydra DictConfig
                          device: torch.device):
    """
    Forecasts values using an ensemble of trained models, with all computations on GPU.
    Predictions from models are averaged on GPU. Averaged forecasts are then conceptually saved.

    Args:
        model (torch.nn.Module): An instance of the model architecture.
                                 Weights will be loaded sequentially. Model will be moved to `device`.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the dataset for forecasting.
        cfg (DictConfig): Hydra configuration object. Expected to contain:
                          - cfg.task.checkpoint_paths (list of str): Paths to model checkpoints.
                          - cfg.task.lead_times (list of int): Lead times for forecasting.
                          - cfg.model.lead_time_hours (int): Model's output step duration in hours.
                          - cfg.task.distributed (bool, optional): Flag for distributed environment.
                          - cfg.output_dir (str, optional): Path to save outputs.
        device (torch.device): The GPU device (e.g., "cuda:0") for model and all computations.

    Returns:
        None: The function is expected to save the forecasted values (GPU tensors).
              Saving logic might require moving tensors to CPU before writing to disk.

    Example:
        >>> # cfg.task.checkpoint_paths = ["/path/to/ckpt1.pt", ..., "/path/to/ckpt5.pt"]
        >>> # forecast_ensemble(model_instance, val_dataloader, cfg, torch.device("cuda:0"))
    """

    if not device.type == 'cuda':
        print(f"Warning: Device is set to '{device.type}'. This function is optimized for GPU ('cuda').")

    if cfg.task.get("distributed", False):
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank != 0:
            raise RuntimeError(
                "Forecasting with this function is designed for a single process (typically rank 0)."
            )
    else:
        local_rank = 0

    checkpoint_paths = cfg.task.checkpoint_paths
    if not isinstance(checkpoint_paths, list) or not checkpoint_paths:
        raise ValueError("cfg.task.checkpoint_paths must be a non-empty list of checkpoint paths.")

    num_models = len(checkpoint_paths)

    print(
        f"Received {num_models} checkpoint paths. Proceeding with {num_models}.")
        # To enforce exactly 5:
        # raise ValueError(f"Expected 5 checkpoint paths, but got {num_models}")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)  # Ensure the model instance is on the target GPU device

    try:
        stats_dict = dataloader.dataset.stats
    except AttributeError:
        raise AttributeError("dataloader.dataset does not have a 'stats' attribute needed for unnormalisation.")
    #
    # # Move tensors within stats_dict to the target device
    # stats_dict = {}
    # for key, value in stats_dict_cpu.items():
    #     if isinstance(value, torch.Tensor):
    #         stats_dict[key] = value.to(device)
    #     else:
    #         stats_dict[key] = value

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Ensemble Forecasting on GPU")):
        inputs = batch["input"].to(device)

        batch_all_models_preds_unnorm_gpu = []  # Stores List[List[Tensor]]

        for model_idx, ckpt_path in enumerate(checkpoint_paths):
            try:
                # `load_checkpoint` should load weights into `model` (already on device).
                # Ensure your `load_checkpoint` handles map_location correctly if loading CPU-saved checkpoints.
                _, _ = load_checkpoint(local_rank, model, None, None, None, ckpt_path)
            except RuntimeError as e:
                raise RuntimeError(f"Error loading checkpoint {ckpt_path} (model {model_idx + 1}): {e}")
            except NameError:
                raise NameError("Function 'load_checkpoint' is not defined. Import or define it.")

            with torch.no_grad():
                try:
                    # `rollout` returns prediction objects/tensors already on the GPU.
                    current_model_preds = []
                    # Each `pred_obj_gpu` is assumed to be on the GPU
                    for pred_obj_gpu, _ in rollout(model, inputs, steps=cfg.task.rollout_steps):
                        current_model_preds.append(pred_obj_gpu.unnormalise(stats=stats_dict))
                except NameError:
                    raise NameError("Function 'rollout' is not defined. Import or define it.")


            batch_all_models_preds_unnorm_gpu.append(current_model_preds)



    return None