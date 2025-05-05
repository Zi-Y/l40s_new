import sys
from logging import info

import torch
import wandb
import datetime
import json


def log_on_rank_zero(func):
    """
    Decorator to ensure that the decorated function is only executed
    when rank is 0 (i.e., the main process in a distributed setup).
    """

    def wrapper(*args, **kwargs):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            return func(*args, **kwargs)
        return None  # Skip logging if not rank 0

    return wrapper


def is_serializable(value):
    """
    Check if a value is JSON-serializable.

    Args:
        value: The value to check.

    Returns:
        bool: True if the value is JSON-serializable, False otherwise.
    """
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False


def sanitize_cfg(cfg):
    """
    Forcefully sanitize the configuration to ensure all nested dictionaries 
    are recognized as standard Python dictionaries.
    """
    sanitized = {}
    for key, value in cfg.items():
        if isinstance(value, dict) or hasattr(value, "items"):  # Check if it's dictionary-like
            sanitized[key] = sanitize_cfg(dict(value))  # Convert to a standard dictionary
        elif isinstance(value, (int, float, str, bool, list)):  # Supported basic types
            sanitized[key] = value
        else:
            print(f"Warning: Key '{key}' has non-serializable value '{value}', converting to string.")
            sanitized[key] = str(value)  # Convert other types to string
    return sanitized


@log_on_rank_zero
def initialize_wandb(cfg):
    """
    Initialize Weights & Biases logging.

    Args:
        cfg: The Hydra configuration object.
    """
    wandb_cfg = sanitize_cfg(cfg)
    wandb_mode = "online" if cfg.logging.use_wandb else "disabled"
    wandb.init(
        project=cfg.logging.project_name,
        config=wandb_cfg,
        name=generate_run_name(cfg),
        group=cfg.logging.group_name,
        job_type=cfg.task.task,
        entity="hpi-deep-learning",
        mode=wandb_mode,
    )


@log_on_rank_zero
def generate_run_name(cfg):
    """
    Generate a descriptive and unique run name based on configuration parameters.

    Args:
        cfg: The Hydra configuration object.
    Returns:
        str: A formatted run name.
    """
    task = cfg.task.task
    batch_size = cfg.dataloader.batch_size
    warmup_steps = cfg.lr_scheduler.warmup_steps
    start_lr = cfg.lr_scheduler.start_lr
    final_lr = cfg.lr_scheduler.final_lr
    total_steps = cfg.task.total_steps

    return f"test_{task}_bs{batch_size}_wu{warmup_steps}_lr{start_lr}-{final_lr}_steps{total_steps}"


@log_on_rank_zero
def log_metrics(step_stats, loss, lr, timing_stats):
    """
    Log metrics to Weights & Biases.

    Args:
        step_stats (dict): Dictionary containing step_stats such as global_step, epoch, etc.
        loss (Tensor): The current loss value.
        lr (float): The current learning rate.
        timing_stats (dict): Dictionary containing timing statistics.
    """
    eta_seconds = timing_stats.get("eta_seconds", 0)
    eta_hours = eta_seconds / 3600
    sys.stdout.write("\r"
        f"Epoch {step_stats['epoch']}/{step_stats['num_epochs']} | "
        f"Global Step: {step_stats['global_step']}/{step_stats['total_steps']} | "
        f"Batch {step_stats['batch_index'] + 1}/{step_stats['num_batches']} | "
        f"Loss: {loss.item():.4f} | "
        f"LR: {lr:.2e} | "
        f"Steps/sec: {step_stats['steps_per_sec']:.2f} | "
        f"ETA: {eta_hours:.2f} hours"
    )
    sys.stdout.flush()
    wandb.log(
        {
            "global_step": step_stats["global_step"],
            "epoch": step_stats["epoch"],
            "loss": loss.item(),
            "batch": step_stats["batch_index"] + 1,
            "learning_rate": lr,
            "steps/sec": step_stats["steps_per_sec"],
            "eta": eta_hours,
        }
    )


@log_on_rank_zero
def print_time(description: str, timestamp: float):
    """
    Print a human-readable timestamp to the console.

    Args:
        description (str): The description of the event being logged.
        timestamp (float): The timestamp as returned by time.time().
    """
    # Convert the timestamp to a datetime object
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    info(f"{description}: { dt_object.strftime('%Y-%m-%d %H:%M:%S')}")


@log_on_rank_zero
def log_message(description):
    info(description)
