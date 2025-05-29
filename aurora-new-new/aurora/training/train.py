import random

import torch
from torch.optim.lr_scheduler import LambdaLR, LinearLR, ConstantLR, SequentialLR
from torch.amp import GradScaler, autocast
import torch.distributed as dist
import time
import os
import pickle

from aurora import rollout
from aurora.data.replay_buffer import ReplayBuffer
# from aurora.batch import Batch, Metadata
from aurora.training.loss import AuroraMeanAbsoluteError, compute_latitude_weights
from aurora.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    save_final_checkpoint,
)
from aurora.training.logging import (
    log_metrics,
    log_validation,
    log_validation_rmse,
    print_time,
    log_message,
    visualize_surf_vars,
)

from aurora.evaluation.metrics import mse

import warnings
# Disable the specific warning related to `epoch` in scheduler.step()
warnings.filterwarnings("ignore", message=".*epoch parameter in .scheduler.step().*")

# The following settings are to solve the problem:
# RuntimeError: CUDA error: invalid configuration argument (if image size >= 1024x1024)
# refer to https://stackoverflow.com/questions/77343471/pytorch-cuda-error-invalid-configuration-argument
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# set cudnn to be deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


RMSE_VARIABLES = [
    {"type": "surf", "name": "2t"},
    {"type": "surf", "name": "10u"},
    {"type": "surf", "name": "10v"},
    {"type": "surf", "name": "msl"},
    {"type": "surf", "name": "tp"},
    {"type": "atmos", "name": "z", "level": 500},
    {"type": "atmos", "name": "t", "level": 850},
    {"type": "atmos", "name": "q", "level": 700},
    {"type": "atmos", "name": "u", "level": 850},
    {"type": "atmos", "name": "v", "level": 850},
]


def get_lr_scheduler(optimizer, num_batches, cfg):
    def lr_lambda(current_step):
        # Half cosine decay (after warm-up)
        progress = float(current_step - cfg.lr_scheduler.warmup_steps) / float(
            max(1, cfg.task.total_steps - cfg.lr_scheduler.warmup_steps)
        )
        progress_tensor = torch.tensor(progress)  # Use the model's device

        return (
            0.5
            * (1.0 + torch.cos(torch.pi * progress_tensor))
            * (cfg.lr_scheduler.start_lr - cfg.lr_scheduler.final_lr)
            + cfg.lr_scheduler.final_lr
        )

    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-10, end_factor=1.0, total_iters=cfg.lr_scheduler.warmup_steps
    )

    # Scheduler setup is based on phase
    if cfg.task.phase == "pretraining":
        second_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    else:
        second_scheduler = ConstantLR(
            optimizer, factor=1.0, total_iters=num_batches - cfg.lr_scheduler.warmup_steps
        )

    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, second_scheduler],
        milestones=[cfg.lr_scheduler.warmup_steps],
    )


def train(model, train_dataloader, val_dataloader, cfg, device):
    """
    Train the given model using mixed precision and an adjustable learning rate schedule.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_dataloader (torch.utils.data.DataLoader): DataLoader providing the training dataset.
        val_dataloader (torch.utils.data.DataLoader): DataLoader providing the validation dataset.
        cfg (DictConfig): Hydra configuration object containing hyperparameters.
        device (torch.device): The device to which the model and data is moved.
    Returns:
        None: The function trains the model in-place

        - Mixed precision is used for improved performance on supported hardware.

    Example:
        >>> train(model, train_dataloader, cfg, torch.device("cuda:0"))
    """
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        if cfg.task.distributed:
            raise RuntimeError("Please use torchrun to start the training.")
        else:
            local_rank = 0

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    if local_rank >= num_gpus:
        raise RuntimeError(f"Local rank {local_rank} exceeds the number of available GPUs {num_gpus}.")

    decay_params = {k: True for k, v in model.named_parameters() if 'weight' in k and not 'norm' in k}
    decay_opt_params = [v for k, v in model.named_parameters() if k in decay_params]
    no_decay_opt_params = [v for k, v in model.named_parameters() if k not in decay_params]
    optimizer = torch.optim.AdamW(
        params=[{'params': decay_opt_params}, {'params': no_decay_opt_params, 'weight_decay': 0}],
        lr=cfg.optimizer.constant_lr,
        weight_decay=cfg.optimizer.weight_decay
    )

    criterion = None
    num_batches = len(train_dataloader)
    num_epochs = (cfg.task.total_steps + num_batches - 1) // num_batches

    model.train()

    num_val_batches = len(val_dataloader)
    total_loss = 0.0
    best_val_loss = float('inf')

    val_interval = cfg.validation.validation_interval

    evaluate_rmse = True
    if cfg.dataset.common._target_ == "aurora.data.dummy.DummyDataset":
        evaluate_rmse = False

    scheduler = get_lr_scheduler(optimizer, num_batches, cfg)
    scaler = GradScaler()  # proper scaling of gradients for mixed precision training.

    if cfg.checkpoint.continue_training:
        ckpt_path = os.path.join(cfg.checkpoint.ckpt_dir, cfg.checkpoint.ckpt_file)
        try:
            start_epoch, global_step = load_checkpoint(local_rank, model, optimizer, scheduler, scaler, ckpt_path)
            wandb_step = global_step

            if cfg.task.distributed:
                dist.barrier()
        except RuntimeError as e:
            raise RuntimeError(f"Error loading checkpoint: {e}")
    else:
        start_epoch = 0
        global_step = 0
        wandb_step = 0


    start_time = time.time()
    log_message("Data is loaded")
    print_time("training_start", start_time)

    if cfg.task.phase == "rollout_long_buffer":
        # If using a replay buffer, there is no need to perform the rollout explicitly
        # The replay buffer will handle the rollout internally
        rollout_steps = 1
    else:
        rollout_steps = cfg.task.rollout_steps

    for epoch in range(start_epoch, num_epochs):
        if cfg.task.distributed:
            train_dataloader.sampler.set_epoch(epoch)
        epoch_start_time = time.time()

        for i, sample in enumerate(train_dataloader):
            batch = sample["input"].to(device, non_blocking=True)

            # remove non-existent RMSE variables from batch input
            if (epoch==start_epoch) and (i==0):
                # Get the variable information from the batch
                surf_vars = batch.surf_vars
                atmos_vars = batch.atmos_vars
                atmos_levels = batch.metadata.atmos_levels

                # Iterate backwards through the list to avoid index issues during deletion
                for k in range(len(RMSE_VARIABLES) - 1, -1, -1):
                    var = RMSE_VARIABLES[k]
                    if var["type"] == "surf":
                        if var["name"] not in surf_vars:
                            print(f"Surface variable {var['name']} does not exist, deleting this entry.")
                            del RMSE_VARIABLES[k]
                    elif var["type"] == "atmos":
                        # Check if the variable name and its corresponding level exist
                        if var["name"] not in atmos_vars or var["level"] not in atmos_levels:
                            print(f"Atmospheric variable {var['name']} or its level {var['level']} does not exist, deleting this entry.")
                            del RMSE_VARIABLES[k]

                latitude_weights = compute_latitude_weights(batch.metadata.lat)
                criterion = AuroraMeanAbsoluteError(variable_weights=cfg.variable_weights,
                                                    latitude_weights=latitude_weights)


            optimizer.zero_grad(set_to_none=True)
            loss = torch.tensor(0.0, device=device)

            # Mixed precision forward pass
            with autocast("cuda", dtype=torch.bfloat16):
                if cfg.task.phase == "rollout_long":
                    rollout_steps_batch = torch.randint(1, rollout_steps, (1,), device=device)
                    torch.distributed.broadcast(rollout_steps_batch, src=0)  # Sync the rollout steps across all GPUs
                    rollout_steps_batch = rollout_steps_batch.cpu().item()
                    with torch.no_grad():
                        step = 0
                        for prediction, rollout_input in rollout(model, batch, rollout_steps_batch):
                            step += 1
                else:
                    step = 0
                    for prediction, _ in rollout(model, batch, rollout_steps):
                        targets = sample[f"target_{step}"].to(device, non_blocking=True)
                        loss += criterion(prediction, targets) / rollout_steps
                        step += 1

                    if cfg.task.phase == "rollout_long_buffer":
                        train_dataloader.add_rollout_samples(sample, prediction)

            if cfg.task.phase == "rollout_long":
                # We apparently can't use autocast and torch.no_grad together and then do another
                # forward pass with grad, so we do it separately
                # See https://discuss.pytorch.org/t/autocast-and-torch-no-grad-unexpected-behaviour/93475/3
                # and https://github.com/pytorch/pytorch/issues/112583
                with autocast("cuda", dtype=torch.bfloat16):
                    targets = sample[f"target_{step}"].to(device, non_blocking=True)
                    pred = model(rollout_input)
                    loss = criterion(pred, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            global_step += 1
            if cfg.task.distributed and dist.is_initialized():
                dist.all_reduce(loss)
                total_loss += loss.item() / dist.get_world_size()
            else:
                total_loss += loss.item()


            if local_rank == 0:
                wandb_step += 1
                elapsed_time = time.time() - start_time
                steps_per_sec = global_step / elapsed_time
                remaining_steps = cfg.task.total_steps - global_step
                eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                step_stats = {
                    "epoch": epoch + 1,
                    "num_epochs": num_epochs,
                    "global_step": global_step,
                    "total_steps": cfg.task.total_steps,
                    "wandb_step": wandb_step,
                    "batch_index": i,
                    "num_batches": num_batches,
                    "steps_per_sec": steps_per_sec,
                }

                timing_stats = {"eta_seconds": eta_seconds}
                lr = optimizer.param_groups[0]["lr"]
                log_metrics(step_stats, loss, lr, timing_stats)

            if global_step >= cfg.task.total_steps:
                # Reached the total number of steps
                break


            # 12904 steps  ==> 1 epoch
            if global_step % val_interval == 0:
                model.eval()
                total_val_loss = 0.0

                if evaluate_rmse:
                    total_mse_results = {f"{var['name']}{'_' + str(var['level']) if 'level' in var else ''}_step{step}": 0.0
                                for var in RMSE_VARIABLES for step in range(rollout_steps)}
                    stats_dict = val_dataloader.dataset.stats

                with torch.no_grad():  # No gradient computation for evaluation
                    for j, sample in enumerate(val_dataloader):
                        batch = sample["input"].to(device, non_blocking=True)
                        val_loss = torch.tensor(0.0, device=device)

                        # Forward pass
                        with autocast("cuda", dtype=torch.bfloat16):
                            predictions = []
                            targets = []
                            rollout_inputs = [batch]

                            step = 0
                            for prediction, rollout_input in rollout(model, batch, rollout_steps):
                                target = sample[f"target_{step}"].to(device, non_blocking=True)
                                val_loss += criterion(prediction, target) / rollout_steps

                                predictions.append(prediction)
                                rollout_inputs.append(rollout_input)
                                targets.append(target)

                                if j == 0:
                                    visualize_surf_vars(
                                        rollout_inputs[step],
                                        prediction,
                                        target,
                                        train_dataloader.dataset.stats,
                                        epoch + 1,
                                        rollout_step=step,
                                        save_on_next_step=True,
                                    )

                                if evaluate_rmse:
                                    prediction_unnormalized = prediction.unnormalise(stats=stats_dict)
                                    target_unnormalized = target.unnormalise(stats=stats_dict)

                                    for var in RMSE_VARIABLES:
                                        if var['type'] == 'surf':
                                            current_mse = mse(prediction_unnormalized, target_unnormalized,
                                                            variable=var['name'], latitude_weights=latitude_weights)
                                        else:
                                            current_mse = mse(prediction_unnormalized, target_unnormalized,
                                                            variable=var['name'], level=var['level'], latitude_weights=latitude_weights)

                                        if cfg.task.distributed and dist.is_initialized():
                                            dist.all_reduce(current_mse)
                                            current_mse = current_mse / dist.get_world_size()

                                        result_key = f"{var['name']}{'_' + str(var['level']) if 'level' in var else ''}_step{step}"
                                        total_mse_results[result_key] += current_mse.item()

                                step += 1

                        if cfg.task.distributed and dist.is_initialized():
                            dist.all_reduce(val_loss)
                            val_loss = val_loss / dist.get_world_size()
                        total_val_loss += val_loss.item()



                val_loss = total_val_loss / num_val_batches
                if evaluate_rmse:
                    rmse_results = {
                        key: (total_mse / num_val_batches) ** 0.5
                        for key, total_mse in total_mse_results.items()
                    }

                if cfg.task.distributed and dist.is_initialized():
                    dist.barrier()

                model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                    # Save best model if validation loss improved
                    # We run save_checkpoint on all ranks, but only rank 0 will save the model
                    if local_rank == 0:
                        os.makedirs(cfg.checkpoint.ckpt_dir, exist_ok=True)
                    best_model_file = f"aurora-{cfg.task.model_name}-{cfg.task.phase}-step{global_step}-best.ckpt"
                    best_model_path = os.path.join(cfg.checkpoint.ckpt_dir, best_model_file)
                    save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_model_path, global_step)

                if local_rank == 0:
                    # Log validation intermediate results
                    step_stats = {
                        "epoch": epoch + 1,
                        "num_epochs": num_epochs,
                        "batch_index": j,
                        "num_batches": num_val_batches,
                        "global_step": global_step,
                    }
                    log_validation(step_stats, val_loss, best_val_loss)
                    if evaluate_rmse:
                        log_validation_rmse(step_stats, rmse_results)

                    # Print information only on the main process
                    epoch_time = time.time() - epoch_start_time
                    print_time(f"epoch_{epoch + 1}_duration", epoch_time)

    train_duration = time.time() - start_time
    print_time("train_duration", train_duration)
