import torch
from torch.optim.lr_scheduler import LambdaLR, LinearLR, ConstantLR, SequentialLR
from torch.amp import GradScaler, autocast
import torch.distributed as dist
import time
import os
from aurora_old.training.loss import AuroraMeanAbsoluteError
from aurora_old.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    save_final_checkpoint,
)
from aurora_old.training.logging import log_metrics, print_time, log_message

# The following settings are to solve the problem:
# RuntimeError: CUDA error: invalid configuration argument (if image size >= 1024x1024)
# refer to https://stackoverflow.com/questions/77343471/pytorch-cuda-error-invalid-configuration-argument
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# set cudnn to be deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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


def train(model, dataloader, cfg, device):
    """
    Train the given model using mixed precision and an adjustable learning rate schedule.

    Args:
        model (torch.nn.Module): The model to be trained.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the training dataset.
        cfg (DictConfig): Hydra configuration object containing hyperparameters.
        device (torch.device): The device to which the model and data is moved.
    Returns:
        None: The function trains the model in-place

    Notes:
        - Mixed precision is used for improved performance on supported hardware.

    Example:
        >>> train(model, train_dataloader, cfg, torch.device("cuda:0"))
    """
    local_rank = 0
    if cfg.task.distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank == -1:
            raise RuntimeError("Please use torchrun to start the training.")
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

    criterion = AuroraMeanAbsoluteError(variable_weights=cfg.variable_weights)
    num_batches = len(dataloader)

    num_epochs = (cfg.task.total_steps + num_batches - 1) // num_batches

    model.train()
    total_loss = 0.0

    scheduler = get_lr_scheduler(optimizer, num_batches, cfg)
    scaler = GradScaler()  # proper scaling of gradients for mixed precision training.

    if cfg.checkpoint.continue_training:
        ckpt_path = os.path.join(cfg.checkpoint.ckpt_dir, cfg.checkpoint.ckpt_file)
        try:
            # 判断是否为分布式训练
            if cfg.task.distributed:
                # 多 GPU 训练，加载检查点
                start_epoch, global_step = load_checkpoint(local_rank, model, optimizer, scheduler, scaler, ckpt_path, cfg.task.distributed)
                dist.barrier()  # 保持多进程同步
            else:
                # 单 GPU 训练，加载检查点
                start_epoch, global_step = load_checkpoint(0, model, optimizer, scheduler, scaler, ckpt_path, cfg.task.distributed)
        except RuntimeError as e:
            raise RuntimeError(f"Error loading checkpoint: {e}")
    else:
        start_epoch = 0
        global_step = 0

    start_time = time.time()
    log_message("Data is loaded")
    print_time("training_start", start_time)

    for epoch in range(start_epoch, num_epochs):
        if cfg.task.distributed:
            dataloader.sampler.set_epoch(epoch)
        # else:
        #     dataloader.set_epoch(epoch)
        epoch_start_time = time.time()

        for i, sample in enumerate(dataloader):
            batch = sample["input"].to(device, non_blocking=True)
            target = sample["target"].to(device, non_blocking=True)
            # sample['input'].metadata.time
            # (datetime.datetime(2007, 1, 2, 6, 0, tzinfo=datetime.timezone.utc),)
            # sample['input'].metadata.time[0].strftime('%Y-%m-%d %H:%M:%S')
            # '2007-01-02 06:00:00'
            # batch.metadata.lat      size=1056,1056

            optimizer.zero_grad(set_to_none=True)

            # Mixed precision forward pass
            with autocast("cuda", dtype=torch.bfloat16):
                prediction = model(batch)
                loss = criterion(prediction, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            global_step += 1
            if cfg.task.distributed:
                dist.all_reduce(loss)
                total_loss += loss.item() / dist.get_world_size()
            else:
                total_loss += loss.item()

            if local_rank == 0:
                elapsed_time = time.time() - start_time
                steps_per_sec = global_step / elapsed_time
                remaining_steps = cfg.task.total_steps - global_step
                eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                step_stats = {
                    "epoch": epoch + 1,
                    "num_epochs": num_epochs,
                    "global_step": global_step,
                    "total_steps": cfg.task.total_steps,
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

        if (epoch + 1) % cfg.checkpoint.ckpt_epoch == 0:
                os.makedirs(cfg.checkpoint.ckpt_dir, exist_ok=True)
                ckpt_file = f"aurora-{cfg.task.model_name}-{cfg.task.phase}-{epoch + 1}-{global_step}.ckpt"
                ckpt_path = os.path.join(cfg.checkpoint.ckpt_dir, ckpt_file)
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, ckpt_path, global_step)

                # Print information only on the main process
        epoch_time = time.time() - epoch_start_time
        print_time(f"epoch_{epoch + 1}_duration", epoch_time)

    save_final_checkpoint(
        model,
        os.path.join(
            cfg.checkpoint.ckpt_dir, f"aurora-{cfg.task.model_name}-{cfg.task.phase}.ckpt"
        ),
    )
    log_message(
        f"Finetuned Model saved: aurora-{cfg.task.model_name}-{cfg.task.phase}.ckpt",
    )

    train_duration = time.time() - start_time
    print_time("train_duration", train_duration)
