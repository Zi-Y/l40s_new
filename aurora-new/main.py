import hydra
import os
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from aurora import Aurora, AuroraSmall
from aurora.data.collate import collate_fn
from aurora.data.replay_buffer import ReplayBuffer
from aurora.evaluation.forecast import forecast
from aurora.training.train import train
from aurora.training.logging import initialize_wandb

# Disable DDP optimization for now
# See https://github.com/pytorch/pytorch/issues/134182
torch._dynamo.config.optimize_ddp = False

os.environ["NCCL_P2P_LEVEL"] = "NVL"

MODEL_REGISTRY = {"Aurora": Aurora, "AuroraSmall": AuroraSmall}
MODEL_CHECKPOINT_REGISTRY = {"Aurora": "aurora-0.25-pretrained.ckpt", "AuroraSmall": "aurora-0.25-small-pretrained.ckpt"}


def check_and_start_debugger():
    """Check if PyCharm remote debugger should be started."""
    import os
    debug_port = int(os.environ.get("REMOTE_PYCHARM_DEBUG_PORT", 12034))
    if os.environ.get("REMOTE_PYCHARM_DEBUG_SESSION", False):
        import pydevd_pycharm
        pydevd_pycharm.settrace(
            "localhost",
            port=debug_port,
            stdoutToServer=True,
            stderrToServer=True,
            suspend=False,
        )

def cleanup():
    dist.destroy_process_group()


def setup_distributed(local_rank):
    """Initializes the distributed environment."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    dist.barrier(device_ids=[local_rank])


@hydra.main(config_name="task", config_path="configs", version_base="1.3.2")
def main(cfg):
    check_and_start_debugger()
    if cfg.task.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        setup_distributed(local_rank)
    else:
        local_rank = 0
    initialize_wandb(cfg)

    model_class = MODEL_REGISTRY.get(cfg.task.model_name)
    if model_class is None:
        raise ValueError(f"Unknown model: {cfg.task.model_name}")

    if hasattr(cfg.dataset, "train"):
        train_cfg = OmegaConf.merge(
            OmegaConf.to_container(cfg.dataset.common, resolve=True),
            OmegaConf.to_container(cfg.dataset.train, resolve=True)
        )
        train_dataset = instantiate(train_cfg)
    else:
        train_dataset = instantiate(cfg.dataset.common)  # For DummyDataset

    if hasattr(cfg.dataset, "val"):
        val_cfg = OmegaConf.merge(
            OmegaConf.to_container(cfg.dataset.common, resolve=True),
            OmegaConf.to_container(cfg.dataset.val, resolve=True)
        )
        val_dataset = instantiate(val_cfg)
    else:
        val_dataset = instantiate(cfg.dataset.common)  # For DummyDataset

    train_sampler = DistributedSampler(train_dataset, drop_last=True) if cfg.task.distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False) if cfg.task.distributed else None

    if cfg.task.task == "train" and cfg.task.phase == "rollout_long_buffer":
        # If using a replay buffer, we need to set the batch size to 1 for the dataloader
        # The batching is handled by the replay buffer
        dl_batch_size = 1
    else:
        dl_batch_size = cfg.dataloader.batch_size

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=dl_batch_size,
        sampler=train_sampler,
        shuffle=not cfg.task.distributed,
        drop_last=not cfg.task.distributed,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True if cfg.dataloader.num_workers > 0 else False,
        multiprocessing_context="forkserver" if cfg.dataloader.num_workers > 0 else None,
        num_workers=cfg.dataloader.num_workers,
    )

    if cfg.task.task == "train" and cfg.task.phase == "rollout_long_buffer":
        train_dataloader = ReplayBuffer(
            train_dataloader,
            cfg.dataloader.batch_size,
            buffer_size=cfg.task.buffer_size,
            refresh_freq=cfg.task.refresh_freq,
            max_rollout_steps=cfg.task.rollout_steps
        )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.dataloader.batch_size*4,
        sampler=val_sampler,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True if cfg.dataloader.num_workers > 0 else False,
        multiprocessing_context="forkserver" if cfg.dataloader.num_workers > 0 else None,
        num_workers=cfg.dataloader.num_workers,
    )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model = model_class(**cfg.model)

    if cfg.task.task == "train" and cfg.task.phase == "finetuning":
        if not cfg.checkpoint.continue_training:
            model_checkpoint = MODEL_CHECKPOINT_REGISTRY[cfg.task.model_name]
            model.load_checkpoint("microsoft/aurora", model_checkpoint, strict=False)

    model.to(device)

    if cfg.task.use_activation_checkpointing:
        model.configure_activation_checkpointing()

    if cfg.task.distributed:
        dist.barrier()

        model = model.to(local_rank)
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    model = torch.compile(model, dynamic=False) if cfg.task.use_torch_compile else model

    if cfg.task.task == "train":
        train(model, train_dataloader, val_dataloader, cfg, device)
    elif cfg.task.task == "forecast":
        forecast(model, val_dataloader, cfg, device)

    if cfg.task.distributed:
        cleanup()


if __name__ == "__main__":
    main()
