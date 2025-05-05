import hydra
import os
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from aurora import Aurora, AuroraSmall
from aurora.data.dummy import DummyDataset
from aurora.data.collate import collate_fn
from aurora.data.cerra import CerraDataset
from aurora.training.train import train
from aurora.training.inference import inference
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


@hydra.main(config_name="task", config_path="configs", version_base="1.3.2")
def main(cfg):
    check_and_start_debugger()
    local_rank = 0
    if cfg.task.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        setup_distributed(local_rank)
    initialize_wandb(cfg)

    model_class = MODEL_REGISTRY.get(cfg.task.model_name)
    if model_class is None:
        raise ValueError(f"Unknown model: {cfg.task.model_name}")

    dataset = instantiate(cfg.dataset)
    sampler = DistributedSampler(dataset, drop_last=True) if cfg.task.distributed else None

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.dataloader.batch_size,
        sampler=sampler,
        shuffle=not cfg.task.distributed,
        drop_last=not cfg.task.distributed,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True if cfg.dataloader.num_workers > 0 else False,
        multiprocessing_context="forkserver" if cfg.dataloader.num_workers > 0 else None,
        num_workers=cfg.dataloader.num_workers,
    )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model = model_class(**cfg.model)

    if cfg.task.phase == "finetuning":
        if not cfg.checkpoint.continue_training:
            model_checkpoint = MODEL_CHECKPOINT_REGISTRY[cfg.task.model_name]
            model.load_checkpoint("microsoft/aurora", model_checkpoint, strict=False)
        
    model.to(device)

    if cfg.task.use_activation_checkpointing:
        model.configure_activation_checkpointing()

    if cfg.task.distributed:
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    model = torch.compile(model, dynamic=False) if cfg.task.use_torch_compile else model

    if cfg.task.task == "train":
        train(model, dataloader, cfg, device)

    if cfg.task.task == "inference":
        inference(model, dataloader, cfg, device)

    if cfg.task.distributed:
        cleanup()


if __name__ == "__main__":
    main()
