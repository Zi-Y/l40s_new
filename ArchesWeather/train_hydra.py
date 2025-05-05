import pytorch_lightning as pl
import torch
from pathlib import Path
import shutil
from functools import partial
import subprocess
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import os
from datetime import datetime
import time

from omegaconf import OmegaConf

from infobatch import InfoBatch
import numpy as np
import warnings
import torch.distributed as dist
import wandb
import json

warnings.filterwarnings('ignore')
os.environ["NCCL_P2P_LEVEL"] = "NVL"

try:
    from pytorch_lightning.callbacks import Callback, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
except:
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger


def get_random_code():
    import string
    import random
    # generate random code that alternates letters and numbers
    l = random.choices(string.ascii_lowercase, k=3)
    n = random.choices(string.digits, k=3)
    return ''.join([f'{a}{b}' for a, b in zip(l, n)])


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
            self,
            dirpath='./',
            save_step_frequency=50000,
            path_save_base='./',
            prefix="checkpoint",
            use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

        self.dirpath = dirpath
        self.path_save_base = path_save_base
        # test_only
        # self.dirpath = '/mnt/cache/data/zi/archesmodels/'

    def on_train_batch_end(self, trainer: pl.Trainer, *args, **kwargs):
        """ Check if we should save a checkpoint after every train batch """
        if not hasattr(self, 'trainer'):
            self.trainer = trainer

        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            self.save()

    def save(self, *args, trainer=None, **kwargs):
        if trainer is None and not hasattr(self, 'trainer'):
            print('No trainer !')
            return
        if trainer is None:
            trainer = self.trainer

        global_step = trainer.global_step
        if self.use_modelcheckpoint_filename:
            filename = trainer.checkpoint_callback.filename
        else:
            filename = f"{self.prefix}_{global_step=}.ckpt"
        ckpt_path = Path(self.path_save_base) / 'checkpoints'
        ckpt_path.mkdir(exist_ok=True, parents=True)
        trainer.save_checkpoint(ckpt_path / filename)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except:
        pass

    # main_node = int(os.environ.get('SLURM_PROCID', 0)) == 0
    # print('is main node', main_node)

    # init some variables
    logger = None
    ckpt_path = None
    # delete submitit handler to let PL take care of resuming
    # signal.signal(signal.SIGTERM, signal.SIG_DFL)

    # # parameters of infobatch
    # info_batch_num_epoch = 14
    # info_batch_ratio = 0.5
    # info_batch_delta = 0.875

    # # first, check if exp exists
    # if Path(cfg.exp_dir).exists():
    #     print('Experiment already exists. Trying to resume it.')
    #     exp_cfg = OmegaConf.load(Path(cfg.exp_dir) / 'config.yaml')
    #     if cfg.resume:
    #         cfg = exp_cfg
    #     else:
    #         # check that new config and old config match
    #         if OmegaConf.to_yaml(cfg.module, resolve=True) != OmegaConf.to_yaml(exp_cfg.module):
    #             print('Module config mismatch. Exiting')
    #             print('Old config', OmegaConf.to_yaml(exp_cfg.module))
    #             print('New config', OmegaConf.to_yaml(cfg.module))
    #
    #         if OmegaConf.to_yaml(cfg.dataloader, resolve=True) != OmegaConf.to_yaml(exp_cfg.dataloader):
    #             print('Dataloader config mismatch. Exiting.')
    #             print('Old config', OmegaConf.to_yaml(exp_cfg.dataloader))
    #             print('New config', OmegaConf.to_yaml(cfg.dataloader))
    #             return
    #
    #     # trying to find checkpoints
    #     ckpt_dir = Path(cfg.exp_dir).joinpath('checkpoints')
    #     if ckpt_dir.exists():
    #         ckpts = list(sorted(ckpt_dir.iterdir(), key=os.path.getmtime))
    #         if len(ckpts):
    #             print('Found checkpoints', ckpts)
    #             ckpt_path = ckpts[-1]
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    # if cfg.log and (local_rank==0):
    if cfg.log and (local_rank==-1):
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

        # os.environ['WANDB_DISABLE_SERVICE'] = 'True'
        print('wandb mode', cfg.cluster.wandb_mode)
        # print('wandb service', os.environ.get('WANDB_DISABLE_SERVICE', 'variable unset'))
        run_id = cfg.name + '-' + get_random_code()

        # logger = pl.loggers.WandbLogger(project=cfg.project,
        #                                 name=cfg.name,
        #                                 id=run_id,
        #                                 save_dir=cfg.cluster.wandb_dir,
        #                                 log_model=False,
        #                                 # offline=(cfg.cluster.wandb_mode != 'online'))
        #                                 offline=False)


        # Initialize wandb directly
        wandb_cfg = sanitize_cfg(cfg)

        wandb_run = wandb.init(
            project=cfg.project,
            name=cfg.name,
            id=run_id,
            # dir=cfg.cluster.wandb_dir,
            dir='/home/zi/research_project/ArchesWeather/wandb_output/',
            config=wandb_cfg,
            mode="offline" if cfg.cluster.wandb_mode != "online" else "online"
        )
        # wandb_cfg = {key: value for key, value in cfg.items()}
        # Log config to wandb
        # wandb_cfg = sanitize_cfg(cfg)
        # wandb_run.config.update(wandb_cfg)


        # Pass the wandb_run to WandbLogger
        logger = pl.loggers.WandbLogger(
            experiment=wandb_run,  # Use the manually initialized wandb_run
            save_dir=cfg.cluster.wandb_dir
        )

        print("Configuration logged to Wandb successfully.")



    # if cfg.log and main_node and not Path(cfg.exp_dir).exists():
    #     print('registering exp on main node')
    #     hparams = OmegaConf.to_container(cfg, resolve=True)
    #     print(hparams)
    #     logger.log_hyperparams(hparams)
    #     Path(cfg.exp_dir).mkdir(parents=True)
    #     with open(Path(cfg.exp_dir) / 'config.yaml', 'w') as f:
    #         f.write(OmegaConf.to_yaml(cfg, resolve=True))

    # valset = instantiate(cfg.dataloader.dataset, domain='val')
    valset = instantiate(cfg.dataloader.dataset, domain='test')
    val_loader = torch.utils.data.DataLoader(valset,
                                             batch_size=cfg.batch_size,
                                             num_workers=cfg.cluster.cpus,
                                             shuffle=True)  # to viz shuffle samples

    trainset = instantiate(cfg.dataloader.dataset, domain='train')

    if cfg.module.use_infobatch:
        trainset = InfoBatch(dataset=trainset, num_epochs=cfg.module.info_batch.info_batch_num_epoch,
                             prune_ratio=cfg.module.info_batch.info_batch_ratio,
                             delta=cfg.module.info_batch.info_batch_delta,
                             prune_easy=cfg.module.info_batch.prune_easy,)
                             # saved_path=cfg.module.path_save_base)

        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=cfg.batch_size,
                                                   num_workers=cfg.cluster.cpus,
                                                   pin_memory=True,
                                                   prefetch_factor=4,
                                                   persistent_workers=True,
                                                   sampler=trainset.sampler)
    else:
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=cfg.batch_size,
                                                   num_workers=cfg.cluster.cpus,
                                                   pin_memory=True,
                                                   prefetch_factor=4,
                                                   persistent_workers=True,
                                                   shuffle=True)

    backbone = instantiate(cfg.module.backbone)

    '''
    first 4 columns are for surface variables
    the next 13*6 columns are for upper-air variables
    the same variable first,
    variable1_level1, variable1_level2, ..., variable1_leve13,
    variable2_level1, variable2_level2, ..., variable6_level13
    the one before the last columns is global training step
    the last columns is for total weighted loss
    '''
    if cfg.module.save_per_sample_loss:
        if cfg.module.use_infobatch:
            '''
            first 4 columns are for surface variables
            the next 13*6 columns are for upper-air variables
            the same variable first,

            variable1_level1, variable1_level2, ..., variable1_leve13,
            variable2_level1, variable2_level2, ..., variable6_level13
            the -4 position saved the ID
            the -3 position saved the occurrence times
            the one before the last columns is global training step
            the last columns is for total weighted loss
            '''
            samples_loss_np = np.zeros((int(58440 * cfg.module.accumulate_grad_batches),
                                        4 + 13 * 6 + 1 + 1 + 1 + 1))
        else:
            '''
            first 4 columns are for surface variables
            the next 13*6 columns are for upper-air variables
            the same variable first,
            variable1_level1, variable1_level2, ..., variable1_leve13,
            variable2_level1, variable2_level2, ..., variable6_level13
            the one before the last columns is global training step
            the last columns is for total weighted loss
            '''
            # samples_loss_np = np.zeros((58440, 4 + 13 * 6 + 1 + 1))
            # new version keep the same size as infobatch
            samples_loss_np = np.zeros((int(58440 * cfg.module.accumulate_grad_batches),
                                        4 + 13 * 6 + 1 + 1 + 1 + 1))
    else:
        # samples_loss_np = np.zeros((58440, 4 + 13 * 6 + 1 + 1))
        samples_loss_np = np.zeros((int(58440 * cfg.module.accumulate_grad_batches),
                                    4 + 13 * 6 + 1 + 1 + 1 + 1))
        # samples_loss_np = 0
    pl_module = instantiate(cfg.module.module,
                            backbone=backbone,
                            dataset=trainset,
                            samples_loss_np=samples_loss_np,
                            save_per_sample_loss=cfg.module.save_per_sample_loss,
                            path_save_base=cfg.module.path_save_base,
                            use_info_batch=cfg.module.use_infobatch,
                            )

    if hasattr(cfg, 'load_ckpt'):
        # load weights w/o resuming run
        pl_module.init_from_ckpt(cfg.load_ckpt)

    checkpointer = CheckpointEveryNSteps(dirpath=cfg.exp_dir,
                                         save_step_frequency=cfg.save_step_frequency,
                                         path_save_base=cfg.module.path_save_base, )

    # print('Manual submitit Requeuing')

    # def handler(*args, **kwargs):
    #     print('GCO: SIGTERM signal received. Requeueing job on main node.')
    #     if main_node:
    #         checkpointer.save()
    #         from submit import main as submit_main
    #         if cfg.cluster.manual_requeue:
    #             submit_main(cfg)
    #     exit()
    #
    # signal.signal(signal.SIGTERM, signal.SIG_DFL)
    # signal.signal(signal.SIGTERM, handler)

    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(cfg.module.seed)
    print('seed is set to {}'.format(cfg.module.seed))
    print('gradient_clip_val:', cfg.module.gradient_clip_val)
    print('accumulate_grad_batches:', cfg.module.accumulate_grad_batches)
    # print('accumulate_grad_batches:', cfg.module.accumulate_grad_batches)

    print('use_info_batch: ', cfg.module.use_infobatch)
    # if cfg.module.use_infobatch:
    #     print('Prune easy samples: ', cfg.module.info_batch.prune_easy)
    #     print('info_batch_num_epoch', cfg.module.info_batch.info_batch_num_epoch)
    #     print('info_batch_ratio', cfg.module.info_batch.info_batch_ratio)
    #     print('info_batch_delta', cfg.module.info_batch.info_batch_delta)
    # print('\n')
    print('num_warmup_steps: ', cfg.module.module.num_warmup_steps)
    print('num_training_steps: ', cfg.module.module.num_training_steps)
    
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        # strategy="auto",
        # strategy="ddp_find_unused_parameters_true",
        strategy="ddp",
        precision=cfg.cluster.precision,
        log_every_n_steps=cfg.log_freq,
        profiler=getattr(cfg, 'profiler', None),
        gradient_clip_val=cfg.module.gradient_clip_val,
        max_steps=cfg.module.module.num_training_steps,
        callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=50),
                   checkpointer],
        logger=logger,
        plugins=[],
        # test_only
        # limit_val_batches=cfg.limit_val_batches, # max 5 samples
        # limit_val_batches=0.03,  # no validation
        # num_sanity_val_steps=0,
        # use_distributed_sampler=False,
        # limit_train_batches=0.25,
        # test_only
        accumulate_grad_batches=cfg.module.accumulate_grad_batches,
    )

    # if cfg.debug:
    #     breakpoint()

    trainer.fit(pl_module, train_loader, val_loader,
                ckpt_path=ckpt_path)

    dist.destroy_process_group()
    del trainer



if __name__ == '__main__':
    main()
