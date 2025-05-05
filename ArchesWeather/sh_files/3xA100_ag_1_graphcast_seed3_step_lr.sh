#! /usr/bin/bash
python train_hydra.py cluster=example-slurm module=3xA100_ag_1_graphcast_seed3_step_lr dataloader=era5-w