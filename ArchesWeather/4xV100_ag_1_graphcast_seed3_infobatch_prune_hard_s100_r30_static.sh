#! /usr/bin/bash
python train_hydra.py cluster=example-slurm module=4xV100_ag_1_graphcast_seed3_infobatch_prune_hard_s100_r30_static dataloader=era5-w