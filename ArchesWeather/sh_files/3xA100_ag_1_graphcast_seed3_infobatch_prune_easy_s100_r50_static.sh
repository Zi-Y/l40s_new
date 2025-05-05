#! /usr/bin/bash
python train_hydra.py cluster=example-slurm module=3xA100_ag_1_graphcast_seed3_infobatch_prune_easy_s100_r50_static dataloader=era5-w