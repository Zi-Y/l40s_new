#! /usr/bin/bash
python train_hydra.py cluster=example-slurm module=3xA100_ag_1_graphcast_seed3_S2L_k350_r70_static dataloader=era5-w