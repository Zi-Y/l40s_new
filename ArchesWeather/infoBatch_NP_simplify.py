import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats
from collections import Counter
import torch
import itertools



variables = dict(
    state_level=['geopotential', 'u_component_of_wind', 'v_component_of_wind',
                 'temperature', 'specific_humidity', 'vertical_velocity'],
    state_surface=['10m_u_component_of_wind', '10m_v_component_of_wind',
                   '2m_temperature', 'mean_sea_level_pressure'])

pressure_levels = np.array([50,  100,  150,  200,  250,  300,
                            400,  500,  600,  700,  850,  925, 1000])
level_coeffs = (pressure_levels/pressure_levels.mean())
surface_coeffs = np.array([0.1, 0.1, 1.0, 0.1])
label_list = []
for variable in variables['state_surface']:
    label_list.append(variable)

for variable in variables['state_level']:
    for pressure in pressure_levels:
        label_list.append(variable + '_level_'+str(pressure))

label_list.append('iteration')
label_list.append('total_loss')


paths = ['/hpi/fs00/share/ekapex/zi/4xA100_ag_1_graphcast_seed3_infobatch_prune_easy_right_raALL0_lr4/infobatch_loss_values',
         '/hpi/fs00/share/ekapex/zi/4xV100_ag_1_graphcast_seed3_infobatch_prune_easy_right_raALL0_lr35/infobatch_loss_values',

         '/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3_infobatch_prune_easy_right_raALL0_s24/infobatch_loss_values',
         '/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3_infobatch_prune_easy_right_raALL0_s26/infobatch_loss_values',

         '/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3_infobatch_prune_easy_s100_r10_static/infobatch_loss_values',
         '/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3_infobatch_prune_hard_s100_r10_static/infobatch_loss_values',

         '/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3_infobatch_prune_easy_s100_r70_static/infobatch_loss_values',
         '/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3_infobatch_prune_easy_s100_r50_static/infobatch_loss_values',

         '/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3_S2L_k350_r30_static/infobatch_loss_values',
         '/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3_S2L_k350_r50_static/infobatch_loss_values',

         '/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3_S2L_k350_r70_static/infobatch_loss_values',
         '/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3_infobatch_prune_hard_s100_r50_static/infobatch_loss_values',

         '/hpi/fs00/share/ekapex/zi/4xV100_ag_1_graphcast_seed3_infobatch_prune_easy_s100_r30_static_new/infobatch_loss_values',
         '/hpi/fs00/share/ekapex/zi/4xV100_ag_1_graphcast_seed3_infobatch_prune_hard_s100_r30_static/infobatch_loss_values',

         '/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3_infobatch_prune_hard_s100_r30_static/infobatch_loss_values',
         '/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3_infobatch_prune_easy_s100_r30_static/infobatch_loss_values',

         # '/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3_infobatch_prune_easy_s100_r30_static/infobatch_loss_values',
         # '/hpi/fs00/share/ekapex/zi/4xV100_ag_1_graphcast_seed3_infobatch_prune_easy_s100_r10_static/infobatch_loss_values',
         # '/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3_infobatch_prune_hard_s100_r30_static/infobatch_loss_values',
         # '/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3_infobatch_prune_hard_s100_r50_static/infobatch_loss_values',
         # '/hpi/fs00/share/ekapex/zi/4xV100_ag_1_graphcast_seed3_infobatch_prune_hard_s100_r30_static/infobatch_loss_values',
         # '/hpi/fs00/share/ekapex/zi/4xV100_ag_1_graphcast_seed3_infobatch_prune_easy_s100_r30_static/infobatch_loss_values',
         # '/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3_infobatch_prune_easy_s100_r50_static/infobatch_loss_values',
         # '/hpi/fs00/share/ekapex/zi/4xV100_ag_1_graphcast_seed3_infobatch_prune_hard_s100_r10_static/infobatch_loss_values',
         # '/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3_infobatch_prune_easy_right_raALL0_s26/infobatch_loss_values',
         # '/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3_infobatch_prune_easy_right_raALL0_s24/infobatch_loss_values',
         # '/hpi/fs00/share/ekapex/zi/4xV100_ag_1_graphcast_seed3_infobatch_prune_easy_right_raALL0_lr35/infobatch_loss_values',
         # '/hpi/fs00/share/ekapex/zi/4xA100_ag_1_graphcast_seed3_infobatch_prune_easy_right_raALL0_lr4/infobatch_loss_values',
         # '/hpi/fs00/share/ekapex/zi/4xV100_ag_1_graphcast_seed3_infobatch_prune_easy_right_raALL0_lr43_error/infobatch_loss_values',
         # '/hpi/fs00/share/ekapex/zi/4xA100_ag_1_graphcast_seed3_infobatch_prune_easy_right_raALL0_lr43_error/infobatch_loss_values',
         # '/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3_infobatch_prune_easy_right_raALL0_lr43_error/infobatch_loss_values',
         # '/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3_infobatch_prune_easy_right_raALL0/infobatch_loss_values',
         # '/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3_infobatch_prune_easy_right_ra0/infobatch_loss_values',
         # '/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3_infobatch_prune_hard_right_ra0/infobatch_loss_values',
         ]


for path in paths:
    if '3xA100' in path:
        num_device = 3
    elif '4xV100' in path:
        num_device = 4
    elif '4xA100' in path:
        num_device = 4
    else:
        print('error')
        print('error')
        print('error')
        print('error')
        print('error')

    total_num_epochs = len(os.listdir(path))//num_device

    # 最后的一个epoch没有遍历所有元素，因此不计入统计

    scores_np = np.zeros(58440)

    total_loss_np = 0

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

    remained_ids_list = []
    trained_ids_list = []
    iterations_list = []
    trained_nums_list = []

    if ('ra0' in path) or ('raALL0' in path):
        prune_ratio = 1.0
    else:
        prune_ratio = 0.5

    loss_epoch_np = 0
    for select_epoch in range(0, total_num_epochs):
        for epoch_index, epoch in enumerate(range(select_epoch, select_epoch+1)):
            # loss_epoch_np = 0
            loss_epoch_np_cache = 0

            for device_id in range(num_device):
                if os.path.getsize(
                        path + "/Epoch_" + str(epoch) + "_device_" + str(device_id) + ".npy") < 36 * 1024 * 1024:
                    continue
                if (device_id == 0) and (epoch_index == 0):
                    loss_epoch_np = np.load(path + "/Epoch_" + str(epoch) + "_device_" + str(device_id) + ".npy")
                    # Find the rows in matrix1 that are all zeros
                    zero_rows = np.all(loss_epoch_np == 0, axis=1)
                    loss_epoch_np = loss_epoch_np[~zero_rows]
                    np.save(path + "/Epoch_" + str(epoch) + "_device_" + str(device_id) + ".npy", loss_epoch_np)

                    epoch_column = np.full((loss_epoch_np.shape[0], 1), epoch)
                    loss_epoch_np = np.hstack((loss_epoch_np, epoch_column))

                else:
                    loss_epoch_np_cache = np.load(path + "/Epoch_" + str(epoch) + "_device_" + str(device_id) + ".npy")
                    zero_rows = np.all(loss_epoch_np_cache == 0, axis=1)
                    loss_epoch_np_cache = loss_epoch_np_cache[~zero_rows]
                    np.save(path + "/Epoch_" + str(epoch) + "_device_" + str(device_id) + ".npy", loss_epoch_np_cache)

                    epoch_column = np.full((loss_epoch_np_cache.shape[0], 1), epoch)
                    loss_epoch_np_cache = np.hstack((loss_epoch_np_cache, epoch_column))

                    loss_epoch_np = np.vstack((loss_epoch_np, loss_epoch_np_cache))
            print('finished epoch ' + str(select_epoch))