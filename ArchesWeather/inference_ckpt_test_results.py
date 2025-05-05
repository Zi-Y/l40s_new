import pytorch_lightning as pl
import torch
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from hydra.utils import instantiate
import os
from datetime import datetime
import time

from omegaconf import OmegaConf
import signal

import numpy as np
import warnings
import matplotlib.pyplot as plt
from evaluation.deterministic_metrics import headline_wrmse

warnings.filterwarnings('ignore')

try:
    from pytorch_lightning.callbacks import Callback, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
except:
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger


# device = 'cpu'
device_ID = 1
device = 'cuda:' + str(device_ID) if torch.cuda.is_available() else 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Device', device)



# device_ID = device_ID -4
# state_surface=['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'mean_sea_level_pressure', vertical speed
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except:
        pass

    # first, check if exp exists
    print('Experiment already exists. Trying to resume it.')

    ds = instantiate(cfg.dataloader.dataset,
                     path='data/era5_240/full/',
                     domain='test')

    backbone = instantiate(cfg.module.backbone)

    samples_loss_np = 0
    pl_module = instantiate(cfg.module.module,
                            backbone=backbone,
                            # dataset=valset,
                            samples_loss_np=samples_loss_np,
                            save_per_sample_loss=cfg.module.save_per_sample_loss,
                            path_save_base=cfg.module.path_save_base,
                            )



    load_single_model = False
    if load_single_model:

        # load weights w/o resuming run
        model_path = '/home/zi/research_project/ArchesWeather/modelstore/ArchesModel/'
        # model_name = f'3xA100_graphcast_seed{device_ID+1}_300000.ckpt'
        # model_name = f'3xA100_graphcast_seed6_300000.ckpt'
        # model_name = f'3xA100_seed1_infobatch_297000.ckpt'
        # model_name = '3xA100_seed3_infobatch_200000.ckpt'
        model_name = '3xA100_seed3_infobatch_prune_easy_280000.ckpt'

        # model_name = f'3xA100_seed2_infobatch_294000.ckpt'
        print('model path:', model_path + model_name)

        pl_module.init_from_ckpt(model_path + model_name)

        pl_module.eval()
        pl_module.to(device)


        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        metrics = 0
        for ID in range(len(ds)):
            batch = {k:(v[None].to(device) if hasattr(v, 'to') else [v]) for k, v in ds[ID].items()}
            with torch.no_grad():
                # start_event.record()
                output = pl_module.forward(batch)
                # end_event.record()

                # torch.cuda.synchronize()
                # inference_time = start_event.elapsed_time(end_event)
                # print(f'Inference Time: {inference_time} ms')

                # denormalize output
                denorm_pred = ds.denormalize(output, batch)
                denorm_batch = ds.denormalize(batch)

                # # get per-sample main metrics from WeatherBench
                if ID==0:
                    metrics = headline_wrmse(denorm_pred, denorm_batch, prefix='next_state')
                else:
                    metrics_cached = headline_wrmse(denorm_pred, denorm_batch, prefix='next_state')
                    for key in metrics:
                        metrics[key] += metrics_cached[key]

                if ID % 146.0 == 0:
                    print('finished {}%,  {}/{}'.format(int(100*ID / (len(ds))),
                                                       ID, len(ds)))

        metrics_mean = {key: value / len(ds) for key, value in metrics.items()}
        print(metrics_mean)

        save_order = ['Z500', 'T850', 'Q700', 'U850', 'V850', 'T2m', 'SP', 'U10', 'V10']

        # 保存到txt文件
        saved_path = '/hpi/fs00/home/zi.yang/research_project/ArchesWeather/results/'
        os.makedirs(saved_path, exist_ok=True)
        file_names = model_name.split('.')[0] + '.txt'
        with open(saved_path + file_names, 'w') as f:
            for key in save_order:
                value = metrics_mean[key].cpu().item()  # 从CUDA tensor转换为CPU tensor并获取数值
                f.write(f'{key}: {value}\n')


        print(f'results saved to {saved_path + file_names}')

    else:
        from pathlib import Path

        # 指定要遍历的文件夹路径
        # folder_path = Path("/hpi/fs00/share/ekapex/zi/3xA100_graphcast_seed3_infobatch")
        # folder_path = Path("/hpi/fs00/share/ekapex/zi/3xA100_graphcast_seed3_infobatch_prune_easy")
        # folder_path = Path("/hpi/fs00/share/ekapex/zi/4xA100_ag_1_graphcast_no_GC_seed3_infobatch_prune_easy")
        # folder_path = Path("/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3")
        # folder_path = Path("/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3_infobatch_prune_easy_right")
        # folder_path = Path("/hpi/fs00/share/ekapex/zi/4xV100_ag_1_graphcast_no_GC_seed3")
        # folder_path = Path("/hpi/fs00/share/ekapex/zi/4xV100_ag_1_graphcast_no_GC_seed3_infobatch_prune_easy")
        # folder_path = Path("/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3_infobatch_prune_hard_right")
        # folder_path = Path("/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3_infobatch_prune_easy_s100_r50_static")
        # folder_path = Path("/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3_infobatch_prune_easy_right_raALL0_s26")
        # folder_path = Path("/hpi/fs00/share/ekapex/zi/4xV100_ag_1_graphcast_seed3_infobatch_prune_easy_right_raALL0_lr35")
        # folder_path = Path("/hpi/fs00/share/ekapex/zi/4xA100_ag_1_graphcast_seed3_infobatch_prune_easy_right_raALL0_lr4")
        # folder_path = Path("/hpi/fs00/share/ekapex/zi/4xV100_ag_1_graphcast_seed2_new_setting_0AMfrom2007")
        folder_path = Path("/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed4_new_setting_0AMfrom2007")
        folder_path = Path("/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3")
        folder_path = Path("/hpi/fs00/share/ekapex/zi/3xA100_ag_1_graphcast_seed3_infobatch_prune_hard_s100_r50_static_step_lr")
        # 3xA100_ag_1_graphcast_seed3_infobatch_prune_hard_s100_r50_static_step_lr


        # 获取所有文件and f.endswith(".txt")
        file_paths = [file_path for file_path in folder_path.rglob("*") if (file_path.is_file() and (file_path.suffix==".ckpt"))]

        sorted_file_paths = sorted(file_paths)

        # sorted_file_paths = sorted_file_paths[3:6]
        # sorted_file_paths = sorted_file_paths[4]
        sorted_file_list = np.array_split(sorted_file_paths, 7)

        # 保持每个分组为 List[Path]
        sorted_file_list = [list(split) for split in sorted_file_list]


        # sorted_file_paths = sorted_file_list[device_ID]
        sorted_file_paths = sorted_file_list[7-device_ID]



        # sorted_file_paths = sorted_file_paths[::-1]

        # if device_ID == 1:
        #     sorted_file_paths = sorted_file_paths[:4]
        #     # 0 1 2 3
        # elif device_ID == 3:
        #     sorted_file_paths = sorted_file_paths[4:8]
        #     # 4 5 6 7
        # elif device_ID == 5:
        #     sorted_file_paths = sorted_file_paths[8:12]
        #     # 8 9 10 11
        # else:
        #     sorted_file_paths = sorted_file_paths[12:]
        #     # 12 13 14 15

        # if device_ID == 1:
        #     sorted_file_paths = sorted_file_paths[2:4]
        #     # 0 1 2 3
        # elif device_ID == 3:
        #     sorted_file_paths = sorted_file_paths[6:8]
        #     # 4 5 6 7
        # elif device_ID == 5:
        #     sorted_file_paths = sorted_file_paths[10:12]
        #     # 8 9 10 11
        # else:
        #     sorted_file_paths = sorted_file_paths[14:]
        #     # 12 13 14 15

        if not isinstance(sorted_file_paths, list):
            sorted_file_paths = [sorted_file_paths]
        # 遍历所有文件并打印路径
        for file_path in sorted_file_paths:
            # print(file_path)

            saved_name = file_path.parent.parent.name +'_'+ file_path.name.split("=")[1][:-5]
            print(saved_name)

            pl_module.init_from_ckpt(file_path)

            pl_module.eval()
            pl_module.to(device)


            metrics = 0
            for ID in range(len(ds)):

                # if 'new_setting' in str(file_path):
                # if '_' in str(file_path):
                #     if ID%4 != 0:
                #         # print(ID)
                #         continue

                batch = {k: (v[None].to(device) if hasattr(v, 'to') else [v]) for k, v in ds[ID].items()}
                with torch.no_grad():

                    output = pl_module.forward(batch)
                    denorm_pred = ds.denormalize(output, batch)
                    denorm_batch = ds.denormalize(batch)

                    # # get per-sample main metrics from WeatherBench
                    if ID == 0:
                        metrics = headline_wrmse(denorm_pred, denorm_batch, prefix='next_state')
                    else:
                        metrics_cached = headline_wrmse(denorm_pred, denorm_batch, prefix='next_state')
                        for key in metrics:
                            metrics[key] += metrics_cached[key]

                    # if ID % 146.0 == 0:
                    if ID % 73.0 == 0:
                        print('finished {}%,  {}/{}'.format(int(100 * ID / (len(ds))),
                                                            ID, len(ds)))

            metrics_mean = {key: value / len(ds) for key, value in metrics.items()}
            print(metrics_mean)

            save_order = ['Z500', 'T850', 'Q700', 'U850', 'V850', 'T2m', 'SP', 'U10', 'V10']

            # 保存到txt文件
            saved_path = '/hpi/fs00/home/zi.yang/research_project/ArchesWeather/results/'
            saved_path = saved_path + file_path.parent.parent.name + '/'
            saved_name = file_path.name.split("=")[1][:-5]
            os.makedirs(saved_path, exist_ok=True)
            file_names = file_path.parent.parent.name + '_'+saved_name + '.txt'
            with open(saved_path + file_names, 'w') as f:
                for key in save_order:
                    value = metrics_mean[key].cpu().item()  # 从CUDA tensor转换为CPU tensor并获取数值
                    f.write(f'{key}: {value}\n')

            print(f'results saved to {saved_path + file_names}')
            print('\n')
            print('\n')
            # os.makedirs(saved_path, exist_ok=True)
            # file_names = saved_name + '.txt'
            # with open(saved_path + file_names, 'w') as f:
            #     for key in save_order:
            #         value = metrics_mean[key].cpu().item()  # 从CUDA tensor转换为CPU tensor并获取数值
            #         f.write(f'{key}: {value}\n')
            #
            # print(f'results saved to {saved_path + file_names}')
            # print('\n')
            # print('\n')




    # read txt file
    # file_path = 'output.txt'
    # # 创建一个列表来存储读取的数值
    # values = []
    # # 读取txt文件中的值
    # with open(file_path, 'r') as f:
    #     for line in f:
    #         # 每行格式为 'key: value'，我们只提取value
    #         value = float(line.strip().split(': ')[1])
    #         values.append(value)
    #
    # # 将读取的值转换为NumPy数组
    # numpy_array = np.array(values)
    #
    # # 将数组转换为1行多列的矩阵（根据原始数据的长度调整列数）
    # numpy_matrix = numpy_array.reshape(1, -1)
    #
    # print("读取的1行多列的NumPy矩阵：")
    # print(numpy_matrix)

if __name__ == '__main__':
    main()
