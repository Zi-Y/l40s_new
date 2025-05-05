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
from pathlib import Path
from matplotlib.animation import FuncAnimation
from scipy.stats import kstest, norm, laplace, cauchy, expon
from scipy.optimize import minimize
import time

warnings.filterwarnings('ignore')

try:
    from pytorch_lightning.callbacks import Callback, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
except:
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

# device = 'cpu'
device_ID = 3
device = 'cuda:' + str(device_ID) if torch.cuda.is_available() else 'cpu'

debug = False
print('Device', device)

lat_coeffs_equi = torch.tensor([torch.cos(x) for x in torch.arange(-torch.pi / 2, torch.pi / 2, torch.pi / 120)])
lat_coeffs_equi = (lat_coeffs_equi / lat_coeffs_equi.mean())[None, None, :, None]
coeffs = lat_coeffs_equi.to(device)

pressure_levels = torch.tensor([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925,
                                1000]).float()
level_coeffs = (pressure_levels / pressure_levels.mean())[None, None, :, None, None]
graphcast_surface_coeffs = torch.tensor([0.1, 0.1, 1.0, 0.1])[None, :, None, None, None]  # graphcast
pangu_surface_coeffs = torch.tensor([0.25, 0.25, 0.25, 0.25])[None, :, None, None, None]  # pangu coeffs

variables = dict(
    state_level=['geopotential', 'u_component_of_wind', 'v_component_of_wind',
                 'temperature', 'specific_humidity', 'vertical_velocity'],
    state_surface=['10m_u_component_of_wind', '10m_v_component_of_wind',
                   '2m_temperature', 'mean_sea_level_pressure'])

name_dict = ['U10', 'V10',
             'T2m', 'SP',
             'Z500', 'T850', 'Q700', 'U850', 'V850']


def loss(pred, batch, prefix='next_state', multistep=False, **kwargs):
    mse_surface = (pred[prefix + '_surface'] - batch[prefix + '_surface'])
    mse_surface = mse_surface.mul(coeffs)  # latitude coeffs

    surface_coeffs = graphcast_surface_coeffs
    mse_surface_w = mse_surface.mul(surface_coeffs.to(device))

    mse_level = (pred[prefix + '_level'] - batch[prefix + '_level'])
    mse_level = mse_level.mul(coeffs)

    mse_level_w = mse_level.mul(level_coeffs.to(device))

    nvar_level = mse_level_w.shape[-4]
    nvar_surface = surface_coeffs.sum().item()

    mse_surface_w_p2 = mse_surface_w.pow(2)
    mse_level_w_p2 = mse_level_w.pow(2)

    # coeffs are for number of variables
    loss_all = (4 * mse_surface_w_p2.mean() + nvar_level * mse_level_w_p2.mean()) / (nvar_level + nvar_surface)

    return mse_surface_w, mse_level_w, loss_all


def werror(x, y):
    # weighted root mean square error
    assert x.shape[-2] == 120, 'Wrong shape for WRMSE computation'
    # err = (x - y).pow(2).mul(coeffs).mean((-2, -1)).sqrt()
    err = (x - y).mul(coeffs)
    return err


# err.cpu().numpy()

def headline_werror(pred, batch, prefix=''):
    # x.shape should be (batch, leadtime, var, level, lat, lon)
    assert prefix + '_level' in batch, prefix + '_level not in batch'
    assert prefix + '_surface' in batch, prefix + '_surface not in batch'

    surface_werror = werror(pred[prefix + '_surface'], batch[prefix + '_surface'])
    level_werror = werror(pred[prefix + '_level'], batch[prefix + '_level'])

    # surface_werror = torch.squeeze(surface_werror)
    # level_werror = torch.squeeze(level_werror)

    # metrics = dict(
    #     T2m=surface_wrmse[..., 2, 0],
    #     SP=surface_wrmse[..., 3, 0],
    #     U10=surface_wrmse[..., 0, 0],
    #     V10=surface_wrmse[..., 1, 0],
    #     Z500=level_wrmse[..., 0, 7],
    #     T850=level_wrmse[..., 3, 10],
    #     Q700=1000 * level_wrmse[..., 4, 9],
    #     U850=level_wrmse[..., 1, 10],
    #     V850=level_wrmse[..., 2, 10])

    return surface_werror, level_werror


# state_surface=['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'mean_sea_level_pressure', vertical speed
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except:
        pass

    # first, check if exp exists
    print('Experiment already exists. Trying to resume it.')

    backbone = instantiate(cfg.module.backbone)

    samples_loss_np = 0
    pl_module = instantiate(cfg.module.module,
                            backbone=backbone,
                            # dataset=valset,
                            samples_loss_np=samples_loss_np,
                            save_per_sample_loss=cfg.module.save_per_sample_loss,
                            path_save_base=cfg.module.path_save_base,
                            )

    # folder_path = Path("/mnt/cache/data/zi/4xl40s_ag_1_graphcast_seed3")
    folder_path = Path("/mnt/ssd/zi/4xl40s_ag_1_graphcast_seed3")

    # 获取所有文件and f.endswith(".txt")
    file_paths = [file_path for file_path in folder_path.rglob("*") if
                  (file_path.is_file() and (file_path.suffix == ".ckpt"))]

    # sorted_file_paths = sorted(file_paths)

    sorted_file_paths = sorted(
        file_paths,
        key=lambda path: int(str(path).split("checkpoint_global_step=")[1].split(".ckpt")[0])
    )

    if not isinstance(sorted_file_paths, list):
        sorted_file_paths = [sorted_file_paths]

    sample_rank_list = np.load('/home/zi/research_project/ArchesWeather/'
                               '4xl40s_ag_1_seed3_mean_loss_default_320k.npy')
    sample_number = 2  # 100
    indices = np.linspace(0, len(sample_rank_list) - 1, sample_number, dtype=int)
    # sample_{ID} 中的值和 sample_rank_list 中的一样，均为sampler的ID，而非真实的sample_ID
    selected_samples = sample_rank_list[indices]

    selected_samples_list = np.array_split(selected_samples, 2)

    # selected_samples_per_gpu = selected_samples_list[device_ID]

    ds = instantiate(cfg.dataloader.dataset,
                     path='data/era5_240/full/',
                     # domain='test')
                     domain='train')

    # 创建 DataLoader，进一步优化
    test_loader = torch.utils.data.DataLoader(ds,
                                              batch_size=1,
                                              num_workers=8,
                                              shuffle=False,
                                              pin_memory=True,  # 加速数据从主机到 GPU 的传输
                                              persistent_workers=True,  # 保持 worker 进程活跃
                                              prefetch_factor=2)  # 每个 worker 提前加载的 batch 数量

    file_path = sorted_file_paths[0]

    pl_module.init_from_ckpt(file_path)

    pl_module.eval()
    pl_module.to(device)

    start_time = time.time()

    # saved_all_npy: sampler_Index, l1_mean, l2_mean,
    # mu, sigma, loc, scale, cauchy_loc, cauchy_scale, weighted mean loss,
    saved_all_npy = np.zeros((len(test_loader), 10))


    mid_point = int(0.5 * len(test_loader))  # 预先计算一半的长度
    for ID, batch in enumerate(test_loader):
        # 根据 device_ID 的范围划分
        if (device_ID == 0 and ID >= mid_point) or (device_ID == 1 and ID < mid_point):
            continue


        batch = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in batch.items()}

        with torch.no_grad():
            output = pl_module.forward(batch)
            # surface, atmos = headline_werror(output, batch, prefix='next_state')
            mse_surface_w, mse_level_w, loss_all = loss(output, batch, prefix='next_state')
            loss_all = loss_all.unsqueeze(0)

        # all_npy = np.concatenate((mse_surface_w.flatten().cpu().numpy(),
        #                           mse_level_w.flatten().cpu().numpy()), axis=0)
        # l1_mean = np.mean(np.abs(all_npy))
        # l2_mean = np.sqrt(np.mean(all_npy ** 2))
        # # 计算参数
        # mu, sigma = np.mean(all_npy), np.std(all_npy)  # 高斯分布参数
        # loc, scale = mu, np.mean(np.abs(all_npy - mu))  # 拉普拉斯分布参数
        # cauchy_loc, cauchy_scale = cauchy.fit(all_npy)  # Cauchy分布参数

        # 将两个张量展平并拼接
        all_tensor = torch.cat((mse_surface_w.flatten(), mse_level_w.flatten()), dim=0)

        # 计算 l1 和 l2 均值
        l1_mean = torch.mean(torch.abs(all_tensor))
        l2_mean = torch.sqrt(torch.mean(all_tensor ** 2))

        # 计算高斯分布参数
        mu = torch.mean(all_tensor)
        sigma = torch.std(all_tensor)  # 标准差

        # 计算拉普拉斯分布参数
        loc = mu
        scale = torch.mean(torch.abs(all_tensor - mu))

        # 计算 Cauchy 分布参数（仍需将数据转为 NumPy，因 scipy 不支持 PyTorch 张量）
        # cauchy_loc, cauchy_scale = cauchy.fit(all_tensor.cpu().numpy())
        cauchy_loc, cauchy_scale = 0, 0


        sampler_index = batch['id'].item() - 4
        saved_all_npy[sampler_index][0] = sampler_index
        saved_all_npy[sampler_index][1] = l1_mean.item()
        saved_all_npy[sampler_index][2] = l2_mean.item()
        saved_all_npy[sampler_index][3] = mu.item()
        saved_all_npy[sampler_index][4] = sigma.item()
        saved_all_npy[sampler_index][5] = loc.item()
        saved_all_npy[sampler_index][6] = scale.item()
        saved_all_npy[sampler_index][7] = cauchy_loc
        saved_all_npy[sampler_index][8] = cauchy_scale
        saved_all_npy[sampler_index][9] = loss_all.item()


        # 已经完成的步骤
        completed_steps = ID + 1

        # 已用时间
        elapsed_time = time.time() - start_time

        # 平均每步所需时间
        average_time_per_step = elapsed_time / completed_steps

        # 估计剩余时间
        remaining_steps = len(test_loader) - completed_steps
        estimated_remaining_time = remaining_steps * average_time_per_step

        # 打印进度和剩余时间
        if ID % 100 == 0:
            print('finished {}%,  {}/{}'.format(int(ID / (len(ds))), ID, len(ds)))
            print(f"Elapsed Time: {elapsed_time:.2f}s, "
                  f"Estimated Remaining Time: {estimated_remaining_time/60.0:.2f}mins")
            if ID % 1000 == 0:
                np.save(f'all_samples_4xl40s_ag_1_graphcast_seed3_20K_iteration_statistics_device{device_ID}.npy',
                        saved_all_npy)
                print(
                    f'save to: all_samples_4xl40s_ag_1_graphcast_seed3_20K_iteration_statistics_device{device_ID}.npy')

    np.save(f'all_samples_4xl40s_ag_1_graphcast_seed3_20K_iteration_statistics_device{device_ID}.npy', saved_all_npy)
    print(f'save to: all_samples_4xl40s_ag_1_graphcast_seed3_20K_iteration_statistics_device{device_ID}.npy')


# sample_{ID} 不是真实的sample ID，是sampler中的ID，也就是真实的sample_ID - 4
# sample_{ID} 不是真实的sample ID，是sampler中的ID，也就是真实的sample_ID - 4
# sample_{ID} 不是真实的sample ID，是sampler中的ID，也就是真实的sample_ID - 4


if __name__ == '__main__':
    main()