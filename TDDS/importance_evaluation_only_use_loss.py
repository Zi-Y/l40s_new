import argparse
import numpy as np
from scipy.special import softmax
from numpy import linalg as LA
import torch
import os

########################################################################################################################
#  Calculate Importance
########################################################################################################################

# Define and parse command line arguments
parser = argparse.ArgumentParser(
    description='Calculate sample-wise importance',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--dynamics_path', type=str, default='./checkpoint/all-dataset/npy/',
                    help='Folder to save dynamics.')
parser.add_argument('--mask_path', type=str, default='./checkpoint/generated_mask/',
                    help='Folder to save mask.')
parser.add_argument('--trajectory_len', default=30, type=int,
                    help='Length of the trajectory.')
parser.add_argument('--window_size', default=30, type=int,
                    help='Size of the sliding window.')
parser.add_argument('--decay', default=0.9, type=float,
                    help='Decay factor for moving average.')
# 新增：选择计算样本重要性的方式：KL散度('kl')或直接使用Loss差值('loss')
parser.add_argument('--importance_metric', type=str, default='loss', choices=['kl', 'loss'],
                    help='Choose "kl" or "loss" to compute sample importance.')

args = parser.parse_args()


def generate(probs, losses, indexes):
    """
    probs  : shape = [trajectory_len, batch_size, num_classes]
    losses : shape = [trajectory_len, batch_size]
    indexes: shape = [trajectory_len, batch_size]
    """
    k = 0
    window_size = args.window_size
    moving_averages = []

    # 在这里根据 --importance_metric 进行分支
    if args.importance_metric == 'kl':
        # --------------------#
        #  使用 KL 散度方式  #
        # --------------------#
        while k < args.trajectory_len - window_size + 1:
            # 取出该滑窗对应的 [probs, indexes]
            probs_window = probs[k: k + window_size, :]
            indexes_window = indexes[k: k + window_size, :]

            # 对每个时间步做 softmax
            probs_window_softmax = softmax(probs_window, axis=2)  # shape=[window_size, batch_size, num_classes]

            # 根据索引重排概率分布
            probs_window_rere = []
            for i in range(window_size):
                # 先建一个与单步概率同维度的零张量
                probs_window_re = torch.zeros_like(torch.tensor(probs_window_softmax[0, :, :]))
                # 将对应下标位置的概率分布进行 index_add
                probs_window_re = probs_window_re.index_add(
                    0,
                    torch.tensor(indexes_window[i], dtype=int),
                    torch.tensor(probs_window_softmax[i, :])
                )
                probs_window_rere.append(probs_window_re)

            # 计算该滑窗内两两相邻时刻的 KL 散度
            probs_window_kd = []
            for j in range(window_size - 1):
                # log p_{t+1} - log p_t
                log_ratio = torch.log(probs_window_rere[j + 1] + 1e-8) - torch.log(probs_window_rere[j] + 1e-8)
                kd = torch.abs(probs_window_rere[j + 1] * log_ratio).sum(axis=1)  # shape=[batch_size]
                probs_window_kd.append(kd)
            probs_window_kd = np.array([arr.numpy() if torch.is_tensor(arr) else arr for arr in probs_window_kd])
            # 维度变为 [window_size-1, batch_size]

            # 计算滑窗内所有差异的平均值
            window_average = probs_window_kd.sum(0) / (window_size - 1)

            # 将每个时刻的差值与平均值的偏差再做范数度量
            window_diffdiff = []
            for ii in range(window_size - 1):
                window_diffdiff.append((probs_window_kd[ii] - window_average))
            window_diffdiff_norm = LA.norm(np.array(window_diffdiff), axis=0)  # shape=[batch_size]

            # 叠加到 moving_averages（带一个随时间衰减的权重）
            moving_averages.append(
                window_diffdiff_norm * args.decay * (1 - args.decay) ** (args.trajectory_len - window_size - k)
            )
            k += 1
            print(str(k) + ' window (KL) ok!')

    else:
        # --------------------#
        #  使用 Loss 方式    #
        # --------------------#
        while k < args.trajectory_len - window_size + 1:
            losses_window = losses[k: k + window_size, :]  # shape=[window_size, batch_size]
            indexes_window = indexes[k: k + window_size, :]  # shape=[window_size, batch_size]

            # 根据索引重排 Loss，方法与概率分布的重排类似
            losses_window_rere = []
            for i in range(window_size):
                # 建立一个与单步 loss 同维度的零张量
                losses_window_re = torch.zeros_like(torch.tensor(losses_window[0]))
                # 将 losses_window[i] 按 indexes_window[i] 的索引位置进行放置
                losses_window_re = losses_window_re.index_add(
                    0,
                    torch.tensor(indexes_window[i], dtype=int),
                    torch.tensor(losses_window[i])
                )
                losses_window_rere.append(losses_window_re)

            # losses_window_rere_np = np.array(losses_window_rere).T
            # new_column = np.arange(losses_window_rere_np.shape[0]).reshape(-1, 1)
            # new_arr = np.hstack((new_column, losses_window_rere_np))
            # np.save('Cifar100_loss_np_epoch30.npy', new_arr)

            # 计算该滑窗内两两相邻时刻的 loss 差异
            losses_window_diff = []
            for j in range(window_size - 1):
                diff = torch.abs(losses_window_rere[j + 1] - losses_window_rere[j])
                losses_window_diff.append(diff.numpy())
            losses_window_diff = np.array(losses_window_diff)  # shape=[window_size-1, batch_size]

            # 计算滑窗内所有差异的平均值
            window_average = losses_window_diff.sum(0) / (window_size - 1)

            # 每个时刻与平均值之间的差距，再做范数
            window_diffdiff = []
            for ii in range(window_size - 1):
                window_diffdiff.append(losses_window_diff[ii] - window_average)
            window_diffdiff_norm = LA.norm(np.array(window_diffdiff), axis=0)  # shape=[batch_size]

            # 叠加到 moving_averages（带一个随时间衰减的权重）
            moving_averages.append(
                window_diffdiff_norm * args.decay * (1 - args.decay) ** (args.trajectory_len - window_size - k)
            )
            k += 1
            print(str(k) + ' window (Loss) ok!')

    # 将每个滑窗的结果累加后进行排序
    moving_averages_sum = np.squeeze(sum(np.array(moving_averages), 0))  # shape=[batch_size]
    data_mask = moving_averages_sum.argsort()  # 从小到大排序对应的索引
    moving_averages_sum_sort = np.sort(moving_averages_sum)

    # Save the generated mask and scores
    if not os.path.exists(args.mask_path):
        os.makedirs(args.mask_path)
    # np.save(args.mask_path + f'data_loss_mask_{args.importance_metric}_win{args.window_size}_ep{args.trajectory_len}.npy',
    #         data_mask)
    # np.save(args.mask_path + f'score_loss_{args.importance_metric}_win{args.window_size}_ep{args.trajectory_len}.npy',
    #         moving_averages_sum_sort)


if __name__ == '__main__':
    # Load sample probabilities, losses, and indexes
    probs_list = []
    losses_list = []
    indexes_list = []
    for i in range(args.trajectory_len):
        probs_list.append(np.load(args.dynamics_path + f'{i}_Output.npy'))
        losses_list.append(np.load(args.dynamics_path + f'{i}_Loss.npy'))
        indexes_list.append(np.load(args.dynamics_path + f'{i}_Index.npy'))

    probs = np.array(probs_list)  # shape=[trajectory_len, batch_size, num_classes]
    losses = np.array(losses_list)  # shape=[trajectory_len, batch_size]
    indexes = np.array(indexes_list)  # shape=[trajectory_len, batch_size]

    generate(probs=probs, losses=losses, indexes=indexes)