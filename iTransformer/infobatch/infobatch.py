import math
import torch
from torch.utils.data import Dataset
from operator import itemgetter
import warnings
import numpy as np

__all__ = ['InfoBatch']
#
def info_hack_indices(self):
    # 为了兼容 InfoBatch，我们在获取数据时记录当前 batch 的下标
    with torch.autograd.profiler.record_function(self._profile_name):
        if self._sampler_iter is None:
            self._reset()  # type: ignore[call-arg]
        if isinstance(self._dataset, InfoBatch):
            indices, data = self._next_data()
        else:
            data = self._next_data()
        self._num_yielded += 1
        # 略去 IterableDataset 长度不匹配的警告逻辑
        if isinstance(self._dataset, InfoBatch):
            self._dataset.set_active_indices(indices)
        return data
# def info_hack_indices(self):
#     with torch.autograd.profiler.record_function(self._profile_name):
#         if self._sampler_iter is None:
#             # TODO(https://github.com/pytorch/pytorch/issues/76750)
#             self._reset()  # type: ignore[call-arg]
#         if isinstance(self._dataset, InfoBatch):
#             indices, data = self._next_data()
#         else:
#             data = self._next_data()
#         self._num_yielded += 1
#         if self._dataset_kind == _DatasetKind.Iterable and \
#                 self._IterableDataset_len_called is not None and \
#                 self._num_yielded > self._IterableDataset_len_called:
#             warn_msg = ("Length of IterableDataset {} was reported to be {} (when accessing len(dataloader)), but {} "
#                         "samples have been fetched. ").format(self._dataset, self._IterableDataset_len_called,
#                                                                 self._num_yielded)
#             if self._num_workers > 0:
#                 warn_msg += ("For multiprocessing data-loading, this could be caused by not properly configuring the "
#                                 "IterableDataset replica at each worker. Please see "
#                                 "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples.")
#             warnings.warn(warn_msg)
#         if isinstance(self._dataset, InfoBatch):
#             self._dataset.set_active_indices(indices)
#         return data
# 将 _BaseDataLoaderIter 的 __next__ 方法打补丁（可保留）
from torch.utils.data.dataloader import _BaseDataLoaderIter
_BaseDataLoaderIter.__next__ = info_hack_indices


class InfoBatch(Dataset):
    """
    InfoBatch 利用样本 loss 的分布信息，在训练过程中随机剪枝一部分不“重要”的样本，
    并对保留样本的梯度进行 rescale，以近似原始梯度，达到加速训练的目的。
    参见: https://arxiv.org/pdf/2303.04947.pdf

    注意：假定原始 dataset 的大小是不变的。

    参数:
        dataset: 用于训练的原始数据集。
        num_epochs (int): 剪枝过程的训练周期数。
        prune_ratio (float, optional): 剪枝时要舍弃的样本比例。
        delta (float, optional): 剪枝过程持续的训练周期数比例，通常接近 1，默认 0.875。
        device (torch.device, optional): 存储变量的设备，默认为 CUDA。
    """
    def __init__(self, dataset: Dataset, num_epochs: int,
                 prune_ratio: float = 0.5, delta: float = 0.875, args=None,
                 device: torch.device = torch.device('cuda')):
        self.device = device
        self.dataset = dataset
        self.keep_ratio = min(1.0, max(1e-1, 1.0 - prune_ratio))
        self.num_epochs = num_epochs
        self.delta = delta
        self.args = args

        # scores 和 weights 均在 GPU 上初始化
        if self.args.pruning_method in {9, 10, 11, 12, 13, 14, 15, 16, 17}:
            self.per_sample_scores = torch.ones(len(self.dataset), device=self.device)
            self.scores = torch.ones((len(self.dataset), self.args.pred_len, self.args.enc_in), device=self.device) * 3
            # self.weights = torch.ones((len(self.dataset), self.args.pred_len, self.args.enc_in), device=self.device)
            if self.args.pred_len == 96:
                best_model_epoch = 5
            elif self.args.pred_len == 192:
                best_model_epoch = 4
            elif self.args.pred_len == 336:
                best_model_epoch = 2
            elif self.args.pred_len == 720:
                best_model_epoch = 7

            # because we save the loss values from forward pass, so I guess it's better to save best_model_epoch+1
            best_model_epoch = best_model_epoch+1


            # if self.args.pred_len == 96:
            #     file_path = ("/mnt/ssd/zi/itransformer_results/"
            #                  "seed0_pm0_pr0_low10_high10_start0_int20_tr50_test101_"
            #                  "iTransformer_custom_ftM_sl96_ll48_pl96_dm512_nh8_el3_"
            #                  "dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0/iter_6830_"
            #                  "train_set_all_sample_all_tokens.npy")
            #     self.trained_per_token_loss = torch.tensor(np.load(file_path),
            #                                                dtype=torch.float32, device=device)

            # else:
            #     file_path = (
            #         f"/home/local/zi/research_project/iTransformer/all_saved_seed0_trained_per_token_loss/"
            #         f"all_saved_seed0_pm0_pr0_low10_high10_start0_int20_test1_"
            #         f"iTransformer_custom_ftM_sl96_ll48_pl{self.args.pred_len}_"
            #         f"dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0/"
            #         f"loss_all_epoch_all_variable_all_stamp0.npy")
            #
            #     self.trained_per_token_loss = torch.tensor(np.load(file_path)[:,best_model_epoch,:,:], dtype=torch.float32, device=device)

            file_path = (
                f"/home/local/zi/research_project/iTransformer/all_saved_seed0_trained_per_token_loss/"
                f"all_saved_seed0_pm0_pr0_low10_high10_start0_int20_test1_"
                f"iTransformer_custom_ftM_sl96_ll48_pl{self.args.pred_len}_"
                f"dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0/"
                f"loss_all_epoch_all_variable_all_stamp0.npy")

            self.trained_per_token_loss = torch.tensor(np.load(file_path)[:,best_model_epoch,:,:], dtype=torch.float32, device=device)

            if self.args.pruning_method == 10:
                self.avg_token_pruning_rate = 0.0
            elif self.args.pruning_method == 11:
                # self.args.infobatch_remove_low means token level pruning rate only for method 11
                self.avg_token_pruning_rate = self.args.infobatch_remove_low
            elif self.args.pruning_method == 12:
                self.global_token_mask = torch.ones((len(self.dataset),
                                                     self.args.pred_len,
                                                     self.args.enc_in),
                                                    device=self.device)
        else:
            self.scores = torch.ones(len(self.dataset), device=self.device) * 3
            # self.weights = torch.ones(len(self.dataset), device=self.device)

        self.weights = torch.ones(len(self.dataset), device=self.device)
        self.num_pruned_samples = 0
        self.cur_batch_index = None

    # def __getattr__(self, name):
    #     如果本类中没有找到对应属性，则交由原始 dataset 处理
        # return getattr(self.dataset, name)

    def __getattr__(self, name):
        try:
            # 先尝试从 InfoBatch 本身获取属性
            return object.__getattribute__(self, name)
        except AttributeError:
            # 如果不存在，则委托给内部 dataset
            return getattr(self.dataset, name)

    def set_active_indices(self, cur_batch_indices):
        # 确保当前 batch 的下标存储在 GPU 上
        if not torch.is_tensor(cur_batch_indices):
            self.cur_batch_index = torch.tensor(cur_batch_indices, device=self.device)
        else:
            self.cur_batch_index = cur_batch_indices.to(self.device)

    def update(self, values, only_update_saved_loss_metric=False):
        assert torch.is_tensor(values)
        batch_size = values.shape[0]
        assert self.cur_batch_index is not None and self.cur_batch_index.numel() == batch_size, 'not enough index'
        # 直接使用 GPU 上的 weights，无需转换
        indices = self.cur_batch_index.long()

        if (self.args.pruning_method == 9 or self.args.pruning_method == 10 or
                self.args.pruning_method == 11):
            if self.args.token_pr_rate >= 0.0:
                diff_trained_current = values - self.trained_per_token_loss[indices]
            else:
                diff_trained_current = self.trained_per_token_loss[indices] - values

            if self.args.pruning_method == 9:
                threshold = torch.quantile(diff_trained_current, abs(self.args.token_pr_rate))
                weights = (diff_trained_current >= threshold).float()

            elif self.args.pruning_method == 10 or self.args.pruning_method == 11:
                # 尽可能让 tensor 连续，避免额外开销
                flat_diff = diff_trained_current.contiguous().reshape(-1)

                # n = flat_diff.numel()

                # 如果数据量很大，则对其随机采样（比如采样1000个元素），以近似计算分位数
                # sample_size = 1000
                # if n > sample_size:
                #     # 从 [0, n) 中随机采样 sample_size 个索引
                #     rand_idx = torch.randint(0, n, (sample_size,), device=flat_diff.device)
                #     sampled_diff = flat_diff[rand_idx]
                # else:
                #     sampled_diff = flat_diff

                # 计算近似分位数对应的 kth 索引
                k = max(1, int(abs(self.avg_token_pruning_rate) * flat_diff.numel()))
                threshold, _ = flat_diff.kthvalue(k)

                # 根据阈值生成权重
                weights = (diff_trained_current >= threshold).float()


                # threshold = torch.quantile(diff_trained_current, abs(self.avg_token_pruning_rate))

            # loss_val = values.detach().clone()

        elif self.args.pruning_method == 12:
            weights = self.global_token_mask[indices].float()

        elif self.args.pruning_method in (13, 17):
            if only_update_saved_loss_metric:
                self.trained_per_token_loss[indices] = values
            else:
                if self.args.token_pr_rate >= 0.0:
                # if self.args.token_pr_rate < 0.0:
                    diff_trained_current = self.trained_per_token_loss[indices] - values
                else:
                    diff_trained_current = values - self.trained_per_token_loss[indices]

                if self.args.pruning_method == 17:
                    diff_trained_current = diff_trained_current/self.trained_per_token_loss[indices]

                flat_diff = diff_trained_current.contiguous().reshape(-1)
                # 计算近似分位数对应的 kth 索引
                k = max(1, int(abs(self.args.token_pr_rate) * flat_diff.numel()))
                threshold, _ = flat_diff.kthvalue(k)

                # 根据阈值生成权重
                weights = (diff_trained_current >= threshold).float()


        elif self.args.pruning_method in (14, 15):
            if only_update_saved_loss_metric:
                self.trained_per_token_loss[indices] = values
            else:
                if self.args.token_pr_rate >= 0.0:
                    diff_trained_current = self.trained_per_token_loss[indices] - values
                else:
                    diff_trained_current = values - self.trained_per_token_loss[indices]

                if self.args.pruning_method == 15:
                    diff_trained_current = diff_trained_current/self.trained_per_token_loss[indices]

                flat_diff = diff_trained_current.contiguous().reshape(-1)
                # 计算近似分位数对应的 kth 索引
                k = max(1, int(abs(self.args.token_pr_rate) * flat_diff.numel()))
                threshold, _ = flat_diff.kthvalue(k)

                # 根据阈值生成权重
                weights = (diff_trained_current >= threshold).float()
                # 再次更新数值
                self.trained_per_token_loss[indices] = values

        elif self.args.pruning_method == 16:
            # uniform 分布 + 比阈值
            # 在 [0,1) 上均匀采样，rand < pr 时置 0，否则置 1
            weights = (torch.rand_like(values) >= self.args.token_pr_rate).float()

        else:
            # 直接使用 GPU 上的 weights，无需转换
            weights = self.weights[self.cur_batch_index.long()]
            # loss_val = values.detach().clone()


        self.cur_batch_index = None  # 重置
        if self.args.pruning_method != 16:
            self.scores[indices] = values  # 直接在 GPU 上更新
        if not only_update_saved_loss_metric:
            values.mul_(weights)

        return values.mean()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # 返回样本下标以及原始数据
        return index, self.dataset[index]

    def prune(self):
        # 基于 loss 分布剪枝：认为 loss 较低（已较好学习）的样本更“好”
        if (self.args.pruning_method == 4) or (self.args.pruning_method == 5):
            well_learned_mask = self.scores < self.scores.mean()
        # 计算 10% 和 90% 分位数
        elif (self.args.pruning_method == 6) or (self.args.pruning_method == 7):
            low_quantile = torch.quantile(self.scores, self.args.infobatch_remove_low)
            high_quantile = torch.quantile(self.scores, (1.0-self.args.infobatch_remove_high))
            # 生成 well_learned_mask：
            # 选取前 infobatch_remove_low 和后 infobatch_remove_high 的值
            # 也就是随机删除最简单和最困难的
            well_learned_mask = (self.scores <= low_quantile) | (self.scores >= high_quantile)
            if self.args.pruning_method == 7:
                # 生成 well_learned_mask
                # 选取前 infobatch_remove_low 直到后 infobatch_remove_high 的值
                # 也就是随机删除最中间的部分 保留最简单和最困难的
                well_learned_mask = ~well_learned_mask

        elif self.args.pruning_method == 8:
            low_quantile = torch.quantile(self.scores, self.args.infobatch_remove_start)
            if self.args.infobatch_remove_start + self.args.infobatch_remove_interval > 1.0:
                high_quantile = torch.quantile(self.scores,1.0)
            else:
                high_quantile = torch.quantile(self.scores,
                                               self.args.infobatch_remove_start + self.args.infobatch_remove_interval)
            well_learned_mask = (self.scores <= low_quantile) | (self.scores >= high_quantile)
            well_learned_mask = ~well_learned_mask

        elif self.args.pruning_method == 9:
            # well_learned_mask should be all false mask
            # because we do not want to remove any sample, but we want to remove tokens
            # well_learned_mask = self.per_sample_scores > 1000000.0
            well_learned_mask = torch.zeros_like(self.per_sample_scores, dtype=torch.bool)

        elif self.args.pruning_method == 10 or self.args.pruning_method == 11:

            # 1. 计算差值矩阵
            difference_all_token = self.scores - self.trained_per_token_loss

            # 2. 统计每个样本（第一维）中大于0的元素个数
            # 这里对第1和第2维求和（即每个 sample 内所有 token 的统计）
            positive_counts = (difference_all_token > 0).sum(dim=(1, 2))

            if self.args.token_pr_rate < 0.0:
                positive_counts = - positive_counts

            threshold = torch.quantile(positive_counts.float(), abs(self.args.token_pr_rate))

            well_learned_mask = (positive_counts <= threshold)

        elif self.args.pruning_method == 12:
            # well_learned_mask should be all false mask
            # because we do not want to remove any sample, but we want to remove tokens
            well_learned_mask = torch.zeros_like(self.per_sample_scores, dtype=torch.bool)


            if self.args.token_pr_rate >= 0.0:
                diff_trained_current = self.scores - self.trained_per_token_loss
            else:
                diff_trained_current = self.trained_per_token_loss - self.scores

            k = max(1, int(abs(self.args.token_pr_rate) * diff_trained_current.numel()))
            threshold, _ = diff_trained_current.view(-1).kthvalue(k)
            self.global_token_mask = (diff_trained_current >= threshold).float()


        elif self.args.pruning_method in {13, 14, 15, 16, 17}:
            # well_learned_mask should be all false mask
            # because we do not want to remove any sample, but we want to remove tokens
            well_learned_mask = torch.zeros_like(self.per_sample_scores, dtype=torch.bool)

        well_learned_indices = torch.nonzero(well_learned_mask, as_tuple=False).squeeze()
        remained_indices = torch.nonzero(~well_learned_mask, as_tuple=False).squeeze()
        num_well_learned = int(well_learned_mask.sum().item())
        num_remained = int((~well_learned_mask).sum().item())
        print(f'#well learned samples {num_well_learned}, #remained samples {num_remained}, len(dataset) = {len(self.dataset)}')
        if well_learned_indices.dim() == 0:  # 如果只有一个元素，则扩展维度
            well_learned_indices = well_learned_indices.unsqueeze(0)
        selected_count = int(self.keep_ratio * well_learned_indices.numel())
        if selected_count > 0:
            perm = torch.randperm(well_learned_indices.numel(), device=self.device)
            selected_indices = well_learned_indices[perm[:selected_count]]
            self.weights[selected_indices] = 1 / self.keep_ratio
            final_indices = torch.cat((remained_indices, selected_indices))
        else:
            final_indices = remained_indices
        self.num_pruned_samples += len(self.dataset) - final_indices.numel()
        # 打乱最终下标
        perm_final = torch.randperm(final_indices.numel(), device=self.device)
        final_indices = final_indices[perm_final]

        # 计算当前情况下, difference_all_token大于的比例 以便得出在每一个batch中token level的keep rate
        if self.args.pruning_method == 10:
            self.avg_token_pruning_rate = (difference_all_token < 0).float().mean().item()
            del difference_all_token
            print(f'method 10 token level pruning rate in a batch {self.avg_token_pruning_rate:.4f}')
        elif self.args.pruning_method == 11:
            print(f'method 11 token level pruning rate in a batch {self.avg_token_pruning_rate:.4f}')

        return final_indices

    def no_prune(self):
        indices = torch.randperm(len(self), device=self.device)
        return indices

    def mean_score(self):
        return self.scores.mean()

    def get_weights(self, indexes):
        return self.weights[indexes]

    def get_pruned_count(self):
        return self.num_pruned_samples

    @property
    def stop_prune(self):
        return self.num_epochs * self.delta

    @property
    def get_pruning_method(self):
        return self.args.pruning_method



    def reset_weights(self):
        self.weights.fill_(1)

    @property
    def sampler(self):
        # 单 GPU 训练下只需返回 IBSampler
        return IBSampler(self)


class IBSampler(object):
    def __init__(self, dataset: InfoBatch):
        self.dataset = dataset
        self.stop_prune = dataset.stop_prune
        self.get_pruning_method = dataset.get_pruning_method
        self.iterations = 0
        self.sample_indices = None
        self.reset()

    def reset(self):
        # 使用 torch 的随机种子，确保在 GPU 上进行随机操作
        torch.manual_seed(self.iterations)
        # print('Infobatch: reset(self): current iteration for pruning: ', self.iterations)

        # if self.get_pruning_method !=10 or (self.get_pruning_method ==10 and self.iterations > 1):
        if self.iterations > 1:

            if self.iterations > self.stop_prune:
                print(f'we are going to stop prune, #stop prune {self.stop_prune}, #cur iterations {self.iterations}')
                if self.iterations == self.stop_prune + 1:
                    self.dataset.reset_weights()
                self.sample_indices = self.dataset.no_prune()
            else:
                print(f'we are going to continue pruning, #stop prune {self.stop_prune}, #cur iterations {self.iterations}')
                self.sample_indices = self.dataset.prune()
        else:
            self.sample_indices = self.dataset.no_prune()

        self.iterations += 1



    def __iter__(self):
        self.reset()
        # DataLoader 需要 Python 的 int 索引，所以这里转换为 CPU int 后迭代
        return iter(self.sample_indices.cpu().tolist())

    def __len__(self):
        return self.sample_indices.numel()

    def __getitem__(self, idx):
        return self.sample_indices[idx]