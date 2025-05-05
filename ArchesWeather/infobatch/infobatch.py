import math
import numpy as np
from typing import Iterator, Optional
import torch
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch.utils.data import Dataset, _DatasetKind
from torch.utils.data.distributed import DistributedSampler
from operator import itemgetter
import torch.distributed as dist
import warnings

__all__ = ['InfoBatch']


def info_hack_indices(self):
    with torch.autograd.profiler.record_function(self._profile_name):
        if self._sampler_iter is None:
            # https://github.com/pytorch/pytorch/issues/76750
            self._reset()  # type: ignore[call-arg]
        if isinstance(self._dataset, InfoBatch):
            indices, data = self._next_data()
        else:
            data = self._next_data()
        self._num_yielded += 1
        if self._dataset_kind == _DatasetKind.Iterable and \
                self._IterableDataset_len_called is not None and \
                self._num_yielded > self._IterableDataset_len_called:
            warn_msg = ("Length of IterableDataset {} was reported to be {} (when accessing len(dataloader)), but {} "
                        "samples have been fetched. ").format(self._dataset, self._IterableDataset_len_called,
                                                              self._num_yielded)
            if self._num_workers > 0:
                warn_msg += ("For multiprocessing data-loading, this could be caused by not properly configuring the "
                             "IterableDataset replica at each worker. Please see "
                             "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples.")
            warnings.warn(warn_msg)
        if isinstance(self._dataset, InfoBatch):
            self._dataset.set_active_indices(indices)
        return data


_BaseDataLoaderIter.__next__ = info_hack_indices


@torch.no_grad()
def concat_all_gather(tensor, dim=0):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=dim)
    return output


class InfoBatch(Dataset):
    """
    InfoBatch aims to achieve lossless training speed up by randomly prunes a portion of less informative samples
    based on the loss distribution and rescales the gradients of the remaining samples to approximate the original
    gradient. See https://arxiv.org/pdf/2303.04947.pdf

    .. note::.
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for training.
        num_epochs (int): The number of epochs for pruning.
        prune_ratio (float, optional): The proportion of samples being pruned during training.
        delta (float, optional): The first delta * num_epochs the pruning process is conducted. It should be close to 1. Defaults to 0.875.
    """

    def __init__(self, dataset: Dataset, num_epochs: int,
                 prune_ratio: float = 0.5, delta: float = 0.875,
                 prune_easy: int = 0):
        self.dataset = dataset
        # self.keep_ratio = min(1.0, max(1e-1, 1.0 - prune_ratio))
        self.keep_ratio = prune_ratio
        self.num_epochs = num_epochs
        self.delta = delta
        self.prune_easy = prune_easy
        # self.scores stores the loss value of each sample. Note that smaller value indicates the sample is better learned by the network.
        self.scores = torch.ones(len(self.dataset)) * 3
        self.weights = torch.ones(len(self.dataset))
        self.num_pruned_samples = 0
        self.cur_batch_index = None
        self.dataset_len = len(self.dataset)

        print('   InfoBatch - Prune easy samples: ', self.prune_easy)
        if self.prune_easy == -1:
            print('   InfoBatch - static pruning: removing easy samples based on avg loss 4xl40s_ag_1_seed3_mean_loss_default_320k.npy')
            self.num_epochs = 10000
        if self.prune_easy == -2:
            print('   InfoBatch - static pruning: removing hard samples based on avg loss 4xl40s_ag_1_seed3_mean_loss_default_320k.npy')
            self.num_epochs = 10000

        if self.prune_easy == -10:
            print(
                f'   S2L - static pruning: keep {int(self.keep_ratio * 100.0)}% samples based on: ./3xA100_ag_1_graphcast_seed3_loss_kmean_350_k{int(self.keep_ratio * 100.0)}')
            self.num_epochs = 10000
        if self.prune_easy == -20:
            print(
                f'   S2L - static pruning: keep {int(self.keep_ratio * 100.0)}% samples based on: ./4xV100_ag_1_graphcast_seed3_loss_kmean_550_k{int(self.keep_ratio * 100.0)}')
            self.num_epochs = 10000

        if self.prune_easy == -50:
            print(
                f'   random - static pruning: keep {int(self.keep_ratio * 100.0)}% samples based on: ./random_sample_indices_seed3.npy')
            self.num_epochs = 10000

        if self.prune_easy == -51:
            print(
                f'   TDDS - static pruning: keep easy {int(self.keep_ratio * 100.0)}% samples based on: ./4xl40s_ag_1_seed3_TDDS_sam2lar.npy')
            self.num_epochs = 10000

        if self.prune_easy == -52:
            print(
                f'   TDDS - static pruning: keep hard {int(self.keep_ratio * 100.0)}% samples based on: ./4xl40s_ag_1_seed3_TDDS_sam2lar.npy')
            self.num_epochs = 10000

        if self.prune_easy == -100:
            print(
                '   new setting only select single timestamp (0AM) from 2007: samples based on: ./sample_index_pangu_lite_only00_from_2007')
            self.num_epochs = 10000
            self.keep_ratio = 1.0

        print('   InfoBatch - info_batch_num_epoch', self.num_epochs)
        print('   InfoBatch - info_batch_ratio', self.keep_ratio)
        print('   InfoBatch - info_batch_delta', self.delta)

    def __getattr__(self, name):
        # Delegate the method call to the self.dataset if it is not found in Wrapper
        return getattr(self.dataset, name)

    def set_active_indices(self, cur_batch_indices: torch.Tensor):
        self.cur_batch_index = cur_batch_indices

    def update(self, values, ID):
        assert isinstance(values, torch.Tensor)
        # test_only
        ID -= 4
        device = values.device
        if values.dim() == 0:
            # self.cur_batch_index = ID
            weights = self.weights[self.cur_batch_index.item()].to(device)
        else:
            batch_size = values.shape[0]
            assert len(self.cur_batch_index) == batch_size, 'not enough index'
            weights = self.weights[self.cur_batch_index].to(device)

        # test_only
        if self.cur_batch_index.item() != ID.item():
            print('error')
            print('error')
            print('error')
            print('error')
            print('error')
            print(self.cur_batch_index)

        indices = self.cur_batch_index.to(device)
        loss_val = values.detach().clone()
        self.cur_batch_index = []
        # count = torch.sum(self.scores != 3.0).item()
        # self.scores[self.scores != 3].cpu().numpy()

        if dist.is_available() and dist.is_initialized():
            iv = torch.cat([indices.view(1, -1), loss_val.view(1, -1)], dim=0)
            iv_whole_group = concat_all_gather(iv, 1)
            indices = iv_whole_group[0]
            loss_val = iv_whole_group[1]

        self.scores[indices.cpu().long()] = loss_val.cpu()
        values.mul_(weights)
        return values.mean()

    def __len__(self):
        # print('\n')
        # print('InfoBatch - dataset len:', len(self.dataset))

        # if self.iterations > self.stop_prune:
        #     # print('we are going to stop prune, #stop prune %d, #cur iterations %d' % (self.iterations, self.stop_prune))
        #     len_dataset = len(self.dataset)
        # else:

        # if self.prune_easy:
        #     well_learned_mask = (self.scores < self.scores.mean()).numpy()
        # else:
        #     well_learned_mask = (self.scores > self.scores.mean()).numpy()
        #
        # well_learned_indices = np.where(well_learned_mask)[0]
        # remained_indices = np.where(~well_learned_mask)[0].tolist()
        # selected_indices = np.random.choice(well_learned_indices, int(
        #     self.keep_ratio * len(well_learned_indices)), replace=False)
        # if len(selected_indices) > 0:
        #     remained_indices.extend(selected_indices)
        # self.dataset_len = len(remained_indices)

        # print('InfoBatch - new dataset len:', self.dataset_len)

        return len(self.dataset)
        # return self.dataset_len

    def __getitem__(self, index):
        # self.cur_batch_index.append(index)
        return index, self.dataset[index]  # , index
        # return self.dataset[index], index, self.scores[index]

    def prune(self):
        # Prune samples that are well learned, rebalance the weight by scaling up remaining
        # well learned samples' learning rate to keep estimation about the same
        # for the next version, also consider new class balance

        if self.prune_easy == -2 or self.prune_easy == -1:
            # rank_score_np = np.load('./mean_loss_seed3_default_320k.npy')
            rank_score_np = np.load('./4xl40s_ag_1_seed3_mean_loss_default_320k.npy')
            if self.prune_easy == -2:
                remained_indices = rank_score_np[:int(len(rank_score_np) * self.keep_ratio)]
            else:
                remained_indices = rank_score_np[-int(len(rank_score_np) * self.keep_ratio):]
            np.random.shuffle(remained_indices)
            self.dataset_len = len(remained_indices)
            print('Pruning step: static pruning #remained samples %d, total len(dataset) = %d '
                  % (np.sum(self.dataset_len), len(self.dataset)))

        elif self.prune_easy == -10:
            file_name = f'./3xA100_ag_1_graphcast_seed3_loss_kmean_350_k{int(self.keep_ratio * 100.0)}.npy'
            remained_indices = np.load(file_name)
            print(f'load ranking files: {file_name}')
            np.random.shuffle(remained_indices)
            self.dataset_len = len(remained_indices)
            print('Pruning step: static pruning #remained samples %d, total len(dataset) = %d '
                  % (np.sum(self.dataset_len), len(self.dataset)))

        elif self.prune_easy == -20:
            file_name = f'./4xV100_ag_1_graphcast_seed3_loss_kmean_550_k{int(self.keep_ratio * 100.0)}.npy'
            remained_indices = np.load(file_name)
            print(f'load ranking files: {file_name}')
            np.random.shuffle(remained_indices)
            self.dataset_len = len(remained_indices)
            print('Pruning step: static pruning #remained samples %d, total len(dataset) = %d '
                  % (np.sum(self.dataset_len), len(self.dataset)))

        elif self.prune_easy == -50:
            rank_score_np = np.load('./random_sample_indices_seed3.npy')
            remained_indices = rank_score_np[:int(len(rank_score_np) * self.keep_ratio)]
            np.random.shuffle(remained_indices)
            self.dataset_len = len(remained_indices)
            print('Pruning step: static pruning #remained samples %d, total len(dataset) = %d '
                  % (np.sum(self.dataset_len), len(self.dataset)))

        elif (self.prune_easy == -51) or (self.prune_easy == -52):
            rank_score_np = np.load('./4xl40s_ag_1_seed3_TDDS_sam2lar.npy')
            if self.prune_easy == -51:
                # keep small scores / easy
                remained_indices = rank_score_np[:int(len(rank_score_np) * self.keep_ratio)]
            else:
                # keep large scores / hard
                remained_indices = rank_score_np[-int(len(rank_score_np) * self.keep_ratio):]
            np.random.shuffle(remained_indices)
            self.dataset_len = len(remained_indices)
            print('Pruning step: static pruning #remained samples %d, total len(dataset) = %d '
                  % (np.sum(self.dataset_len), len(self.dataset)))



        elif self.prune_easy == -100:
            file_name = './sample_index_pangu_lite_only00_from_2007.npy'
            remained_indices = np.load(file_name)
            # test_only
            # remained_indices = remained_indices[:100]
            # 58432
            # test = remained_indices[remained_indices != 58432]
            # np.save(file_name, test)

            print(f'load ranking files: {file_name}')
            np.random.shuffle(remained_indices)
            self.dataset_len = len(remained_indices)
            print('Pruning step: static pruning #remained samples %d, total len(dataset) = %d '
                  % (np.sum(self.dataset_len), len(self.dataset)))

        else:
            if self.prune_easy == 1:
                well_learned_mask = (self.scores < self.scores.mean()).numpy()
            elif self.prune_easy == 0:
                well_learned_mask = (self.scores > self.scores.mean()).numpy()
            else:
                print('error')
                print('error')
                print('error')
                print('error')
                print('error')

            well_learned_indices = np.where(well_learned_mask)[0]
            remained_indices = np.where(~well_learned_mask)[0].tolist()
            if self.keep_ratio == 0.0:
                self.reset_weights()
                print('pruning step: self.keep_ratio = 0, remove all')
            else:
                selected_indices = np.random.choice(well_learned_indices, int(
                    self.keep_ratio * len(well_learned_indices)), replace=False)
                self.reset_weights()
                if len(selected_indices) > 0:
                    self.weights[selected_indices] = 1 / self.keep_ratio
                    remained_indices.extend(selected_indices)
                self.num_pruned_samples += len(self.dataset) - len(remained_indices)

            np.random.shuffle(remained_indices)
            self.dataset_len = len(remained_indices)
            print('Pruning step: #well learned samples %d, #remained samples %d, total len(dataset) = %d '
                  % (np.sum(well_learned_mask), np.sum(self.dataset_len), len(self.dataset)))

        return remained_indices

    @property
    def sampler(self):
        sampler = IBSampler(self)
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedIBSampler(sampler)
        return sampler

    def no_prune(self):
        samples_indices = list(range(len(self)))
        np.random.shuffle(samples_indices)
        return samples_indices

    def mean_score(self):
        return self.scores.mean()

    def get_weights(self, indexes):
        return self.weights[indexes]

    def get_pruned_count(self):
        return self.num_pruned_samples

    @property
    def stop_prune(self):
        return self.num_epochs * self.delta

    def reset_weights(self):
        self.weights[:] = 1


class IBSampler(object):
    def __init__(self, dataset: InfoBatch):
        self.dataset = dataset
        self.stop_prune = dataset.stop_prune
        self.iterations = 0
        self.sample_indices = None
        self.iter_obj = None
        self.reset()

    def __getitem__(self, idx):
        return self.sample_indices[idx]

    def reset(self):
        np.random.seed(self.iterations)
        if self.iterations > self.stop_prune:
            # print('we are going to stop prune, #stop prune %d, #cur iterations %d' % (self.iterations, self.stop_prune))
            if self.iterations == self.stop_prune + 1:
                self.dataset.reset_weights()
            self.sample_indices = self.dataset.no_prune()
        else:
            # print('we are going to continue pruning, #stop prune %d, #cur iterations %d' % (self.iterations, self.stop_prune))
            self.sample_indices = self.dataset.prune()
        self.iter_obj = iter(self.sample_indices)
        self.iterations += 1
        self.dataset.dataset_len = len(self.sample_indices)
        print(f'Reset - Next epoch iterations: {self.dataset.dataset_len}')
        print('\n')

    def __next__(self):
        return next(self.iter_obj)  # may raise StopIteration

    def __len__(self):
        return len(self.sample_indices)

    def __iter__(self):
        self.reset()
        return self


class DistributedIBSampler(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler can change size during training.
    """

    class DatasetFromSampler(Dataset):
        def __init__(self, sampler: IBSampler):
            self.dataset = sampler
            # self.indices = None

        def reset(self, ):
            self.indices = None
            self.dataset.reset()

        def __len__(self):
            print('DistributedIBSampler - len dataset', len(self.dataset))
            return len(self.dataset)

        def __getitem__(self, index: int):
            """Gets element of the dataset.
            Args:
                index: index of the element in the dataset
            Returns:
                Single element by index
            """
            # if self.indices is None:
            #    self.indices = list(self.dataset)
            return self.dataset[index]

    def __init__(self, dataset: IBSampler, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = True) -> None:
        sampler = self.DatasetFromSampler(dataset)
        super(DistributedIBSampler, self).__init__(
            sampler, num_replicas, rank, shuffle, seed, drop_last)
        self.sampler = sampler
        self.dataset = sampler.dataset.dataset  # the real dataset.
        self.iter_obj = None

    def __iter__(self) -> Iterator[int]:
        """
        Notes self.dataset is actually an instance of IBSampler rather than InfoBatch.
        """
        self.sampler.reset()
        if self.drop_last and len(self.sampler) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.sampler) - self.num_replicas) /
                self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(
                len(self.sampler) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # type: ignore[arg-type]
            indices = torch.randperm(len(self.sampler), generator=g).tolist()
        else:
            indices = list(range(len(self.sampler)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size /
                                                len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        # print('distribute iter is called')
        self.iter_obj = iter(itemgetter(*indices)(self.sampler))
        return self.iter_obj

