import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 sample_ids=None, sample_weights=None,
                 pruning_method=None, pruning_rate=0, args=None):
        # size [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size
        assert flag in ['train', 'test', 'val', 'infer']
        self.flag = flag
        type_map = {'train': 0, 'val': 1, 'test': 2, 'infer': 0,}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        # 计算原始滑窗样本数
        valid_length = len(self.data_x) - self.seq_len - self.pred_len + 1
        # 仅当flag为'train'时进行剪枝，否则不做样本删除
        if self.flag == 'train' and pruning_method == 0 and pruning_rate > 0:
            rs = np.random.RandomState(42)  # 固定随机种子确保 reproducibility
            num_keep = int(valid_length * (1 - pruning_rate))
            self.indices = np.sort(rs.choice(np.arange(valid_length), size=num_keep, replace=False))
        else:
            self.indices = np.arange(valid_length)

        # 如果没有传入 sample_ids，则默认使用滑窗起始索引作为 sample_id
        if sample_ids is None:
            self.sample_ids = self.indices.copy()
        else:
            self.sample_ids = sample_ids
        # sample_weights 初始化为 1
        if sample_weights is None:
            self.sample_weights = np.ones(len(self.indices))
        else:
            self.sample_weights = sample_weights

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, axis=1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, axis=1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), axis=1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, axis=1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        orig_index = self.indices[index]
        s_begin = orig_index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        sample_id = self.sample_ids[index]
        sample_weight = self.sample_weights[index]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, sample_id, sample_weight

    def __len__(self):
        return len(self.indices)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t',
                 sample_ids=None, sample_weights=None,
                 pruning_method=None, pruning_rate=0, args=None):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size
        assert flag in ['train', 'test', 'val', 'infer']
        self.flag = flag
        type_map = {'train': 0, 'val': 1, 'test': 2, 'infer': 0,}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        valid_length = len(self.data_x) - self.seq_len - self.pred_len + 1
        if self.flag == 'train' and pruning_method == 0 and pruning_rate > 0:
            rs = np.random.RandomState(42)
            num_keep = int(valid_length * (1 - pruning_rate))
            self.indices = np.sort(rs.choice(np.arange(valid_length), size=num_keep, replace=False))
        else:
            self.indices = np.arange(valid_length)

        if sample_ids is None:
            self.sample_ids = self.indices.copy()
        else:
            self.sample_ids = sample_ids
        if sample_weights is None:
            self.sample_weights = np.ones(len(self.indices))
        else:
            self.sample_weights = sample_weights

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, axis=1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, axis=1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), axis=1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, axis=1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, axis=1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        orig_index = self.indices[index]
        s_begin = orig_index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        sample_id = self.sample_ids[index]
        sample_weight = self.sample_weights[index]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, sample_id, sample_weight

    def __len__(self):
        return len(self.indices)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 sample_ids=None, sample_weights=None,
                 pruning_method=None, pruning_rate=0, args=None):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size
        assert flag in ['train', 'test', 'val', 'infer']
        self.flag = flag
        type_map = {'train': 0, 'val': 1, 'test': 2, 'infer': 0,}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        valid_length = len(self.data_x) - self.seq_len - self.pred_len + 1
        if (self.flag == 'train'
                and pruning_rate != 0
                and args.pruning_method not in {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}):

            if pruning_method == 0 and pruning_rate > 0:
                num_keep = int(valid_length * (1 - pruning_rate))
                self.indices = np.sort(np.random.choice(np.arange(valid_length), size=num_keep, replace=False))
            elif pruning_method != 0:

                if pruning_method == 2:
                    new_pr = pruning_rate
                else:
                    new_pr = 0.0

                rank_file_name = ('/home/local/zi/research_project/iTransformer/rank_list/seed{}_pm{}_pr{}_{}_{}_'
                           '{}_ft{}_sl{}_ll{}_pl{}_'
                           'dm{}_nh{}_el{}_dl{}_df{}_'
                           'fc{}_eb{}_dt{}_{}_{}_{}.npy').format(
                    0,  # 1
                    args.pruning_method,  # 2
                    int(new_pr * 100),  # 3
                    'weather_96_96',  # 4
                    args.model,  # 5
                    args.data,  # 6
                    args.features,  # 7
                    args.seq_len,  # 8
                    args.label_len,  # 9
                    args.pred_len,  # 10
                    args.d_model,  # 11
                    args.n_heads,  # 12
                    args.e_layers,  # 13
                    args.d_layers,  # 14
                    args.d_ff,  # 15
                    args.factor,  # 16
                    args.embed,  # 17
                    args.distil,  # 18
                    args.des,  # 19
                    args.class_strategy,  # 20
                    0,  # 21
                )

                # 根据异常值的大小，从小到大排列
                # if pruning_method in (21, 22):
                #     rank_file_name = ("/mnt/ssd/zi/itransformer_results/"
                #                  "trend_scores/seed0_pm0_pr0_low10_high10_start0_int20_tr30_test101_"
                #                  "iTransformer_custom_ftM_sl96_ll48_"
                #                  "pl96_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0/trend_error_train_set_all_sample_all_tokens.npy")

                rank_matrix_np = np.load(rank_file_name)
                print('load rank file:', rank_file_name)

                sorted_sample_ids = rank_matrix_np[:,0]

                # 计算要移除多少个
                remove_count = int(valid_length * abs(pruning_rate))

                if pruning_method == 21:
                    # 移除最“容易”的前 remove_count 个
                    remaining_ids = sorted_sample_ids[remove_count:]
                elif pruning_method == 22:
                    # 移除最“困难”的后 remove_count 个
                    remaining_ids = sorted_sample_ids[:-remove_count]
                    # remaining_ids = sorted_sample_ids[remove_count:]
                else:
                    if pruning_rate >= 0:
                        # 移除最“容易”的前 remove_count 个
                        remaining_ids = sorted_sample_ids[remove_count:]
                        # sample_weights = rank_matrix_np[:,1][remove_count:]
                    else:
                        # 移除最“困难”的后 remove_count 个
                        remaining_ids = sorted_sample_ids[:-remove_count]
                        # sample_weights = rank_matrix_np[:,1][:-remove_count]


                self.indices = remaining_ids.astype(int)

        else:
            self.indices = np.arange(valid_length)

        if sample_ids is None:
            self.sample_ids = self.indices.copy()
        else:
            self.sample_ids = sample_ids
        if sample_weights is None:
            self.sample_weights = np.ones(len(self.indices))
        else:
            self.sample_weights = sample_weights

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        # 调整列顺序
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, axis=1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, axis=1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), axis=1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, axis=1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        orig_index = self.indices[index]
        s_begin = orig_index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        sample_id = self.sample_ids[index]
        sample_weight = self.sample_weights[index]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, sample_id, sample_weight

    def __len__(self):
        return len(self.indices)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 sample_ids=None, sample_weights=None,
                 pruning_method=None, pruning_rate=0, args=None):
        self.seq_len, self.label_len, self.pred_len = size
        assert flag in ['train', 'test', 'val', 'infer']
        self.flag = flag
        type_map = {'train': 0, 'val': 1, 'test': 2, 'infer': 0,}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        valid_length = len(self.data_x) - self.seq_len - self.pred_len + 1
        if self.flag == 'train' and pruning_method == 0 and pruning_rate > 0:
            rs = np.random.RandomState(42)
            num_keep = int(valid_length * (1 - pruning_rate))
            self.indices = np.sort(rs.choice(np.arange(valid_length), size=num_keep, replace=False))
        else:
            self.indices = np.arange(valid_length)

        if sample_ids is None:
            self.sample_ids = self.indices.copy()
        else:
            self.sample_ids = sample_ids
        if sample_weights is None:
            self.sample_weights = np.ones(len(self.indices))
        else:
            self.sample_weights = sample_weights

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data_x = df
        self.data_y = df

    def __getitem__(self, index):
        orig_index = self.indices[index]
        s_begin = orig_index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        sample_id = self.sample_ids[index]
        sample_weight = self.sample_weights[index]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, sample_id, sample_weight

    def __len__(self):
        return len(self.indices)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 sample_ids=None, sample_weights=None,
                 pruning_method=None, pruning_rate=0, args=None):
        self.seq_len, self.label_len, self.pred_len = size
        assert flag in ['train', 'test', 'val', 'infer']
        self.flag = flag
        type_map = {'train': 0, 'val': 1, 'test': 2, 'infer': 0,}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        valid_length = len(self.data_x) - self.seq_len - self.pred_len + 1
        if self.flag == 'train' and pruning_method == 0 and pruning_rate > 0:
            rs = np.random.RandomState(42)
            num_keep = int(valid_length * (1 - pruning_rate))
            self.indices = np.sort(rs.choice(np.arange(valid_length), size=num_keep, replace=False))
        else:
            self.indices = np.arange(valid_length)

        if sample_ids is None:
            self.sample_ids = self.indices.copy()
        else:
            self.sample_ids = sample_ids
        if sample_weights is None:
            self.sample_weights = np.ones(len(self.indices))
        else:
            self.sample_weights = sample_weights

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        orig_index = self.indices[index]
        s_begin = orig_index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        sample_id = self.sample_ids[index]
        sample_weight = self.sample_weights[index]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, sample_id, sample_weight

    def __len__(self):
        return len(self.indices)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None,
                 sample_ids=None, sample_weights=None,
                 pruning_method=None, pruning_rate=0, args=None):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        # 对于预测数据集，样本数 = len(data_x) - seq_len + 1
        valid_length = len(self.data_x) - self.seq_len + 1
        # flag不为'train'时，不进行剪枝
        if self.flag == 'train' and pruning_method == 0 and pruning_rate > 0:
            rs = np.random.RandomState(42)
            num_keep = int(valid_length * (1 - pruning_rate))
            self.indices = np.sort(rs.choice(np.arange(valid_length), size=num_keep, replace=False))
        else:
            self.indices = np.arange(valid_length)

        if sample_ids is None:
            self.sample_ids = self.indices.copy()
        else:
            self.sample_ids = sample_ids
        if sample_weights is None:
            self.sample_weights = np.ones(len(self.indices))
        else:
            self.sample_weights = sample_weights

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, axis=1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, axis=1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), axis=1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, axis=1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, axis=1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        orig_index = self.indices[index]
        s_begin = orig_index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        sample_id = self.sample_ids[index]
        sample_weight = self.sample_weights[index]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, sample_id, sample_weight

    def __len__(self):
        return len(self.indices)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
