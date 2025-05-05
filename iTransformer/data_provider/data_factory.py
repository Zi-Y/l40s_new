from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, \
    Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False  # original drop_last = True
        batch_size = args.batch_size * 256 # original batch_size=1 for evaluation
        freq = args.freq
    elif flag == 'infer':
        shuffle_flag = False
        drop_last = False  # original drop_last = True
        batch_size = args.batch_size * 256  # original batch_size=1 for evaluation
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        sample_ids=None,
        sample_weights=None,
        pruning_method=args.pruning_method,
        pruning_rate=args.pruning_rate,
        args=args,
    )
    print(flag, len(data_set))
    # 这个条件语句，无影响
    if args.pruning_method in [4, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] and flag == 'train':
        return data_set, 0
    # if (args.pruning_method == 12) and (args.pruning_method == 11) and (args.pruning_method == 10) and (args.pruning_method == 9) and (args.pruning_method == 4) and (flag == 'train'):
    #     return data_set, 0
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            pin_memory=True,  # 让数据固定在内存，加快 GPU 访问
            prefetch_factor=2,  # 提前加载数据
            drop_last=drop_last)
        return data_set, data_loader
