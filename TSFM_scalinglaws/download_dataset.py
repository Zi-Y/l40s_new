# 首先，确保安装了 datasets 库：
# pip install datasets

from datasets import load_dataset

def download_tsfm_scalinglaws_dataset(save_to_disk: bool = False, cache_dir: str = None):
    """
    下载并加载 Qingren/TSFM-ScalingLaws-Dataset 数据集。

    参数:
        save_to_disk (bool): 如果为 True，则将下载后的数据集保存到本地（save_to_disk_path）。
        cache_dir (str, optional): 指定本地缓存目录，用于存放下载的原始文件及预处理文件。

    返回:
        dataset (DatasetDict): 加载后的数据集对象，包含所有 split。
    """
    # load_dataset 会自动下载并缓存到本地 ~/.cache/huggingface/datasets/ 下，
    # 如果传入 cache_dir，则会缓存到指定目录。
    dataset = load_dataset(
        "Qingren/TSFM-ScalingLaws-Dataset",
        cache_dir=cache_dir  # 可选，如果想指定缓存地址
    )

    if save_to_disk:
        # 将加载后的数据集以 Arrow 格式保存到本地，方便离线复用
        save_to_disk_path = "/mnt/ssd/zi/TSFM_scalinglaws/"
        dataset.save_to_disk(save_to_disk_path)
        print(f"数据集已保存到：{save_to_disk_path}")

    return dataset

if __name__ == "__main__":
    # 调用示例：下载并缓存到默认位置，同时将处理后的数据保存到本地
    ds = download_tsfm_scalinglaws_dataset(save_to_disk=True)

    # 打印一下各个 split 的基本信息
    for split_name, split_data in ds.items():
        print(f"Split: {split_name}, 数据条目数: {len(split_data)}")

    # 例如，查看第一个条目：
    print(ds["train"][0])