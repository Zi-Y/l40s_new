import numpy as np
import os
import glob
import re
import concurrent.futures
import cupy as cp
from tqdm import tqdm
import itertools

def fit_linear(losses):
    """
    对序列 losses 进行简单线性回归，返回 (a, b)
    使得 y = a*x + b 拟合 l_i 与 x_i (x_i = i) 的关系。
    losses 可以是 numpy 数组或 cupy 数组，但本函数始终在 GPU 上运行
    """
    # 将 losses 转换为 cupy 数组，确保在 GPU 上运行
    losses = cp.asarray(losses)
    n = losses.shape[0]
    x = cp.arange(n)
    a, b = cp.polyfit(x, losses, 1)
    return a, b


def classify_sample(losses, threshold=0.2, global_L_mean=None):
    """
    根据给定的单个样本 (losses: [l_0, l_1, ..., l_n])（必须为 cupy 数组），
    在 GPU 上拟合线性函数 y = a*x + b 并进行分类。

    返回分类标签，如 'H->L', 'L->H', 'H->H', 'L->L' 等。
    如果 global_L_mean 参数不为 None，则使用其作为所有样本最后一个 checkpoint 的平均值（CPU 标量）。
    """
    n = losses.shape[0]
    if n <= 1:
        return "SinglePoint"

    # 1. 线性拟合（GPU上执行）
    a, b = fit_linear(losses)

    # 2. 计算 L_start, L_end, ΔL
    L_start = b             # x = 0 时
    L_end = a * (n - 1) + b   # x = n-1 时
    delta_L = L_end - L_start

    # 3. 使用 global_L_mean（如果提供），否则计算当前样本的平均损失（转回 CPU 计算）
    if global_L_mean is None:
        L_mean = cp.mean(losses).get()
    else:
        L_mean = global_L_mean

    # 4. 将 GPU 标量转换为 CPU 标量进行比较
    delta_L = float(delta_L.get()) if isinstance(delta_L, cp.ndarray) else float(delta_L)
    L_start = float(L_start.get()) if isinstance(L_start, cp.ndarray) else float(L_start)
    L_end = float(L_end.get()) if isinstance(L_end, cp.ndarray) else float(L_end)

    # 5. 根据规则分类
    if delta_L < -threshold:
        category = "H->L"
    elif delta_L > threshold:
        category = "L->H"
    else:
        start_is_high = (L_start > L_mean)
        end_is_high = (L_end > L_mean)
        if start_is_high and end_is_high:
            category = "H->H"
        elif not start_is_high and not end_is_high:
            category = "L->L"
        else:
            category = "Ambiguous"
    return category


if __name__ == "__main__":
    # 定义存放 loss 文件的目录
    # data_dir = ("/mnt/ssd/zi/itransformer_results/"
    #             "seed0_pm0_pr0_low10_high10_start0_int20_tr50_"
    #             "test101_iTransformer_custom_ftM_sl96_ll48_pl96_"
    #             "dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0")

    data_dir = ("/mnt/ssd/zi/itransformer_results/new/"
                "seed0_pm0_pr0_low10_high10_start0_"
                "int20_tr50_test101_iTransformer_custom_"
                "ftM_sl96_ll48_pl96_dm512_nh8_el3_dl1_df512_"
                "fc1_ebtimeF_dtTrue_exp_projection_0")




    # 构造匹配文件的路径模式
    pattern = os.path.join(data_dir, "iter_*_train_set_all_sample_all_tokens.npy")

    # 添加一个选项，选择 'typical' 或 'random'
    select_method = 'random'  # 修改为 'random' 则随机选择

    # 定义 sample_level 的值，例如 'sample', 'all_token' 或 'unique_token'
    sample_level = 'all_token'
    seed = 0


    # 使用 glob 获取所有匹配的文件路径
    file_paths = glob.glob(pattern)
    file_paths = sorted(file_paths,
                        key=lambda x: int(re.search(r'iter_(\d+)_', os.path.basename(x)).group(1))
                        if re.search(r'iter_(\d+)_', os.path.basename(x)) else 0)
    # file_paths = file_paths[::2]  # 只读取第 1,3,5,... 个文件

    file_paths = file_paths[:66]  # 只读取第 1,3,5,... 个文件

    if not file_paths:
        print("未找到匹配的loss文件。")
    else:
        # 使用线程池并行加载文件，并显示加载进度
        import itertools


        def load_file(file_path, sample_level):
            # 使用内存映射模式加载文件
            mat = np.load(file_path, mmap_mode='r')

            if sample_level == 'sample':  # 对于 'sample'，计算每个样本的均值
                mat = mat.mean(axis=(1, 2))
            elif sample_level == 'all_token':
                mat = mat.reshape(-1)

            elif sample_level == 'unique_token':
                # 将 mat 转换为 cupy 数组，确保在 GPU 上运行
                mat = cp.asarray(mat)
                # 假设 mat 的 shape 为 (num_samples, forecast_steps, num_vars)
                num_samples, forecast_steps, num_vars = mat.shape

                # --- 第一步：重排 A ---
                # 将样本和预测步长合并，得到 B 的 shape 为 (num_samples*forecast_steps, num_vars)
                B = mat.reshape(-1, num_vars)

                # --- 第二步：生成时间 index ---
                # 对于第 i 个样本和第 j 个预测步，其真实时间 = i + j
                time_index = (cp.arange(num_samples)[:, None] + cp.arange(forecast_steps)[None, :]).reshape(-1)

                # --- 第三步：对相同 time_index 的 B 行求均值 ---
                # 对 time_index 进行排序，并重排 B
                order = cp.argsort(time_index)
                time_index_sorted = time_index[order]
                B_sorted = B[order]

                # 利用 cp.unique 找到所有唯一的时间值及每组的起始位置和计数
                unique_time, start_indices, counts = cp.unique(time_index_sorted, return_index=True, return_counts=True)

                # 利用 cp.add.reduceat 在排序后的 B 上对各组进行求和
                group_sums = cp.add.reduceat(B_sorted, start_indices, axis=0)
                # 计算均值：每组和除以该组样本数（利用广播）
                mean_B = group_sums / counts[:, None]
                # 最终 mean_B 的 shape 为 (len(unique_time), num_vars)

                # 若后续需要在 CPU 上使用，可调用 .get()，这里直接返回 GPU 数组
                mat = mean_B.reshape(-1).get()

            return mat




        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            matrices = list(tqdm(executor.map(load_file, file_paths, itertools.repeat(sample_level)),
                                 total=len(file_paths), desc="Loading files"))

        combined_matrix_cpu = np.vstack(matrices).T

        # 假设 matrices 是一个长度为 N 的 Python 列表，每项是形如 (T,) 的 NumPy 数组
        N = len(matrices)
        batch_size = 1024  # 每次处理 1024 条曲线，你可以根据显存灵活调整

        # 在 GPU 上累加和
        sum_gpu = cp.array(0.0, dtype=cp.float64)
        count = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            # 在 CPU 上先提取这一 batch 的最后一个 checkpoint（浮点数列表）
            last_vals_cpu = np.array([mat[-1] for mat in matrices[start:end]], dtype=np.float64)
            # 上传到 GPU 并累加
            last_vals_gpu = cp.asarray(last_vals_cpu)
            sum_gpu += last_vals_gpu.sum()
            count += last_vals_gpu.size

        # 计算全局平均
        global_L_mean_gpu = sum_gpu / count
        print("所有样本最后一个 checkpoint 的平均值 (L_mean, GPU):", global_L_mean_gpu)


        # # 假设你的 matrices 列表里原本是 numpy 数组，先把它们转换到 GPU
        # combined_matrix_gpu = [cp.asarray(mat) for mat in matrices]
        #
        # # 在 GPU 上沿第 0 维拼接，再转置，得 (样本数, 时间点数)
        # combined_matrix_gpu = cp.vstack(combined_matrix_gpu).T
        # print("loss 矩阵的维度 (GPU):", combined_matrix_gpu.shape)
        #
        # # 计算所有样本最后一个 checkpoint 的平均值
        # global_L_mean_gpu = cp.mean(combined_matrix_gpu[:, -1])
        # print("所有样本最后一个 checkpoint 的平均值 (L_mean, GPU):", global_L_mean_gpu)





        # # 在 CPU 上合并所有文件数据，形状为 (时间点数量, 样本数量)，然后转置为 (样本数量, 时间点数量)
        # combined_matrix_cpu = np.vstack(matrices).T
        # print("loss矩阵的维度 (CPU):", combined_matrix_cpu.shape)
        #
        # # 计算所有样本最后一个 checkpoint 的平均值（在 CPU 上计算）
        # global_L_mean = np.mean(combined_matrix_cpu[:, -1])
        # print("所有样本最后一个checkpoint的平均值 (L_mean):", global_L_mean)

        # 分批次处理样本，采用 GPU 向量化进行批量分类，避免逐个样本循环
        batch_size = 4096 * 4
        num_samples = combined_matrix_cpu.shape[0]
        threshold = 0.2
        category_codes = []  # 存放每个批次的分类代码数组

        # 将 global_L_mean 转为 GPU 标量
        # global_L_mean_gpu = cp.asarray(global_L_mean)

        for i in tqdm(range(0, num_samples, batch_size), desc="Classifying batches"):
            batch = combined_matrix_cpu[i:i+batch_size]  # shape: (B, T)
            batch_gpu = cp.asarray(batch)  # 转移到 GPU
            T = batch_gpu.shape[1]
            x = cp.arange(T)  # 时间序列
            mean_x = (T - 1) / 2.0
            var_x = cp.sum((x - mean_x) ** 2)

            # 计算每个样本的均值
            mean_y = cp.mean(batch_gpu, axis=1)  # shape: (B,)
            # 计算线性回归参数 a 和 b
            a = cp.sum((x - mean_x) * (batch_gpu - mean_y[:, None]), axis=1) / var_x  # shape: (B,)
            b = mean_y - a * mean_x

            # 计算预测起始值和末尾值
            L_start = b
            L_end = a * (T - 1) + b
            delta = L_end - L_start

            # 初始化分类代码数组，使用 int32 类型
            codes = cp.empty(delta.shape, dtype=cp.int32)

            # 对于 delta < -threshold, 设为 0 (H->L)
            codes[delta < -threshold] = 0
            # 对于 delta > threshold, 设为 1 (L->H)
            codes[delta > threshold] = 1

            # 对于 -threshold <= delta <= threshold 的样本，进一步判断
            mask = (delta >= -threshold) & (delta <= threshold)
            # 默认设置为 4 (Ambiguous)
            codes[mask] = 4
            # 如果 L_start 和 L_end 均大于 global_L_mean, 设为 2 (H->H)
            # mask2 = mask & ((L_start > global_L_mean_gpu) & (L_end > global_L_mean_gpu))
            mask2 = mask & (L_end > global_L_mean_gpu)
            codes[mask2] = 2
            # 如果 L_start 和 L_end 均小于等于 global_L_mean, 设为 3 (L->L)
            # mask3 = mask & ((L_start <= global_L_mean_gpu) & (L_end <= global_L_mean_gpu))
            mask3 = mask & (L_end <= global_L_mean_gpu)
            codes[mask3] = 3

            category_codes.append(codes)

        # 合并所有批次的分类结果
        all_codes = cp.concatenate(category_codes)
        all_codes_cpu = cp.asnumpy(all_codes)
        unique, counts = np.unique(all_codes_cpu, return_counts=True)
        total_samples = all_codes_cpu.shape[0]
        code_to_cat = {0: "H->L", 1: "L->H", 2: "H->H", 3: "L->L", 4: "Ambiguous"}
        category_counts = {code_to_cat[k]: v for k, v in zip(unique, counts)}

        print("样本总数:", total_samples)
        print("各分类所占比例：")
        for cat, count in category_counts.items():
            proportion = count / total_samples
            print(f"{cat}: {proportion:.2%}")

        # 选择每个类别中5个样本，并保存其loss曲线数据和绘图
        import matplotlib.pyplot as plt



        # 构建一个字典，每个类别对应样本索引列表
        category_indices = {cat: [] for cat in code_to_cat.values()}
        for i, code in enumerate(all_codes_cpu):
            cat = code_to_cat[code]
            category_indices[cat].append(i)

        loss_curves = combined_matrix_cpu  # shape: (num_samples, T)

        for cat, indices in category_indices.items():
            if len(indices) == 0:
                continue
            if select_method == 'typical':
                # 将当前类别的 loss 曲线转换到 GPU 上进行向量化计算
                curves_cpu = loss_curves[indices, :]  # shape: (n, T)
                curves_gpu = cp.asarray(curves_cpu)

                # 计算该类别的均值曲线（在 GPU 上）
                mean_curve_gpu = cp.mean(curves_gpu, axis=0)  # shape: (T,)

                # 计算每个样本与均值曲线的 L2 距离（在 GPU 上）
                distances_gpu = cp.linalg.norm(curves_gpu - mean_curve_gpu, axis=1)

                # 对距离进行排序，并转换为 CPU 数组
                sorted_order = cp.asnumpy(cp.argsort(distances_gpu))

                # 选取距离最小的5个样本（最典型的样本）
                selected_indices = [indices[j] for j in sorted_order[:5]]
            elif select_method == 'random':
                import random
                random.seed(seed)
                if len(indices) < 5:
                    selected_indices = indices
                else:
                    selected_indices = random.sample(indices, 5)
            else:
                print("Unknown select_method: choose 'typical' or 'random'")
                continue

            selected_curves = loss_curves[selected_indices, :]

            # 绘制 loss 曲线并保存图像
            import os
            output_dir = "./results_typical_sample/new/"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.figure()
            from tqdm import tqdm
            for curve in tqdm(selected_curves, desc=f"Plotting curves for {cat}", leave=False):
                plt.plot(curve)
            plt.title(f"{select_method.capitalize()} loss curves for {cat} ({sample_level})")
            plt.xlabel("Checkpoint index")
            plt.ylabel("Loss")
            plt.savefig(os.path.join(output_dir, f"{sample_level}_{select_method}_{cat}_seed_{seed}.png"),dpi=300)
            plt.close()

            print(f"Category {cat}: saved {len(selected_indices)} {select_method} loss curves.")