import torch
import torch.nn.functional as F  # 尽管 mae 被移除，但其他地方可能仍会用到 F (目前没有)
import numpy as np
import argparse  # 用于解析命令行参数，方便配置脚本行为
import matplotlib.pyplot as plt  # 用于绘图

# --- 从用户上传的 'derivatives.py' 文件中直接复制的内容 ---
from scipy.sparse import spdiags, coo_matrix, dia_matrix
from itertools import chain
import cvxpy  # 凸优化库，是 trendfilter 的核心


def first_derivative_matrix(n):
    """
    来自 derivatives.py
    生成表示一阶导数算子的稀疏矩阵。
    参数 n: 序列长度。
    返回: 应用一阶导数算子的稀疏矩阵。
    """
    e = np.ones((1, n))
    return spdiags(np.vstack((-1 * e, e)), range(2), n - 1, n)


def first_derv_nes_cvxpy(x_coords, y_var):
    """
    来自 derivatives.py (略作修改以适应 cvxpy 变量)
    计算非等间隔点上的一阶导数 (cvxpy 版本)。
    参数 x_coords: x 坐标 (numpy 数组)。
    参数 y_var: cvxpy 变量，表示 y 值。
    返回: 一阶导数的 cvxpy 表达式。
    """
    n = len(x_coords)
    ep = 1e-9
    idx = 1.0 / (x_coords[1:] - x_coords[0:-1] + ep)
    matrix = first_derivative_matrix(n)
    return cvxpy.multiply(idx, matrix @ y_var)


def second_derivative_matrix_nes(x_coords, a_min=0.0, a_max=None, scale_free=False):
    """
    来自 derivatives.py
    获取非等间隔点的二阶导数矩阵。
    参数 x_coords: x 坐标的 numpy 数组。
    参数 a_min, a_max: x 坐标间隔的最小/最大值约束。
    参数 scale_free: 是否进行无尺度计算。
    返回: 二阶导数算子的稀疏矩阵。
    假设点已排序。
    """
    n = len(x_coords)
    m = n - 2
    values = []
    for i in range(1, n - 1):
        a0 = float(x_coords[i + 1] - x_coords[i])
        a2 = float(x_coords[i] - x_coords[i - 1])
        assert (a0 >= 0) and (a2 >= 0), "Points do not appear to be sorted"
        assert (a0 > 0) and (a2 > 0), "Second derivative doesn't exist for repeated points"
        if a_max is not None:
            a0 = min(a0, a_max)
            a2 = min(a2, a_max)
        a0 = max(a0, a_min)
        a2 = max(a2, a_min)
        a1 = a0 + a2
        if scale_free:
            scf = a1 / 2.0
        else:
            scf = 1.0
        vals = [2.0 * scf / (a1 * a2), -2.0 * scf / (a0 * a2), 2.0 * scf / (a0 * a1)]
        values.extend(vals)
    i_indices = list(chain(*[[_] * 3 for _ in range(m)]))
    j_indices = list(chain(*[[_, _ + 1, _ + 2] for _ in range(m)]))
    d2 = coo_matrix((values, (i_indices, j_indices)), shape=(m, n))
    return dia_matrix(d2)


# --- 从用户上传的 'trendfilter.py' 文件中直接复制/适配的内容 ---
def get_trend_objective_function_and_model(y_signal, loss_type='mae', y_err=None, positive=False):
    """
    来自 trendfilter.py 的 get_obj_func_model (简化版)
    构造数据拟合项目标函数和模型表达式。
    """
    n = len(y_signal)
    if y_err is None:
        y_err = np.ones(n)
    else:
        assert len(y_err) == n
    buff = 0.01 * np.median(np.abs(y_signal)) if n > 0 and np.median(np.abs(y_signal)) > 1e-6 else 0.01
    buff_2 = buff ** 2
    isig = 1 / np.sqrt(buff_2 + y_err ** 2)
    base_model_s = cvxpy.Variable(n, pos=positive)
    model_expression = base_model_s
    diff = cvxpy.multiply(isig, model_expression - y_signal)
    if loss_type == 'mse':
        objective_fit_term = cvxpy.sum(cvxpy.huber(diff))
    elif loss_type == 'mae':
        objective_fit_term = cvxpy.sum(cvxpy.abs(diff))
    else:
        raise ValueError(f"Unsupported loss type for trendfilter: {loss_type}")
    return {
        'base_model': base_model_s,
        'objective_function': objective_fit_term
    }


def get_trend_regularization(x_coords, base_model_s, derv_1_expression, l_norm_penalty, alpha1_penalty, alpha2_penalty):
    """
    来自 trendfilter.py 的 get_reg (简化版)
    构造正则化项。
    """
    d2_matrix = second_derivative_matrix_nes(x_coords, scale_free=True)
    if l_norm_penalty == 2:
        norm_func = cvxpy.sum_squares
    elif l_norm_penalty == 1:
        norm_func = cvxpy.norm1
    else:
        raise ValueError(f"Unsupported l_norm_penalty: {l_norm_penalty}")
    reg_1 = alpha1_penalty * norm_func(derv_1_expression)
    reg_2 = alpha2_penalty * norm_func(d2_matrix @ base_model_s)
    regs_list = [reg_1, reg_2]
    reg_sum_expression = sum(regs_list)
    return reg_sum_expression, regs_list


def calculate_actual_trend_for_dimension(
        y_signal_dim: np.ndarray,
        lambda_reg: float,
        trend_loss_type: str = 'mae',
        time_indices: np.ndarray = None,
        solver: str = 'ECOS'
) -> np.ndarray:
    """
    为单个维度的时间序列计算实际趋势。(RobustTSF 论文 Equation 4)
    """
    n = len(y_signal_dim)
    if n == 0: return np.array([])
    if n <= 2 and lambda_reg > 0: return y_signal_dim

    if time_indices is None:
        time_indices = np.arange(n, dtype=float)
    else:
        assert len(time_indices) == n, "time_indices must have the same length as y_signal_dim"

    obj_and_model = get_trend_objective_function_and_model(y_signal_dim, loss_type=trend_loss_type, positive=False)
    base_model_s = obj_and_model['base_model']
    objective_fit_term = obj_and_model['objective_function']
    derv_1_expression = first_derv_nes_cvxpy(time_indices, base_model_s)
    reg_sum_expression, _ = get_trend_regularization(
        time_indices, base_model_s, derv_1_expression,
        l_norm_penalty=1,
        alpha1_penalty=0.0,
        alpha2_penalty=lambda_reg
    )
    total_objective = cvxpy.Minimize(objective_fit_term + reg_sum_expression)
    problem = cvxpy.Problem(total_objective)
    try:
        problem.solve(solver=solver, verbose=False)
        if base_model_s.value is None:
            # print(f"Warning: Trend calculation failed for a dimension (solver {solver}, status {problem.status}). Returning NaNs.")
            return np.full_like(y_signal_dim, np.nan)
        return np.array(base_model_s.value)
    except Exception as e:
        # print(f"Error during trend calculation for a dimension with solver {solver}: {e}. Returning NaNs.")
        return np.full_like(y_signal_dim, np.nan)


# --- 从用户上传的 'train_noisy_weighting.py' 文件中迁移和适配的函数 ---
# 确保 cal_anoscore 定义在此处，以解决 NameError
def cal_anoscore(src_err_for_sample_dim: torch.Tensor, func: str = 'dirac', start_T: int = 15,
                 end_T: int = 16) -> float:
    """
    为单个特征维度的误差序列计算异常得分 A(x_n)。
    核心逻辑来自用户上传的 'train_noisy_weighting.py' 文件中的 `cal_anoscore` 函数。
    """
    anoscore = 0.0
    actual_start_T = max(0, start_T)
    actual_end_T = min(len(src_err_for_sample_dim), end_T)
    if func == 'exp':
        for t in range(actual_start_T, actual_end_T):
            anoscore += np.exp(-np.square(t - (actual_end_T - 1))) * src_err_for_sample_dim[t].item()
    elif func == 'dirac':
        for t in range(actual_start_T, actual_end_T):
            anoscore += src_err_for_sample_dim[t].item()
    return anoscore


def diff_series_single_sample_dim(src_sample_dim: torch.Tensor, win_size: int = 1) -> torch.Tensor:
    """
    为单个特征维度的样本计算差分序列。
    核心逻辑来自用户上传的 'train_noisy_weighting.py' 文件中的 `diff_series` 函数。
    """
    src_np = src_sample_dim.cpu().numpy()
    length = len(src_np)
    if length <= win_size:
        return torch.tensor([], dtype=torch.float32)
    length_new = length - win_size
    src_new_np = np.zeros(length_new)
    for j in range(length_new):
        src_new_np[j] = src_np[j + win_size] - np.mean(src_np[j: j + win_size])
    return torch.tensor(np.abs(src_new_np), dtype=torch.float32)


# --- 核心函数：计算多维样本的异常得分 A(x_n) ---
def calculate_multidim_anomaly_score(
        observed_sample: torch.Tensor,  # 形状: (seq_length, num_features)
        trend_sample: torch.Tensor,  # 形状: (seq_length, num_features)
        # residual_sample: torch.Tensor, # 如果 anoscore='STL_resid' 则需要，暂时注释掉以简化
        args: argparse.Namespace,
        anomaly_score_agg_method: str = 'mean'  # 'mean' 或 'max' 用于聚合各维度异常分
) -> float:
    """
    为单个多维样本计算聚合后的异常得分 A(x_n)。
    此函数基于 RobustTSF 论文 Equation 5 的思想，并扩展到多维。
    它不再计算基础损失或指示函数。

    参数:
        observed_sample (torch.Tensor): 观测到的多维输入样本 (tilde_x_n)。
        trend_sample (torch.Tensor): 对应的多维趋势序列 (s_n)。
        args (argparse.Namespace): 包含配置参数的对象，例如:
                                   args.anoscore - 计算异常得分所依据的误差类型。
                                   args.score_func - 异常得分的加权函数 (传递给 cal_anoscore 的 func 参数)。
                                   args.lay_back - 从误差序列末尾取多少个点计算异常得分。
                                   args.win_size - 'diff' 方法的窗口大小。
        anomaly_score_agg_method (str): 如何聚合来自多个维度的异常得分 ('mean' 或 'max')。

    返回:
        float: 聚合后的样本异常得分 A(x_n)。
    """
    num_features = observed_sample.shape[1]  # 获取样本的特征维度数量
    all_dim_pscores = []  # 用于存储每个维度的异常得分

    # 遍历每个特征维度，计算该维度的误差序列和异常得分
    for dim_idx in range(num_features):
        obs_dim = observed_sample[:, dim_idx]  # 当前维度的观测序列
        trend_dim = trend_sample[:, dim_idx]  # 当前维度的趋势序列
        # res_dim = residual_sample[:, dim_idx] # 如果需要 STL_resid

        # 根据 args.anoscore 的设置，选择计算误差项 err_for_score_dim 的方法
        if args.anoscore == 'trendfilter':
            err_for_score_dim = torch.abs(obs_dim - trend_dim)
        elif args.anoscore == 'diff':
            err_for_score_dim = diff_series_single_sample_dim(obs_dim, win_size=args.win_size)
        elif args.anoscore == 'STL':  # 在原始代码中，STL 的误差计算方式与 trendfilter 类似
            err_for_score_dim = torch.abs(obs_dim - trend_dim)
        # elif args.anoscore == 'STL_resid': # 暂时注释，因为 residual_sample 被注释掉了
        #     err_for_score_dim = torch.abs(res_dim)
        else:
            raise ValueError(f"Unknown anoscore type: {args.anoscore}")

        if len(err_for_score_dim) == 0:  # 如果误差序列为空
            dim_pscore = 0.0
        else:
            seq_length_for_score = len(err_for_score_dim)
            end_T_score = seq_length_for_score
            start_T_score = end_T_score - args.lay_back
            if start_T_score < 0: start_T_score = 0

            if start_T_score >= end_T_score:  # 如果计算范围无效
                dim_pscore = 0.0
            else:
                dim_pscore = cal_anoscore(err_for_score_dim, func=args.score_func, start_T=start_T_score,
                                          end_T=end_T_score)
        all_dim_pscores.append(dim_pscore)

    # 聚合来自所有维度的异常得分
    if not all_dim_pscores:
        pscore = 0.0
    elif anomaly_score_agg_method == 'mean':
        pscore = np.mean(all_dim_pscores)
    elif anomaly_score_agg_method == 'max':
        pscore = np.max(all_dim_pscores)
    else:
        raise ValueError(f"Unknown anomaly_score_agg_method: {anomaly_score_agg_method}")

    return float(pscore)


def plot_observed_and_trend_multidim(observed_sample: torch.Tensor, trend_sample: torch.Tensor,
                                     sample_name: str = "Sample"):
    """
    为多维样本的每个特征维度分别绘制观测数据和趋势数据。
    """
    num_features = observed_sample.shape[1]
    seq_length = observed_sample.shape[0]
    time_steps = np.arange(seq_length)
    fig, axes = plt.subplots(num_features, 1, figsize=(12, 3 * num_features), sharex=True)
    if num_features == 1: axes = [axes]
    fig.suptitle(f'{sample_name}: Observed vs. Trend Data (Multidimensional)', fontsize=16)
    for i in range(num_features):
        ax = axes[i]
        ax.plot(time_steps, observed_sample[:, i].cpu().numpy(), label=f'Observed Dim {i + 1}', marker='o',
                linestyle='-')
        trend_dim_data = trend_sample[:, i].cpu().numpy()
        if np.isnan(trend_dim_data).any():
            ax.plot(time_steps, trend_dim_data, label=f'Trend Dim {i + 1} (with NaN)', marker='x', linestyle=':')
        else:
            ax.plot(time_steps, trend_dim_data, label=f'Trend Dim {i + 1}', marker='x', linestyle='--')
        ax.set_ylabel(f'Value Dim {i + 1}')
        ax.legend();
        ax.grid(True)
    axes[-1].set_xlabel('Time Step')
    plt.tight_layout(rect=[0, 0, 1, 0.96]);
    plt.show()


# --- 主执行示例 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate Anomaly Score A(x_n) for a multi-dimensional sample with actual trend calculation and plot")
    # 参数用于趋势和异常得分计算
    parser.add_argument('--anoscore', type=str, help='Error term type for anomaly score base [trendfilter, diff, STL]',
                        default='trendfilter')
    parser.add_argument('--score_func', type=str, help='Weighting function for anomaly score [exp, dirac]',
                        default='dirac')  # 论文固定 Dirac
    parser.add_argument('--lay_back', type=int, default=10, #1
                        help='Number of points from end of error series for anomaly score (1 implies K\'=K-1)')  # 论文固定 K'-K-1
    parser.add_argument('--win_size', type=int, help='Window size for diff method', default=1)
    parser.add_argument('--seq_length', type=int, default=16,
                        help='Sequence length of samples (K in paper, e.g., 16 for Elec/Traffic)')  # 论文实验常用值
    parser.add_argument('--num_features', type=int, default=2, help='Number of features for multivariate sample')
    parser.add_argument('--anomaly_score_agg', type=str, default='mean', choices=['mean', 'max'],
                        help='Method to aggregate anomaly scores from multiple dimensions')
    parser.add_argument('--lambda_reg_trend', type=float, default=0.3,
                        help='Regularization parameter (lambda) for trend filtering (Equation 4)')  # 论文固定 lambda=0.3
    parser.add_argument('--trend_loss_type', type=str, default='mae', choices=['mae', 'mse'],
                        help='Loss type for fitting term in trend calculation (should be mae for Eq.4)')  # 论文 Eq.4 使用绝对值
    parser.add_argument('--cvxpy_solver', type=str, default='ECOS', help='Solver for cvxpy (e.g., ECOS, SCS)')

    args = parser.parse_args()
    print("Using arguments:", args)
    print("-" * 30)

    # --- 1. 创建多维观测样本 (observed_sample_md) ---
    print(f"Generating Example Multidimensional Sample (Features: {args.num_features}, SeqLen: {args.seq_length})")
    observed_features_list_np = []
    for i in range(args.num_features):
        base_seq = np.sin(np.linspace(0, (i + 1) * np.pi, args.seq_length * 2)) + np.random.randn(
            args.seq_length * 2) * 0.1 + (i + 1)
        obs_dim_np = base_seq[:args.seq_length]
        if i == 1 and args.num_features > 1 and args.seq_length > 5:
            # 让最后两个点都有异常
            num_anom_points = min(2, max(5, args.lay_back))  # 至少影响一个点，最多影响 lay_back 个点或2个点
            obs_dim_np[-num_anom_points:] += 3.0 + i * 0.8
        elif i == 0 and args.num_features > 0 and args.seq_length > 5 and args.lay_back > 2:  # 为第一个特征的中间引入异常
            mid_point = args.seq_length // 2
            obs_dim_np[mid_point - 1: mid_point + 1] -= 3.0
        observed_features_list_np.append(obs_dim_np)
    observed_sample_np = np.stack(observed_features_list_np, axis=1)
    observed_sample_md = torch.from_numpy(observed_sample_np).float()

    # --- 2. 为每个维度计算实际趋势 (trend_sample_md) ---
    print("Calculating actual trend for each dimension...")
    calculated_trend_features_list_np = []
    time_indices_for_trend = np.arange(args.seq_length, dtype=float)
    for i in range(args.num_features):
        y_signal_dim_np = observed_sample_md[:, i].cpu().numpy()
        print(f"  Calculating trend for dimension {i + 1}...")
        trend_dim_np = calculate_actual_trend_for_dimension(
            y_signal_dim_np,
            lambda_reg=args.lambda_reg_trend,
            trend_loss_type=args.trend_loss_type,
            time_indices=time_indices_for_trend,
            solver=args.cvxpy_solver
        )
        if np.isnan(trend_dim_np).any():
            print(
                f"  Warning: Trend calculation for dimension {i + 1} resulted in NaNs. Using observed signal as fallback trend.")
            calculated_trend_features_list_np.append(y_signal_dim_np.copy())
        else:
            calculated_trend_features_list_np.append(trend_dim_np)
    trend_sample_np = np.stack(calculated_trend_features_list_np, axis=1)
    trend_sample_md = torch.from_numpy(trend_sample_np).float()

    # --- 3. 计算聚合的异常得分 A(x_n) ---
    print("\nCalculating anomaly score A(x_n) for the sample...")
    # residual_sample_md = observed_sample_md - trend_sample_md # 如果需要STL_resid，则取消注释
    anomaly_score = calculate_multidim_anomaly_score(
        observed_sample_md,
        trend_sample_md,
        # residual_sample_md, # 如果 anoscore 是 STL_resid，则需要它
        args,
        anomaly_score_agg_method=args.anomaly_score_agg
    )
    print(f"\n--- Results for the Multidimensional Sample ---")
    print(f"Observed Sample (shape): {observed_sample_md.shape}")
    print(f"Calculated Trend (shape): {trend_sample_md.shape}")
    print(
        f"Aggregated Anomaly Score A(x_n): {anomaly_score:.4f} (Aggregation: {args.anomaly_score_agg}, Score Func: {args.score_func}, Layback: {args.lay_back})")

    # --- 4. 绘图 ---
    print("\nPlotting observed vs. calculated trend...")
    plot_observed_and_trend_multidim(observed_sample_md, trend_sample_md,
                                     f"Multidim Sample (Trend Calculated, Agg: {args.anomaly_score_agg})")
    print("\nScript finished.")