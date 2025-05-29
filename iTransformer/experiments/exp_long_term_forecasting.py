from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from infobatch import InfoBatch
import math

import copy
from torch.utils.data import DataLoader, Subset

from torch.nn.utils import parameters_to_vector
import torch.nn.functional as F
warnings.filterwarnings('ignore')

# 计算trend ----------------------开始----------------------
# --- 依赖函数 (来自用户上传的 derivatives.py 和 trendfilter.py 的适配) ---
from scipy.sparse import spdiags, coo_matrix, dia_matrix
from itertools import chain
import cvxpy
import matplotlib.pyplot as plt


def first_derivative_matrix(n):
    e = np.ones((1, n))
    return spdiags(np.vstack((-1 * e, e)), range(2), n - 1, n)


def first_derv_nes_cvxpy(x_coords, y_var):
    n = len(x_coords)
    ep = 1e-9
    idx = 1.0 / (x_coords[1:] - x_coords[0:-1] + ep)
    matrix = first_derivative_matrix(n)
    return cvxpy.multiply(idx, matrix @ y_var)


def second_derivative_matrix_nes(x_coords, a_min=0.0, a_max=None, scale_free=False):
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


def get_trend_objective_function_and_model(y_signal, loss_type='mae', y_err=None, positive=False):
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
    return {'base_model': base_model_s, 'objective_function': objective_fit_term}


def get_trend_regularization(x_coords, base_model_s, derv_1_expression, l_norm_penalty, alpha1_penalty,
                             alpha2_penalty):
    d2_matrix = second_derivative_matrix_nes(x_coords, scale_free=True)
    if l_norm_penalty == 2:
        norm_func = cvxpy.sum_squares
    elif l_norm_penalty == 1:
        norm_func = cvxpy.norm1
    else:
        raise ValueError(f"Unsupported l_norm_penalty: {l_norm_penalty}")
    reg_1 = alpha1_penalty * norm_func(derv_1_expression)
    reg_2 = alpha2_penalty * norm_func(d2_matrix @ base_model_s)
    return sum([reg_1, reg_2]), [reg_1, reg_2]


def _calculate_actual_trend_for_dimension_internal(
        y_signal_dim: np.ndarray,
        lambda_reg: float,
        trend_loss_type: str = 'mae',
        time_indices: np.ndarray = None,
        solver: str = 'ECOS',
        verbose: bool = False
) -> np.ndarray:
    n = len(y_signal_dim)
    if n == 0: return np.array([])
    if n <= 2 and lambda_reg > 0:
        if verbose: print(f"    TrendCalc: Seq length {n} <= 2, returning original signal.")
        return y_signal_dim.copy()

    if time_indices is None:
        time_indices = np.arange(n, dtype=float)
    else:
        assert len(time_indices) == n

    obj_and_model = get_trend_objective_function_and_model(y_signal_dim, loss_type=trend_loss_type,
                                                           positive=False)
    base_model_s = obj_and_model['base_model']
    objective_fit_term = obj_and_model['objective_function']
    derv_1_expression = first_derv_nes_cvxpy(time_indices, base_model_s)
    reg_sum_expression, _ = get_trend_regularization(
        time_indices, base_model_s, derv_1_expression,
        l_norm_penalty=1, alpha1_penalty=0.0, alpha2_penalty=lambda_reg)

    total_objective_cvxpy = objective_fit_term + reg_sum_expression
    problem = cvxpy.Problem(cvxpy.Minimize(total_objective_cvxpy))

    if verbose:
        print(f"    TrendCalc: Solving for n={n}, lambda={lambda_reg}, loss='{trend_loss_type}'.")

    try:
        problem.solve(solver=solver)
        if problem.status not in [cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE]:
            if verbose: print(f"    TrendCalc WARNING: Solver {solver} finished with status {problem.status}.")
            if base_model_s.value is None:
                if verbose: print(f"    TrendCalc WARNING: base_model_s.value is None. Returning NaNs.")
                return np.full_like(y_signal_dim, np.nan)
            if verbose: print(
                f"    TrendCalc WARNING: Using potentially inaccurate solution from status {problem.status}.")
        trend_result = np.array(base_model_s.value) if base_model_s.value is not None else np.full_like(
            y_signal_dim,
            np.nan)  # Ensure result is array even if None
        if verbose and np.isnan(trend_result).any():
            print(f"    TrendCalc WARNING: Resulting trend contains NaNs after solve().")
        return trend_result
    except Exception as e:
        if verbose: print(
            f"    TrendCalc ERROR: Exception during solve with solver {solver}: {e}. Returning NaNs.")
        return np.full_like(y_signal_dim, np.nan)


def _cal_anoscore_internal(src_err_for_sample_dim: torch.Tensor, func: str = 'dirac', start_T: int = 0,
                           end_T: int = 0) -> float:
    anoscore = 0.0
    actual_start_T = max(0, start_T)
    actual_end_T = min(len(src_err_for_sample_dim), end_T)
    if actual_start_T >= actual_end_T: return 0.0
    if func == 'exp':
        for t in range(actual_start_T, actual_end_T):
            anoscore += np.exp(-np.square(t - (actual_end_T - 1))) * src_err_for_sample_dim[t].item()
    elif func == 'dirac':
        for t in range(actual_start_T, actual_end_T):
            anoscore += src_err_for_sample_dim[t].item()
    return anoscore


def _diff_series_single_sample_dim_internal(src_sample_dim: torch.Tensor, win_size: int = 1) -> torch.Tensor:
    src_np = src_sample_dim.cpu().numpy()
    length = len(src_np)
    if length <= win_size: return torch.tensor([], dtype=torch.float32)
    length_new = length - win_size
    src_new_np = np.zeros(length_new)
    for j in range(length_new):
        src_new_np[j] = src_np[j + win_size] - np.mean(src_np[j: j + win_size])
    return torch.tensor(np.abs(src_new_np), dtype=torch.float32)


def _plot_observed_and_trend_multidim_internal(
        observed_sample: torch.Tensor,
        trend_sample: torch.Tensor,
        error_tensor: np.ndarray,
        dim_specific_scores: list[float],  # 新增参数：每个维度的异常得分
        sample_name: str = "Sample"
):
    num_features = observed_sample.shape[1]
    seq_length = observed_sample.shape[0]
    time_steps = np.arange(seq_length)
    num_features = 2

    fig, axes = plt.subplots(num_features * 2, 1, figsize=(14, 4 * num_features * 2), sharex=True,
                             squeeze=False)  # squeeze=False确保axes总是2D
    axes = axes.flatten()

    fig.suptitle(f'{sample_name}: Observed, Trend, and Error (Multidimensional)', fontsize=16)

    for i in range(num_features):
        ax1_idx = i * 2
        ax2_idx = i * 2 + 1

        current_dim_score = dim_specific_scores[i]  # 获取当前维度的异常得分

        # Plot Observed vs Trend
        ax1 = axes[ax1_idx]
        ax1.plot(time_steps, observed_sample[:, i].cpu().numpy(), label=f'Observed', marker='o', linestyle='-',
                 markersize=4, alpha=0.7)
        trend_dim_data = trend_sample[:, i].cpu().numpy()
        if np.isnan(trend_dim_data).any():
            ax1.plot(time_steps, trend_dim_data, label=f'Trend (with NaN)', marker='x', linestyle=':',
                     markersize=4)
        else:
            ax1.plot(time_steps, trend_dim_data, label=f'Trend', marker='x', linestyle='--', markersize=4)
        ax1.set_ylabel(f'Value')
        ax1.legend()
        ax1.grid(True)
        # 为观测/趋势子图添加包含维度异常得分的副标题
        ax1.set_title(f"Dimension {i + 1} - Anomaly Score: {current_dim_score:.4f}", fontsize=10)

        # Plot Error (|Obs - Trend|)
        ax2 = axes[ax2_idx]
        ax2.bar(time_steps, error_tensor[:, i], label=f'Error (|Obs-Trend|)', color='gray', alpha=0.7)
        ax2.set_ylabel(f'Error')
        ax2.legend()
        ax2.grid(True)
        # 为误差子图添加副标题（可选，或与上面合并）
        ax2.set_title(f"Dimension {i + 1} - Error Distribution", fontsize=10)

    axes[-1].set_xlabel('Time Step')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # plt.show()
    plt.savefig(f'./plot_results/{sample_name}.png')


# --- 主函数 ---
def process_batch_and_get_anomaly_scores(
        observed_batch_md: torch.Tensor,
        plot_first_n_samples: int = 0,
        verbose: bool = False
) -> tuple[list[float], np.ndarray]:
    lambda_reg_trend = 0.3
    lay_back = 1
    score_func = 'dirac'
    trend_loss_type = 'mae'
    anoscore_base_type = 'trendfilter'
    cvxpy_solver = 'ECOS'
    anomaly_score_agg_method = 'mean'

    batch_size, seq_length, num_features = observed_batch_md.shape

    if verbose:
        print("--- Running RobustTSF Anomaly Score Calculation for a Batch ---")
        print(f"  Input Batch Shape: ({batch_size}, {seq_length}, {num_features})")
        print(f"  Hyperparameters (fixed as per paper/defaults for each sample):")
        print(f"    lambda_reg_trend: {lambda_reg_trend}, lay_back: {lay_back}, score_func: '{score_func}'")
        print(
            f"    trend_loss_type: '{trend_loss_type}', anoscore_base_type: '{anoscore_base_type}', agg_method: '{anomaly_score_agg_method}'")
        print("-" * 30)

    batch_anomaly_scores_aggregated = []  # 存储每个样本聚合后的异常得分
    batch_error_tensors_list_np = []  # 存储每个样本的 (seq_length, num_features) 误差 NumPy 数组

    for sample_idx in range(batch_size):
        # if verbose:
        if sample_idx % 50 == 0:
            print(f"\nProcessing Sample {sample_idx + 1}/{batch_size} in Batch...")
        observed_sample_md = observed_batch_md[sample_idx, :, :]

        if verbose: print("  Calculating actual trend for each dimension of the current sample...")
        calculated_trend_features_list_np = []
        time_indices_for_trend = np.arange(seq_length, dtype=float)
        for i in range(num_features):
            y_signal_dim_np = observed_sample_md[:, i].cpu().numpy()
            if verbose: print(f"    Calculating trend for Dim {i + 1} of Sample {sample_idx + 1}...")
            trend_dim_np = _calculate_actual_trend_for_dimension_internal(
                y_signal_dim_np, lambda_reg=lambda_reg_trend, trend_loss_type=trend_loss_type,
                time_indices=time_indices_for_trend, solver=cvxpy_solver, verbose=verbose
            )
            if np.isnan(trend_dim_np).any():
                if verbose: print(
                    f"    Warning: Trend for Dim {i + 1} (Sample {sample_idx + 1}) has NaNs. Using observed signal as fallback.")
                calculated_trend_features_list_np.append(y_signal_dim_np.copy())
            else:
                calculated_trend_features_list_np.append(trend_dim_np)
        trend_sample_md_np = np.stack(calculated_trend_features_list_np, axis=1)
        trend_sample_md = torch.from_numpy(trend_sample_md_np).float()


        if verbose: print(
            f"\n  Calculating anomaly score A(x_n) and collecting error series for Sample {sample_idx + 1}...")
        current_sample_all_dim_individual_scores = []  # 存储当前样本每个维度的 pscore
        current_sample_all_err_for_score_dim_np = []

        if verbose: print(
            f"\n    --- Debugging Anomaly Score Calculation per Dimension (Sample {sample_idx + 1}) ---")
        for dim_idx in range(num_features):
            if verbose: print(f"      Processing Dimension: {dim_idx + 1}")
            obs_dim = observed_sample_md[:, dim_idx]
            trend_dim = trend_sample_md[:, dim_idx]
            trend_dim = trend_dim.to(obs_dim.device)

            if verbose:
                print(f"        Dim {dim_idx + 1} - Observed (last 3): {obs_dim[-3:].cpu().numpy()}")
                print(f"        Dim {dim_idx + 1} - Trend    (last 3): {trend_dim[-3:].cpu().numpy()}")

            if anoscore_base_type == 'trendfilter':
                err_for_score_dim = torch.abs(obs_dim - trend_dim)
            elif anoscore_base_type == 'diff':
                current_win_size = 1
                err_for_score_dim = _diff_series_single_sample_dim_internal(obs_dim, win_size=current_win_size)
            elif anoscore_base_type == 'STL':
                err_for_score_dim = torch.abs(obs_dim - trend_dim)
            else:
                raise ValueError(f"Unknown anoscore_base_type: {anoscore_base_type}")
            current_sample_all_err_for_score_dim_np.append(err_for_score_dim.cpu().numpy())
            if verbose: print(
                f"        Dim {dim_idx + 1} - err_for_score_dim (last 3 points): {err_for_score_dim[-3:].cpu().numpy()}")

            dim_pscore = 0.0
            if len(err_for_score_dim) > 0:
                seq_length_for_score = len(err_for_score_dim)
                end_T_score = seq_length_for_score
                start_T_score = end_T_score - lay_back
                if start_T_score < 0: start_T_score = 0
                if verbose: print(
                    f"        Dim {dim_idx + 1} - _cal_anoscore_internal params: start_T={start_T_score}, end_T={end_T_score}")
                if start_T_score < end_T_score:
                    error_values_for_scoring = err_for_score_dim[start_T_score:end_T_score]
                    if verbose: print(
                        f"        Dim {dim_idx + 1} - Error value(s) considered: {error_values_for_scoring.cpu().numpy()}")
                    dim_pscore = _cal_anoscore_internal(err_for_score_dim, func=score_func,
                                                        start_T=start_T_score,
                                                        end_T=end_T_score)
                elif verbose:
                    print(
                        f"        Dim {dim_idx + 1} - Invalid range for _cal_anoscore_internal, dim_pscore = 0.0")
            elif verbose:
                print(f"      Dim {dim_idx + 1} - err_for_score_dim is empty, dim_pscore = 0.0")
            if verbose: print(f"        Dim {dim_idx + 1} - Calculated dim_pscore: {dim_pscore:.4f}")
            current_sample_all_dim_individual_scores.append(dim_pscore)
        if verbose: print(
            f"    --- End Debugging Anomaly Score Calculation per Dimension (Sample {sample_idx + 1}) ---")

        pscore_current_sample_aggregated = 0.0
        if current_sample_all_dim_individual_scores:
            if anomaly_score_agg_method == 'mean':
                pscore_current_sample_aggregated = np.mean(current_sample_all_dim_individual_scores)
            elif anomaly_score_agg_method == 'max':
                pscore_current_sample_aggregated = np.max(current_sample_all_dim_individual_scores)
            else:
                raise ValueError(f"Unknown anomaly_score_agg_method: {anomaly_score_agg_method}")

        batch_anomaly_scores_aggregated.append(float(pscore_current_sample_aggregated))
        current_sample_full_error_tensor_np = np.stack(current_sample_all_err_for_score_dim_np, axis=1)
        batch_error_tensors_list_np.append(current_sample_full_error_tensor_np)

        if verbose:
            print(f"  Aggregated pscore for Sample {sample_idx + 1}: {pscore_current_sample_aggregated:.4f}")
            print(
                f"  Shape of collected error tensor for Sample {sample_idx + 1}: {current_sample_full_error_tensor_np.shape}")

        if plot_first_n_samples > 0 and sample_idx < plot_first_n_samples:
            if verbose: print(f"\n  Plotting observed vs. calculated trend for Sample {sample_idx + 1}...")
            _plot_observed_and_trend_multidim_internal(
                observed_sample_md,
                trend_sample_md,
                current_sample_full_error_tensor_np,
                current_sample_all_dim_individual_scores,  # 传递每个维度的得分
                sample_name=f"Sample_{sample_idx + 1}_{batch_size}_OverallAggScore_{pscore_current_sample_aggregated:.2f}"
            )

    full_batch_error_tensor_np = np.stack(batch_error_tensors_list_np, axis=0)
    if verbose: print("\nFunction finished processing batch.")
    return batch_anomaly_scores_aggregated, full_batch_error_tensor_np


# 计算trend ----------------------结束--------------------------------


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader


    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # def _select_criterion(self):
    #     criterion = nn.MSELoss()
    #     return criterion
    def _select_criterion(self):
        # 【Modified】：当使用概率模型时，根据用户选择的损失函数决定 criterion
        if self.args.probabilistic:
            loss_func = self.args.probabilistic_loss_function
            if loss_func == "torchGaussianNLL":
                def gaussian_nll_loss_wrapper(pred, true):
                    # pred 是一个二元组 [mu, sigma]
                    mu, sigma = pred
                    # 确保sigma不小于eps
                    sigma = torch.clamp(sigma, min=1e-6)
                    # 注意：torch.nn.GaussianNLLLoss要求输入: mean, target, variance (sigma^2)
                    # 这里设置 reduction="mean" 且 eps=1e-6
                    loss_fn = torch.nn.GaussianNLLLoss(eps=1e-6, reduction="mean")
                    # loss_fn = torch.nn.GaussianNLLLoss(eps=1e-6, reduction="mean")
                    # loss = loss_fn(mu, true, sigma ** 2)
                    return loss_fn(mu, true, sigma ** 2)
                criterion = gaussian_nll_loss_wrapper

            elif loss_func == "CRPS":
                # 采用高斯CRPS的闭式解: CRPS(mu, sigma, y) = sigma * [1/sqrt(pi) - 2*phi(z) - z*(2*Phi(z)-1)]
                def gaussian_crps_loss(pred, true):
                    mu, sigma = pred
                    sigma = torch.clamp(sigma, min=1e-6)  # 防止sigma过小
                    z = (true - mu) / sigma
                    pdf = 1.0 / math.sqrt(2 * math.pi) * torch.exp(-0.5 * z**2)
                    cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
                    crps = sigma * (1/math.sqrt(math.pi) - 2 * pdf - z * (2 * cdf - 1))
                    return crps.mean()
                criterion = gaussian_crps_loss
            else:
                def custom_gaussian_nll_loss(pred, true):
                    mu, sigma = pred
                    sigma = torch.clamp(sigma, min=1e-6)  # 不允许sigma过小
                    # NLL loss: 0.5*log(2*pi*sigma^2) + ((y-mu)^2)/(2*sigma^2)
                    loss = 0.5 * torch.log(2 * math.pi * sigma**2) + ((true - mu)**2) / (2 * sigma**2)
                    return loss.mean()
                criterion = custom_gaussian_nll_loss
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        MSE_loss = []
        MAE_loss = []
        criterion_MSE = nn.MSELoss(reduction='none')
        self.model.eval()

        sum_mse = torch.tensor(0.0, device=self.device)
        sum_mae = torch.tensor(0.0, device=self.device)
        total_samples = 0

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, sample_id, sample_weight) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if self.args.probabilistic:
                    # 【Modified】处理概率模式：提取 mu, sigma，并取最后 pred_len 时间步
                    mu, sigma = outputs
                    mu = mu[:, -self.args.pred_len:, :]
                    sigma = sigma[:, -self.args.pred_len:, :]
                    loss = criterion([mu, sigma], batch_y[:, -self.args.pred_len:, :])
                    loss_mse = criterion_MSE(mu, batch_y[:, -self.args.pred_len:, :])
                    loss_mae = torch.mean(torch.abs(mu - batch_y[:, -self.args.pred_len:, :]))

                else:
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss_mse = criterion_MSE(outputs, batch_y)
                    loss_mse = loss_mse.mean(dim=(1, 2))
                    loss_mae = torch.abs(outputs - batch_y)
                    loss_mae = loss_mae.mean(dim=(1, 2))
                    # loss = loss_mse
                    # loss = criterion(outputs, batch_y)

                    sum_mse += loss_mse.sum()
                    sum_mae += loss_mae.sum()
                    total_samples += loss_mse.numel()  # B

                # 5) 在 GPU 上算平均
                avg_mse = sum_mse / total_samples
                avg_mae = sum_mae / total_samples


                # total_loss.append(loss.item())

                # MSE_loss.extend(loss_mse.tolist())
                # MAE_loss.extend(loss_mae.tolist())

        # total_loss = np.average(total_loss)
        MSE_loss = avg_mse.cpu().numpy()
        MAE_loss = avg_mae.cpu().numpy()
        total_loss = MSE_loss
        self.model.train()
        # 1. MSE loss, 2. MAE, 3. pre-selected loss,
        return MSE_loss, MAE_loss, total_loss

    def infer_train_set(self, vali_data, vali_loader, criterion, iterations, setting):

        criterion_MSE = nn.MSELoss(reduction='none')
        loss_all_sample_all_variable_all_token = np.zeros((len(vali_data), self.args.pred_len, self.args.enc_in))
        trend_all_sample_all_variable_all_token = np.zeros((len(vali_data), self.args.pred_len, self.args.enc_in))
        trend_ano_score_all_sample = np.zeros((len(vali_data), 2))
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, sample_id, sample_weight) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if self.args.probabilistic:
                    # 【Modified】处理概率模式：提取 mu, sigma，并取最后 pred_len 时间步
                    mu, sigma = outputs
                    mu = mu[:, -self.args.pred_len:, :]
                    sigma = sigma[:, -self.args.pred_len:, :]
                    loss = criterion([mu, sigma], batch_y[:, -self.args.pred_len:, :])

                else:
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss_mse = criterion_MSE(outputs, batch_y)
                    sample_loss_metric = loss_mse.detach().cpu().numpy()

                    top_index = [33625, 33534, 33626, 33617, 33536, 20368, 33525, 20451, 20452,
                                 20458, 14857, 14948, 14946, 14934, 14947, 14933, 14945, 14937, 14935,
                                 14940]
                    bottom_index = [27941, 27940, 27939, 27942, 27943, 8000, 8001, 7999, 7998, 4890]

                    for j, sid in enumerate(sample_id):
                        if sid in top_index:
                            np.save(f'/home/local/zi/research_project/iTransformer/sample_label_plot/sample_{sid}.npy',
                                    batch_y[j].cpu().numpy())
                            print(f'top index: {sid}')
                        elif sid in bottom_index:
                            np.save(f'/home/local/zi/research_project/iTransformer/sample_label_plot/sample_{sid}.npy',
                                    batch_y[j].cpu().numpy())
                            print(f'bottom index: {sid}')

                    cal_save_trend = False
                    if cal_save_trend:
                        # --- 调用trend 计算函数 ---
                        batch_scores, batch_errors = process_batch_and_get_anomaly_scores(
                            batch_y,
                            plot_first_n_samples=0,
                            verbose=False,# False, True
                        )

                        for j, sid in enumerate(sample_id):  # 遍历当前batch的样本id
                            loss_all_sample_all_variable_all_token[int(sid)] = sample_loss_metric[j]
                            # 保存trend 值
                            trend_all_sample_all_variable_all_token[int(sid)] = batch_errors[j]
                            trend_ano_score_all_sample[int(sid),0] = int(sid)
                            trend_ano_score_all_sample[int(sid),1] = batch_scores[j]


                    print(f'finish batch: {i}')


        # epoch_results_path = os.path.join("/mnt/ssd/zi/itransformer_results/new/", setting)
        epoch_results_path_base = os.path.join("/mnt/ssd/zi/itransformer_results/trend_scores/", setting)
        if not os.path.exists(epoch_results_path_base):
            os.makedirs(epoch_results_path_base)
        epoch_results_path = os.path.join(epoch_results_path_base,
                                          f'iter_{iterations}_train_set_all_sample_all_tokens.npy')
        np.save(epoch_results_path, loss_all_sample_all_variable_all_token)

        epoch_results_path = os.path.join(epoch_results_path_base,
                                          f'trend_error_train_set_all_sample_all_tokens.npy')

        np.save(epoch_results_path, trend_all_sample_all_variable_all_token)

        epoch_results_path = os.path.join(epoch_results_path_base,
                                          f'trend_anomaly_score_train_set_all_sample.npy')
        np.save(epoch_results_path, trend_ano_score_all_sample)


        self.model.train()
        # 1. MSE loss, 2. MAE, 3. pre-selected loss,


    def train(self, setting):

        if self.args.pruning_method in (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19):

            train_data, _ = self._get_data(flag='train')

            steps_per_epoch = len(train_data) // self.args.batch_size  # 丢弃不能填充完整 batch 的数据
            total_epoch = self.args.train_iterations / steps_per_epoch if steps_per_epoch > 0 else 0
            total_epoch = math.ceil(total_epoch)

            train_data = InfoBatch(train_data, total_epoch,
                                   prune_ratio=self.args.pruning_rate,
                                   delta=self.args.infobatch_delta,
                                   args=self.args)

            from torch.utils.data import DataLoader
            train_loader = DataLoader(
                train_data,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,  # 让数据固定在内存，加快 GPU 访问
                prefetch_factor=2,  # 提前加载数据
                drop_last=True,
                sampler=train_data.sampler)
        else:
            train_data, train_loader = self._get_data(flag='train')

        steps_per_epoch = len(train_data) // self.args.batch_size  # 丢弃不能填充完整 batch 的数据
        total_epoch = self.args.train_iterations / steps_per_epoch if steps_per_epoch > 0 else 0
        total_epoch = math.ceil(total_epoch)

        infer_data, infer_loader = self._get_data(flag='infer')

        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        if self.args.pruning_method in (100, 101):
            # data_shapley_scores = torch.zeros(len(train_data), device=self.device)
            data_shapley_scores = np.zeros((len(train_data), total_epoch, 4))

            all_vali_data, all_vali_loader = self._get_data(flag='all_val')
            val_batch_x, val_batch_y, val_batch_x_mark, val_batch_y_mark, val_sample_id, val_sample_weight = next(iter(all_vali_loader))
            val_batch_x = val_batch_x.float().to(self.device)
            val_batch_y = val_batch_x.float().to(self.device)

            if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                val_batch_x_mark = None
                val_batch_y_mark = None
            else:
                val_batch_x_mark = val_batch_x_mark.float().to(self.device)
                val_batch_y_mark = val_batch_y_mark.float().to(self.device)



        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        criterion_train = nn.MSELoss(reduction='none')

        if self.args.save_per_sample_loss:
            # if self.args.data_path=='electricity.csv':
            loss_all_epoch = np.zeros((len(train_data), total_epoch+1))
            loss_all_epoch[:, 0] = np.arange(len(train_data))  # 【Modified】保存样本id到第一列
            attention_all_epoch = np.zeros((len(train_data), self.args.train_epochs + 1, 15000))
            loss_all_epoch_all_variable_all_stamp = np.zeros((len(train_data), total_epoch, self.args.pred_len, self.args.enc_in))
            # attention_all_epoch = np.zeros((len(train_data), total_epoch, 1875))


        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        global_step = 0
        use_early_stop = False
        # for epoch in range(self.args.train_epochs):

        for epoch in range(total_epoch):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            if self.args.pruning_method in (100, 101):
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, sample_id, sample_weight) in enumerate(
                        train_loader):
                    iter_count += 1
                    global_step += 1
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    batch_x = torch.cat((batch_x, val_batch_x), dim=0)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y_proc = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    batch_y_proc = torch.cat((batch_y_proc, val_batch_y), dim=0)

                    if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                        batch_x_mark = None
                        batch_y_mark = None
                    else:
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)

                    batch_x_mark = torch.cat((batch_x_mark, val_batch_x_mark), dim=0)

                    # zero the model gradient
                    # only use val for back/forward pass
                    model_optim.zero_grad()
                    outputs_val_only = self.model(val_batch_x, val_batch_x_mark, 0, 0)
                    loss_val_only = criterion(outputs_val_only, val_batch_y)

                    # single-sample gradient
                    loss_val_only.backward()
                    grad_val = {}

                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            grad_val[name] = param.grad.detach().clone()

                    grad_val_list = [grad_val[name] for name, param in self.model.named_parameters()
                                     if param.grad is not None and name in grad_val]
                    flat_grad_val = parameters_to_vector(grad_val_list)
                    l2_norm_flat_val_grad = torch.linalg.norm(flat_grad_val)

                    # only calculate train loss
                    outputs = self.model(batch_x[:self.args.batch_size], batch_x_mark[:self.args.batch_size], 0, 0)
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]

                    train_batch_loss = criterion_train(outputs, batch_y_proc[:self.args.batch_size])
                    train_batch_loss = train_batch_loss.mean(dim=(1, 2))

                    # Now let's compute the "dot product" for each sample in the train batch:
                    for batch_idx in range(self.args.batch_size):
                        # zero grad
                        param_zero = {}
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                param.grad.zero_()


                        # backward on just the i-th example in the train batch
                        # single_logit = self.model(batch_x[batch_idx].unsqueeze(0), batch_x_mark[batch_idx].unsqueeze(0), 0, 0)
                        # single_loss = criterion(single_logit, batch_y_proc[batch_idx])
                        # single_loss.backward()



                        train_batch_loss[batch_idx].backward(retain_graph=True)

                        # grad_train_i_tuple = torch.autograd.grad(train_batch_loss[batch_idx], list(self.model.parameters()), retain_graph=True,
                        #                                          create_graph=False)


                        # # read off dot-product with grad_val
                        # dot_val = 0.0
                        # for name, param in self.model.named_parameters():
                        #     if param.grad is not None and name in grad_val:
                        #         # flatten both param.grad and grad_val[name]
                        #         dot_val += (param.grad.view(-1) * grad_val[name].view(-1)).sum()

                        # 取出所有 param.grad；注意顺序要和 grad_val 一致
                        grad_list = [param.grad for name, param in self.model.named_parameters()
                                     if param.grad is not None and name in grad_val]
                        # 初始化本次迭代的度量值
                        cosine_similarity_value = torch.tensor(0.0, device=self.device)
                        actual_dot_product_value = torch.tensor(0.0, device=self.device)
                        l2_norm_flat_train_grad_value = torch.tensor(0.0, device=self.device)

                        if grad_list:
                            current_flat_grad_train_i = parameters_to_vector(grad_list)

                            # dot_val = torch.dot(flat_grad, flat_grad_val)
                            # cal cosine similarity
                            # dot_val = F.cosine_similarity(flat_grad.unsqueeze(0), flat_grad_val.unsqueeze(0), dim=1)
                            # 1. 计算 L2 Norm (训练梯度)
                            l2_norm_flat_train_grad_value = torch.linalg.norm(current_flat_grad_train_i)

                            # 2. 计算实际的点积
                            actual_dot_product_value = torch.dot(current_flat_grad_train_i, flat_grad_val)

                            # 3. 手动计算余弦相似度
                            denominator = l2_norm_flat_train_grad_value * l2_norm_flat_val_grad
                            # 添加一个小的 epsilon 防止除以零
                            if denominator > 1e-9:
                                cosine_similarity_value = actual_dot_product_value / denominator
                            # else: 保持为 0.0

                        # 获取样本在完整数据集中的索引
                        idx_in_full_dataset = sample_id[batch_idx]

                        # 存储原始的 Shapley 分数 (基于缩放后的余弦相似度)
                        data_shapley_scores[idx_in_full_dataset, epoch, 0] = 1.0 * l2_norm_flat_train_grad_value.item()
                        data_shapley_scores[idx_in_full_dataset, epoch, 1] = 1.0 * l2_norm_flat_val_grad.item()
                        data_shapley_scores[idx_in_full_dataset, epoch, 2] = 100.0 * actual_dot_product_value.item()
                        data_shapley_scores[idx_in_full_dataset, epoch, 3] = 100.0 * cosine_similarity_value.item()


                        # # Add to data_shapley_scores
                        # idx_in_full_dataset = sample_id[batch_idx]  # approximate global index
                        # # data_shapley_scores[idx_in_full_dataset, epoch] = -1 * dot_val.item()  # mul by lr to scale back things
                        # data_shapley_scores[idx_in_full_dataset, epoch] = 100.0*dot_val.item()  # mul by lr to scale back things
                        # # We used lr=0.01 -> so multiply by -lr. Negative sign because the loss change is -( grad_val · grad_train_i ).





                    train_batch_loss = train_batch_loss.mean()
                    train_loss.append(train_batch_loss.item())


                    # Ghost dot product for backward pass
                    model_optim.zero_grad()
                    # retain graph since we need more of the backprop later on
                    train_batch_loss.backward()

                    # saved_state = {}  # save the current state
                    #
                    # # We store the current gradients (which are for train+val) in saved_state.
                    # for name, param in self.model.named_parameters():
                    #     if param.grad is not None:
                    #         # save param.grad clone for later use
                    #         saved_state[name] = param.grad.detach().clone()
                    # # restore the combined grad from saved_state
                    # for name, param in self.model.named_parameters():
                    #     if name in saved_state:
                    #         param.grad = saved_state[name]  # now when it is in param save it to saved_state

                    # Final logging
                    model_optim.step()



            #
            #         # ------------previous---------------------------------------------------------------------------------
            #         combined_loss = criterion(outputs, batch_y_proc)
            #
            #
            #         # Ghost dot product for backward pass
            #         model_optim.zero_grad()
            #         # retain graph since we need more of the backprop later on
            #         combined_loss.backward(retain_graph=True)
            #
            #         saved_state = {}  # save the current state
            #
            #         # We store the current gradients (which are for train+val) in saved_state.
            #         for name, param in self.model.named_parameters():
            #             if param.grad is not None:
            #                 # save param.grad clone for later use
            #                 saved_state[name] = param.grad.detach().clone()
            #
            #         # zero the model gradient
            #         # only use val for back/forward pass
            #         model_optim.zero_grad()
            #         outputs_val_only = self.model(val_batch_x, val_batch_x_mark, 0, 0)
            #         loss_val_only = criterion(outputs_val_only, val_batch_y)
            #
            #         # single-sample gradient
            #         loss_val_only.backward(create_graph=False)
            #         grad_val = {}
            #
            #         for name, param in self.model.named_parameters():
            #             if param.grad is not None:
            #                 grad_val[name] = param.grad.detach().clone()
            #
            #         grad_val_list = [grad_val[name] for name, param in self.model.named_parameters()
            #                          if param.grad is not None and name in grad_val]
            #         flat_grad_val = parameters_to_vector(grad_val_list)
            #
            #         # Now let's compute the "dot product" for each sample in the train batch:
            #         for batch_idx in range(self.args.batch_size):
            #             # zero grad
            #             param_zero = {}
            #             for name, param in self.model.named_parameters():
            #                 if param.grad is not None:
            #                     param.grad.zero_()
            #
            #             # backward on just the i-th example in the train batch
            #             single_logit = self.model(batch_x[batch_idx].unsqueeze(0), batch_x_mark[batch_idx].unsqueeze(0), 0, 0)
            #             single_loss = criterion(single_logit, batch_y_proc[batch_idx])
            #             single_loss.backward()
            #
            #
            #             # # read off dot-product with grad_val
            #             # dot_val = 0.0
            #             # for name, param in self.model.named_parameters():
            #             #     if param.grad is not None and name in grad_val:
            #             #         # flatten both param.grad and grad_val[name]
            #             #         dot_val += (param.grad.view(-1) * grad_val[name].view(-1)).sum()
            #
            #             # 取出所有 param.grad；注意顺序要和 grad_val 一致
            #             grad_list = [param.grad for name, param in self.model.named_parameters()
            #                          if param.grad is not None and name in grad_val]
            #
            #             if grad_list:
            #                 flat_grad = parameters_to_vector(grad_list)
            #                 # dot_val = torch.dot(flat_grad, flat_grad_val)
            #                 # cal cosine similarity
            #                 dot_val = F.cosine_similarity(flat_grad.unsqueeze(0), flat_grad_val.unsqueeze(0), dim=1)
            #
            #             else:
            #                 dot_val = torch.tensor(0.0, device=next(self.model.parameters()).device)
            #
            #             # Add to data_shapley_scores
            #             idx_in_full_dataset = sample_id[batch_idx]  # approximate global index
            #             # data_shapley_scores[idx_in_full_dataset, epoch] = -1 * dot_val.item()  # mul by lr to scale back things
            #             data_shapley_scores[idx_in_full_dataset, epoch] = 100.0*dot_val.item()  # mul by lr to scale back things
            #
            #             # We used lr=0.01 -> so multiply by -lr. Negative sign because the loss change is -( grad_val · grad_train_i ).
            #
            #         # restore the combined grad from saved_state
            #         for name, param in self.model.named_parameters():
            #             if name in saved_state:
            #                 param.grad = saved_state[name]  # now when it is in param save it to saved_state
            #
            #         # Final logging
            #         model_optim.step()
                    # ------------previous---------------------------------------------------------------------------------


                    if (i + 1) % 1 == 0:
                        # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()





            else:
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, sample_id, sample_weight) in enumerate(train_loader):
                    iter_count += 1
                    global_step += 1
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                        batch_x_mark = None
                        batch_y_mark = None
                    else:
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    # encoder - decoder
                    if self.args.use_amp:
                        pass
                    else:
                        if self.args.output_attention:
                            outputs, attentions = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        if self.args.probabilistic:
                            # 【Modified】提取概率模式输出
                            mu, sigma = outputs
                            mu = mu[:, -self.args.pred_len:, f_dim:]
                            sigma = sigma[:, -self.args.pred_len:, f_dim:]
                            batch_y_proc = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = criterion([mu, sigma], batch_y_proc)
                        else:
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y_proc = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                            # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                            if self.args.pruning_method in (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19):

                                loss = criterion_train(outputs, batch_y_proc)

                                if self.args.pruning_method == 5:
                                    loss = loss.mean(dim=(1, 2))
                                    loss = train_data.update(-loss)
                                    loss = -loss


                                elif self.args.pruning_method in (9, 10, 11, 12, 16, 18, 19):
                                    loss = train_data.update(loss)

                                # 第一个epoch不计算weighted loss值 只记录当前的loss
                                elif self.args.pruning_method in (13, 14, 15, 17):
                                    if epoch == 0:
                                        loss = train_data.update(loss, only_update_saved_loss_metric=True)
                                    else:
                                        loss = train_data.update(loss, only_update_saved_loss_metric=False)


                                else:
                                    loss = loss.mean(dim=(1, 2))
                                    loss = train_data.update(loss)
                            else:
                                loss = criterion(outputs, batch_y_proc)

                        # else:
                        #     if self.args.probabilistic:
                        #         loss = criterion([mu, sigma], batch_y_proc)
                        #     else:
                        #         loss = criterion(outputs, batch_y_proc)

                        train_loss.append(loss.item())
                        # 【Modified】记录每个样本的loss
                        if self.args.save_per_sample_loss:
                            sample_loss_metric = ((outputs - batch_y_proc) ** 2)
                            sample_loss = (sample_loss_metric.mean(dim=[1,2])).detach().cpu().numpy()
                            sample_loss_metric = sample_loss_metric.detach().cpu().numpy()# 计算每个样本的MSE
                            for j, sid in enumerate(sample_id):  # 遍历当前batch的样本id
                                loss_all_epoch[int(sid), epoch+1] = sample_loss[j]
                                loss_all_epoch_all_variable_all_stamp[int(sid), epoch]=sample_loss_metric[j]


                                if self.args.output_attention:
                                    attention_all_epoch[int(sid), epoch] = torch.cat([item.view(32, -1) for item in attentions], dim=1).detach().cpu().numpy()[j]
                                    # attention_all_epoch[int(sid), epoch] = torch.cat([item.mean(dim=1).view(32, -1) for item in attentions], dim=1).detach().cpu().numpy()[j]
                                # 保存loss到对应epoch的列

                    if (global_step) % 100 == 0:

                        # self.infer_train_set(infer_data, infer_loader, criterion, global_step, setting)


                        vali_MSE_loss, vali_MAE_loss, vali_total_loss = self.vali(vali_data, vali_loader, criterion)
                        test_MSE_loss, test_MAE_loss, test_total_loss = self.vali(test_data, test_loader, criterion)


                        iteration_results_path = os.path.join(self.args.checkpoints, setting, f'iter_{global_step}_results.npy')

                        np.save(iteration_results_path, np.array([global_step,
                                                              epoch + 1, 0, vali_MSE_loss, vali_MAE_loss,
                                                              vali_total_loss,
                                                              test_MSE_loss, test_MAE_loss, test_total_loss]))
                        print("\titers: {0}, epoch: {1} | loss: {2:.6f}, val loss: {3:.6f}, test loss: {4:.6f}"
                              .format(global_step, epoch, loss, vali_total_loss, test_total_loss))

                    if (i + 1) % 300 == 0:
                        # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)



            print(f'Saving data_shapley_scores_epoch_{epoch}.npy')
            np.save(os.path.join(self.args.checkpoints, setting, f'data_shapley_l2_dot_cos_scores_epoch_{epoch}.npy'),
                    data_shapley_scores)

            # # 1. MSE loss, 2. MAE, 3. pre-selected loss,

            vali_MSE_loss, vali_MAE_loss, vali_total_loss = self.vali(vali_data, vali_loader, criterion)
            test_MSE_loss, test_MAE_loss, test_total_loss = self.vali(test_data, test_loader, criterion)


            epoch_results_path = os.path.join(self.args.checkpoints, setting, f'epoch_{epoch}_results.npy')

            np.save(epoch_results_path, np.array([global_step,
                                                  epoch + 1, train_loss, vali_MSE_loss, vali_MAE_loss, vali_total_loss,
                                                  test_MSE_loss, test_MAE_loss, test_total_loss]))

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_total_loss, test_total_loss))


            # vali_MSE_loss, vali_MAE_loss, vali_total_loss = self.vali(vali_data, vali_loader, criterion)
            # test_MSE_loss, test_MAE_loss, test_total_loss = self.vali(test_data, test_loader, criterion)
            #
            # epoch_results_path = os.path.join(self.args.checkpoints, setting, f'epoch_{epoch}_results.npy')
            #
            # np.save(epoch_results_path, np.array([global_step,
            #                                       epoch + 1, train_loss, vali_MSE_loss, vali_MAE_loss, vali_total_loss,
            #                                       test_MSE_loss, test_MAE_loss, test_total_loss]))
            #
            # print("Epoch: {0}, Steps: {1} | loss: {2:.6f}, val loss: {3:.6f}, test loss: {4:.6f}"
            #       .format(global_step, epoch, train_loss, vali_total_loss, test_total_loss))





            # 1,

            # if self.args.output_attention:
            #     import matplotlib.pyplot as plt
            #
            #     # 假设 result 是你的 tensor 列表
            #     result = attentions  # 这里需要替换为你的实际数据
            #
            #     # 根目录
            #     output_root = "output_images"
            #     os.makedirs(output_root, exist_ok=True)
            #
            #     # 遍历 list，分别保存每个 batch 的 25x25 图片
            #     for list_idx, tensor in enumerate(result):
            #         tensor = tensor.cpu().detach()  # 确保 tensor 在 CPU 上
            #         for batch_idx in range(tensor.shape[0]):  # 32 batches
            #             # 选取一个通道的 25x25 图片（这里选择通道平均）
            #             image = tensor[batch_idx].mean(dim=0).numpy()  # 计算 8 个通道的均值
            #
            #             # 也可以选择单个通道，例如通道 0
            #             # image = tensor[batch_idx, 0].numpy()
            #
            #             # 创建画布
            #             fig, ax = plt.subplots()
            #             img = ax.imshow(image, cmap="gray")  # 以灰度图显示
            #             plt.axis("off")  # 不显示坐标轴
            #
            #             # 添加 color bar
            #             cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
            #             cbar.ax.tick_params(labelsize=8)  # 调整 color bar 标签大小
            #
            #             # 构造文件名
            #             filename = os.path.join(output_root, f"layer{list_idx}_epoch{epoch + 1}_batch{batch_idx}.png")
            #             plt.savefig(filename, bbox_inches="tight", pad_inches=0.1, dpi=300)
            #             plt.close()
            #
            #     print("图片保存完成！")

            if use_early_stop:
                early_stopping(vali_total_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        if self.args.save_per_sample_loss:
            print("Saving model weights")
            torch.save(self.model.state_dict(), path + '/' + f'checkpoint_epoch_{epoch}_seed_{self.args.seed}.pth')

            print(f'Saving loss_all_epoch_seed_{self.args.seed}.npy')
            np.save(os.path.join(self.args.checkpoints, setting, f'loss_all_epoch_seed_{self.args.seed}.npy'),
                    loss_all_epoch)

            print(f'Saving attention_all_epoch_seed_{self.args.seed}.npy')
            np.save(os.path.join(self.args.checkpoints, setting, f'attention_all_epoch_seed_{self.args.seed}.npy'),
                    attention_all_epoch)

            print(f'Saving loss_all_epoch_all_variable_all_stamp{self.args.seed}.npy')
            np.save(os.path.join(self.args.checkpoints, setting, f'loss_all_epoch_all_variable_all_stamp{self.args.seed}.npy'),
                    loss_all_epoch_all_variable_all_stamp)


        # best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, sample_id, sample_weight) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                if self.args.probabilistic:
                    # 【Modified】在概率模式下，使用预测均值作为点预测
                    mu, _ = outputs
                    outputs = mu[:, -self.args.pred_len:, :]
                else:
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #         shape = input.shape
                #         input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, sample_id, sample_weight) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return


class Exp_Long_Term_Forecast_GRPO(Exp_Basic):
    def __init__(self, args):
        """
        初始化实验类，加载基础设置并构建GRPO的选择网络。
        """
        super(Exp_Long_Term_Forecast_GRPO, self).__init__(args)
        # 样本总数，用于选择网络输入输出维度
        self.num_samples = len(self.train_data)
        hidden_dim = args.selector_hidden_dim
        # 选择网络：输入每个样本的历史损失，输出每个样本被选中概率
        self.selector_net = nn.Sequential(
            nn.Linear(1, hidden_dim),  # 每个样本输入其历史损失值
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 输出每个样本的优先级得分
            nn.Sigmoid()  # 输出选择概率
        ).to(self.device)
        # 选择网络的优化器
        self.optimizer_selector = optim.Adam(
            self.selector_net.parameters(),
            lr=args.selector_lr
        )

    def train(self, setting):
        """
        训练函数：整合GRPO策略与iTransformer训练
        """
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        train_loader = self._get_data_loader(
            self.train_data,
            batch_size=self.args.batch_size,
            shuffle=True
        )
        vali_loader = self._get_data_loader(
            self.vali_data,
            batch_size=self.args.batch_size,
            shuffle=False
        )

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 记录每个样本的历史损失
        loss_record = torch.ones(len(self.train_data), device=self.device)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            # ===== GRPO策略选择样本 =====
            subset_indices_list = []
            log_probs = []
            rewards = []

            # 多候选采样
            for g in range(self.args.num_candidates):
                # 获取每个样本的选择概率
                probs = self.selector_net(loss_record.unsqueeze(1)).squeeze()

                # 采样子集
                indices = torch.nonzero(
                    torch.bernoulli(probs)
                ).squeeze(1)
                if len(indices) < self.args.batch_size:  # 确保至少有一个批次
                    indices = torch.randperm(len(self.train_data))[:self.args.batch_size]

                log_prob = torch.sum(torch.log(probs[indices]))
                subset_indices_list.append(indices)
                log_probs.append(log_prob)

                # 评估子集
                subset_loader = self._get_data_loader(
                    Subset(self.train_data, indices.cpu().numpy()),
                    batch_size=self.args.batch_size,
                    shuffle=True
                )

                val_loss = self.vali(vali_loader)
                rewards.append(-val_loss)

            # 选择最佳子集
            best_idx = np.argmax(rewards)
            best_subset = subset_indices_list[best_idx]
            best_loader = self._get_data_loader(
                Subset(self.train_data, best_subset.cpu().numpy()),
                batch_size=self.args.batch_size,
                shuffle=True
            )

            # 更新选择网络
            advantages = torch.tensor(rewards, device=self.device) - torch.mean(
                torch.tensor(rewards, device=self.device))
            selector_loss = -sum(adv * lp for adv, lp in zip(advantages, log_probs)) / self.args.num_candidates

            self.optimizer_selector.zero_grad()
            selector_loss.backward()
            self.optimizer_selector.step()

            # 使用最佳子集训练模型
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, _, _) in enumerate(best_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

            # 更新loss记录
            with torch.no_grad():
                for idx in best_subset:
                    loss_record[idx] = torch.mean(torch.tensor(train_loss, device=self.device))

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader)
            test_loss = self.vali(self.test_loader)

            print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f}".format(
                epoch + 1, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def _get_data_loader(self, dataset, batch_size, shuffle=True):
        """辅助函数：创建数据加载器"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
            drop_last=True
        )


# # 收集所有需要梯度的参数
# params = [(name, param)
#           for name, param in self.model.named_parameters()
#           if param.requires_grad]
# # 计算总参数量
# total_params = sum(param.numel() for _, param in params)
#
# # 打印表头
# header = f"{'Name':40s} {'Shape':25s} {'#Params':>12s} {'% of Total':>12s}"
# print(header)
# print("-" * len(header))
#
# # 遍历打印每一项
# for name, param in params:
#     shape = tuple(param.size())
#     num = param.numel()
#     pct = 100.0 * num / total_params
#     print(f"{name:60s} {str(shape):25s} {num:12d} {pct:12.2f}%")
#
# # 打印总计
# print("-" * len(header))
# print(f"{'Total trainable parameters':40s} {'':25s} {total_params:12d} {100.00:12.2f}%")