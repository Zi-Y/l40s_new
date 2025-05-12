import torch
# import torch.nn.functional as F
import numpy as np
# import argparse # Not used
import matplotlib.pyplot as plt

# --- Dependency Functions (from user-uploaded derivatives.py and trendfilter.py adaptations) ---
from scipy.sparse import spdiags, coo_matrix, dia_matrix
from itertools import chain
import cvxpy


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


def get_trend_regularization(x_coords, base_model_s, derv_1_expression, l_norm_penalty, alpha1_penalty, alpha2_penalty):
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

    obj_and_model = get_trend_objective_function_and_model(y_signal_dim, loss_type=trend_loss_type, positive=False)
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
        trend_result = np.array(base_model_s.value) if base_model_s.value is not None else np.full_like(y_signal_dim,
                                                                                                        np.nan)  # Ensure result is array even if None
        if verbose and np.isnan(trend_result).any():
            print(f"    TrendCalc WARNING: Resulting trend contains NaNs after solve().")
        return trend_result
    except Exception as e:
        if verbose: print(f"    TrendCalc ERROR: Exception during solve with solver {solver}: {e}. Returning NaNs.")
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
            ax1.plot(time_steps, trend_dim_data, label=f'Trend (with NaN)', marker='x', linestyle=':', markersize=4)
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
    plt.show()


# --- 主函数 ---
def process_batch_and_get_anomaly_scores(
        observed_batch_md: torch.Tensor,
        plot_first_n_samples: int = 0,
        verbose: bool = False
) -> tuple[list[float], np.ndarray]:
    lambda_reg_trend = 0.3
    lay_back = 5
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
        if verbose: print(f"\nProcessing Sample {sample_idx + 1}/{batch_size} in Batch...")
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

        if verbose: print(f"\n    --- Debugging Anomaly Score Calculation per Dimension (Sample {sample_idx + 1}) ---")
        for dim_idx in range(num_features):
            if verbose: print(f"      Processing Dimension: {dim_idx + 1}")
            obs_dim = observed_sample_md[:, dim_idx]
            trend_dim = trend_sample_md[:, dim_idx]
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
                    dim_pscore = _cal_anoscore_internal(err_for_score_dim, func=score_func, start_T=start_T_score,
                                                        end_T=end_T_score)
                elif verbose:
                    print(f"        Dim {dim_idx + 1} - Invalid range for _cal_anoscore_internal, dim_pscore = 0.0")
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
                sample_name=f"Sample {sample_idx + 1}/{batch_size} (Overall Agg. Score: {pscore_current_sample_aggregated:.2f})"
            )

    full_batch_error_tensor_np = np.stack(batch_error_tensors_list_np, axis=0)
    if verbose: print("\nFunction finished processing batch.")
    return batch_anomaly_scores_aggregated, full_batch_error_tensor_np


# --- 主执行示例 ---
if __name__ == '__main__':
    example_batch_size = 2
    example_seq_length = 40
    example_num_features = 2

    print(
        f"Generating Example Batch (BatchSize: {example_batch_size}, Features: {example_num_features}, SeqLen: {example_seq_length})")
    batch_observed_list_np = []
    fixed_lay_back = 5

    for b_idx in range(example_batch_size):
        observed_features_list_np = []
        for i in range(example_num_features):
            base_seq = np.sin(np.linspace(b_idx * 0.5, (i + 2 + b_idx * 0.2) * np.pi, example_seq_length * 2)) + \
                       np.random.randn(example_seq_length * 2) * 0.25 + (i * 1.5)
            obs_dim_np = base_seq[:example_seq_length].copy()
            if b_idx == 0:
                if i == 0:
                    if example_seq_length > 5:
                        num_anom_points = min(3, example_seq_length // 5)
                        obs_dim_np[-num_anom_points:] += 4.0  # 持续性强异常
                        print(
                            f"  INFO (Sample {b_idx + 1}, Dim {i + 1}): Added sustained positive anomaly to end. Vals: {obs_dim_np[-num_anom_points:]}")
                elif i == 1:
                    if example_seq_length > 10:
                        mid_point = example_seq_length // 2
                        obs_dim_np[mid_point - 2: mid_point + 2] -= 4.5  # 中间强异常
                        print(
                            f"  INFO (Sample {b_idx + 1}, Dim {i + 1}): Added sharp negative anomaly to middle. Vals: {obs_dim_np[mid_point - 2: mid_point + 2]}")
                        # Also add a distinct anomaly at the very end for this dim
                        obs_dim_np[-1] += 2.0  # 单点末尾异常
                        print(
                            f"  INFO (Sample {b_idx + 1}, Dim {i + 1}): Added point anomaly to end. Val: {obs_dim_np[-1]}")

            elif b_idx == 1:
                if i == 0:
                    if example_seq_length > 3:
                        obs_dim_np[:3] += 2.0  # 开头异常
                        print(f"  INFO (Sample {b_idx + 1}, Dim {i + 1}): Added small positive anomaly to start.")
                elif i == 1:
                    if example_seq_length > 1:
                        obs_dim_np[-1] += 7.0  # 末尾剧烈单点异常
                        print(
                            f"  INFO (Sample {b_idx + 1}, Dim {i + 1}): Added very large positive anomaly to last point. Val: {obs_dim_np[-1]}")
            observed_features_list_np.append(obs_dim_np)
        sample_np = np.stack(observed_features_list_np, axis=1)
        batch_observed_list_np.append(sample_np)

    observed_batch_np_example = np.stack(batch_observed_list_np, axis=0)
    observed_batch_md_example = torch.from_numpy(observed_batch_np_example).float()

    batch_scores, batch_errors = process_batch_and_get_anomaly_scores(
        observed_batch_md_example,
        plot_first_n_samples=example_batch_size,
        verbose=True
    )

    print(f"\n--- FINAL BATCH RESULTS ---")
    for idx, score in enumerate(batch_scores):
        print(f"Sample {idx + 1} - Aggregated Anomaly Score A(x_n): {score:.4f}")
    print(f"Shape of returned batch_error_tensors: {batch_errors.shape}")
    if example_batch_size > 0 and example_num_features > 0 and example_seq_length > 0:
        print(f"Example error value from batch (Sample 0, Dim 0, last point): {batch_errors[0, -1, 0]:.4f}")
    if example_batch_size > 0 and example_num_features > 1 and example_seq_length > 0:
        print(f"Example error value from batch (Sample 0, Dim 1, last point): {batch_errors[0, -1, 1]:.4f}")
    if example_batch_size > 1 and example_num_features > 0 and example_seq_length > 0:
        print(f"Example error value from batch (Sample 1, Dim 0, last point): {batch_errors[1, -1, 0]:.4f}")