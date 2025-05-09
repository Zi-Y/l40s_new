import numpy as np
import cvxpy
from trendfilter_modify.extrapolate import get_interp_extrapolate_functions
from trendfilter_modify.derivatives import second_derivative_matrix_nes, \
    first_derv_nes_cvxpy
from trendfilter_modify.linear_deviations import complete_linear_deviations


def trend_filter(x, y, loss='mse', y_err=None, alpha_1=0.0,
                 alpha_2=0.0, l_norm=2,
                 constrain_zero=False, monotonic=False,
                 positive=False,
                 linear_deviations=None,
                 solver='ECOS'):
    """
    主函数：对给定散点 (x, y) 进行趋势滤波。
    :param x: 自变量 numpy 数组，形状 (n,)；
    :param y: 因变量 numpy 数组，形状 (n,)；
    :param y_err: 观测误差，可选，默认为全 1；
    :param alpha_1: 对一阶导数项的惩罚系数；
    :param alpha_2: 对二阶导数项的惩罚系数；
    :param l_norm: 1 或 2，选择 L1 或 L2 正则；
    :param constrain_zero: 是否强制 s[0]=0；
    :param monotonic: 是否强制一阶导≥0（单调递增）；
    :param positive: 是否强制 s ≥ 0；
    :param linear_deviations: 可选的线性偏差模型；
    :param solver: cvxpy 求解器名称。
    :return: 包含趋势模型和相关信息的字典。
    """
    # 如果没有外加线性偏差模型，初始化为空列表
    if linear_deviations is None:
        linear_deviations = []
    # 补全并初始化所有线性偏差对象
    linear_deviations = complete_linear_deviations(linear_deviations, x)

    # 确保范数参数合法
    assert l_norm in [1, 2]
    n = len(x)

    # 检查 y 与 x 长度一致，并生成默认误差
    assert len(y) == n
    if y_err is None:
        y_err = np.ones(n)
    else:
        assert len(y_err) == n

    # 构造拟合项（data term）和基础模型 base_model
    result = get_obj_func_model(
        y, loss=loss, y_err=y_err,
        positive=positive,
        linear_deviations=linear_deviations
    )

    # 计算一阶导数表达式，用于单调约束或一阶正则
    derv_1 = first_derv_nes_cvxpy(x, result['base_model'])

    # 构造正则项：包括一阶和二阶差分及线性偏差
    reg_sum, regs = get_reg(
        x, result['base_model'], derv_1,
        l_norm, alpha_1, alpha_2,
        linear_deviations=linear_deviations
    )

    # 总目标 = 拟合项 + 正则项
    obj_total = result['objective_function'] + reg_sum
    obj = cvxpy.Minimize(obj_total)

    # 构造约束列表
    constraints = []
    if constrain_zero:
        # 强制 s[0] = 0
        constraints.append(result['model'][0] == 0)
    if monotonic:
        # 强制一阶导 ≥ 0，实现单调递增
        constraints.append(derv_1 >= 0)

    # 定义并求解凸优化问题
    problem = cvxpy.Problem(obj, constraints=constraints)
    problem.solve(solver=solver)

    # 构造趋势插值/外推函数
    func_base, func_deviates, func = \
        get_interp_extrapolate_functions(
            x, result['base_model'], linear_deviations
        )

    # 整理并返回结果字典
    tf_result = {
        'x': x,
        'y': y,
        'y_err': y_err,
        'function': func,
        'function_base': func_base,
        'function_deviates': func_deviates,
        'model': result['model'],            # base_model + 偏差
        'base_model': result['base_model'],  # 纯趋势 s
        'objective_model': result['objective_function'],
        'regularization_total': reg_sum,
        'regularizations': regs,
        'objective_total': obj,
        'y_fit': result['model'].value,      # 拟合值
        'constraints': constraints,
        'linear_deviations': linear_deviations
    }
    return tf_result


# ---------- 辅助函数详解 ----------

def get_reg(x, base_model, derv_1, l_norm, alpha_1, alpha_2, linear_deviations=None):
    """
    构造正则化项。
    - 一阶罚项：alpha_1 * ‖D1 s‖_p
    - 二阶罚项：alpha_2 * ‖D2 s‖_p
    - 线性偏差罚项：对每个 lin_dev 加 alpha*‖dev_var‖_p
    """
    # 生成二阶差分矩阵 D2，形状 (n-2, n)
    d2 = second_derivative_matrix_nes(x, scale_free=True)

    # 选择 L1 或 L2 范数
    if l_norm == 2:
        norm = cvxpy.sum_squares
    else:
        norm = cvxpy.norm1

    # 一阶正则：alpha_1 * norm(D1 s)
    reg_1 = alpha_1 * norm(derv_1)
    # 二阶正则：alpha_2 * norm(D2 s)
    reg_2 = alpha_2 * norm(d2 @ base_model)
    regs = [reg_1, reg_2]

    # 对每个线性偏差模型也加正则
    for lin_dev in linear_deviations:
        reg = lin_dev['alpha'] * norm(lin_dev['variable'])
        regs.append(reg)

    # 汇总所有正则项
    reg_sum = sum(regs)
    return reg_sum, regs


def get_obj_func_model(y, loss='mse', y_err=None, positive=False, linear_deviations=None):
    """
    构造数据拟合项和模型表达式：
    - base_model 是待估计的趋势变量 s (cvxpy.Variable(n)).
    - model = base_model + 线性偏差贡献.
    - 损失：加权 Huber (mse) 或 加权 MAE (mae).
    """
    if linear_deviations is None:
        linear_deviations = []
    n = len(y)

    # 默认误差全 1
    if y_err is None:
        y_err = np.ones(n)
    else:
        assert len(y_err) == n

    # small buffer 避免除零／abs(0) 不稳定
    buff = 0.01 * np.median(np.abs(y))
    buff_2 = buff ** 2
    isig = 1 / np.sqrt(buff_2 + y_err ** 2)

    # 定义趋势变量 base_model (可选非负)
    base_model = cvxpy.Variable(n, pos=positive)
    model = base_model

    # 累加所有线性偏差模型
    for lin_dev in linear_deviations:
        model += lin_dev['model_contribution']

    # 计算加权残差
    diff = cvxpy.multiply(isig, model - y)
    if loss == 'mse':
        # Huber 损失近似 MSE
        obj_func = cvxpy.sum(cvxpy.huber(diff))
    elif loss == 'mae':
        # L1 损失
        obj_func = cvxpy.sum(cvxpy.abs(diff))

    return {
        'base_model': base_model,
        'model': model,
        'objective_function': obj_func
    }
