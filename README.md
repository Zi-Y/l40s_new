下面详细介绍 Trend Filtering 算法 的原理与实现步骤。

⸻

1. 问题定义
给定长度为 $T$ 的观测序列 ${\tilde z_t}{t=1}^T$，Trend Filtering 的目标是同时逼近数据并获得一条平滑的、分段多项式形式的趋势曲线 ${s_t}{t=1}^T$。其通用形式为
$$
\displaystyle
s_{1:T} ;=;\arg\min_{s\in\mathbb R^T}
\underbrace{\sum_{t=1}^T \ell_{\rm data}\bigl(\tilde z_t,,s_t\bigr)}{\text{数据拟合项}}
;+;\lambda;
\underbrace{\sum{t=k+2}^T \bigl|(\Delta^{(k+1)}s)t\bigr|}{\text{高阶差分稀疏正则}},
$$
其中
	•	$\ell_{\rm data}$ 是数据拟合损失，常取平方损失 $\ell=\bigl(\tilde z_t - s_t\bigr)^2$（标准 Trend Filtering）或绝对损失 $\ell=|\tilde z_t - s_t|$（鲁棒 Trend Filtering）；
	•	$\Delta^{(k+1)}$ 表示 $(k+1)$ 阶离散差分算子；
	•	$\lambda>0$ 为正则化超参数，权衡光滑度与拟合度；
	•	$k$ 决定趋势的多项式阶数：
	•	$k=0$ → 分段常数，类似断点检测；
	•	$k=1$ → 分段线性；
	•	$k=2$ → 分段二次；
	•	……

⸻

2. 离散差分算子
	•	一阶差分
$$
(\Delta^{(1)}s)t = s_t - s{t-1},\quad t=2,\dots,T.
$$
	•	二阶差分
$$
(\Delta^{(2)}s)t = (\Delta^{(1)}s)t - (\Delta^{(1)}s){t-1}
= s_t - 2s{t-1} + s_{t-2},\quad t=3,\dots,T.
$$
	•	$(k+1)$ 阶差分
可递归定义，专门惩罚高阶变化，以实现分段 $k$ 次多项式趋势的稀疏性。

⸻

3. 模型性质与几何解释
	•	广义 Lasso
Trend Filtering 问题等价于广义 Lasso：
$$
\min_s \tfrac12|y - s|_2^2 + \lambda|D,s|_1,
$$
其中 $D$ 为差分矩阵。
	•	分段多项式
最优解 $s^$ 在任意一段上是一个 $k$ 次多项式；变化点即 $(\Delta^{(k+1)}s^)_t \neq 0$ 的位置。
	•	平衡拟合与光滑
较小 $\lambda$ → 拟合度高、变化点多；
较大 $\lambda$ → 趋势更平滑、变化点少。

⸻

4. 求解算法
由于目标包含不可导的 $\ell_1$ 项，常用的数值方法包括：
	1.	Primal–Dual Interior-Point (PDIP)
将问题转化为二阶锥规划 (SOCP)，利用内点法直接求解，适用于中等规模。
	2.	ADMM（交替方向乘子法）
	•	将原问题拆分，引入辅助变量 $u = D,s$；
	•	交替最小化关于 $s$ 和 $u$ 的子问题，并更新拉格朗日乘子；
	•	每次迭代包含一次带二次项的线性系统求解和一次软阈值操作。
	3.	Path Algorithm（分段路径算法）
针对二次数据项，沿着 $\lambda$ 变化的轨迹高效追踪解的断点集合，时间复杂度可降至 $O(T)$ 。
	4.	Dynamic Programming（动态规划）
对特定阶数的差分，可设计状态转移，精准定位变化点，常用于 $k=0,1$ 。

⸻

5. 鲁棒 Trend Filtering
在带异常的时间序列预测论文中，使用了 $l_1$ 数据项：
$$
\sum_{t=1}^T \bigl|\tilde z_t - s_t\bigr|
;+;
\lambda\sum_{t=2}^{T-1}\bigl|s_{t-1} - 2s_t + s_{t+1}\bigr|.
$$
	•	数据项：绝对偏差对尖锐异常不敏感；
	•	正则项：二阶差分稀疏化，捕捉全局平滑趋势  ￼。
求解可同样采用 ADMM 或 PDIP，只需将二次数据项替换为 $l_1$ 项，并在 $s$ 更新中用加权中位数等方法。

⸻

6. 参数选择与实践建议
	•	$\lambda$：通过交叉验证或信息准则（AIC/BIC）选取；
	•	数据预处理：中心化、归一化可加速数值收敛；
	•	数值稳定：大规模序列可用分块或者流式算法；
	•	软件实现：Python 中可用 cvxpy、pyglmnet 或专门的 trendfilter 包。

⸻

以上即是 Trend Filtering 算法 的全面解读，从数学模型到求解方法，再到鲁棒扩展与实践要点。