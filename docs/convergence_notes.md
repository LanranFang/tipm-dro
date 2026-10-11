# Monte Carlo convergence experiments

这里收录与论文 Figure 1 / Appendix E.1 及 Figure 5 / Appendix E.2 对应的最终两份实验。源码整理自已有 notebooks；PDF 本身没有 Python 源码或嵌入文件。既有 CSV 和论文图保存在 `reference_results/convergence/`，可以直接查看，不需要重新求解全部实验。

| 论文 | Notebook / 同名 `.py` | 原结果目录 | 本仓库参考结果目录 |
| --- | --- | --- | --- |
| Figure 1，Appendix E.1 | `convergence/suboptimal_dual_gap_vs_M_dimension.ipynb` | `suboptimal_gap_figures/` | `reference_results/convergence/bounded/` |
| Figure 5，Appendix E.2 | `convergence/suboptimal_dual_gap_vs_M_dimension_student_t_xi.ipynb` | `suboptimal_gap_figures_student_t_xi/` | `reference_results/convergence/student_t/` |

## 参数与计算对象

两份程序都保留原实验的随机种子 `20260519`、维度 `[5, 25, 100]`、Monte Carlo 样本数 `[2, 5, 10, 20, 50, 100, 250, 500]`、50 次独立重复、100 个对称历史样本、`epsilon=0`、单位球决策集及 `x_star=0.1*ones(d)`。每个维度有 4 个固定 dense Gaussian unit directions，权重均匀排列在 0.5 至 1.5，再归一化至平均为 1，因此有 8 个 max-affine pieces。每次重复抽取一个 500 点辅助决策池，按 `M` 使用嵌套前缀。

bounded 版本的历史样本来自单位球。Student-t 版本为独立坐标、5 个自由度，缩放后每坐标方差为 `1/(d+2)`；它在整个 `R^d` 上施加 sampled-dual 半无限约束，不添加论文 Section 4 的尾部约束。两个版本均评价 sampled-dual solution 的训练经验损失，相对 gap 以已知经验最优点 `x_star` 的训练经验损失为分母。

每个 raw CSV 有 `3 × 50 × 8 = 1200` 行，每个 summary 有 24 行。绘图显示平均百分比 gap 和四分位区间；参考线为 pooled log-scale 拟合，固定斜率 `-1/2`。对收录的 summary 重新计算得到：

| 数据 | 拟合 `C` | 论文图中的 `C` |
| --- | ---: | ---: |
| bounded | 26.97917269477976 | 26.98 |
| Student-t | 30.59275634347842 | 30.59 |

## 与论文叙述的二阶矩差异及推导前提

**Appendix E.1：** PDF 写二阶矩界 `Omega` 为训练经验二阶矩 `mean(||xi_i||^2)`；原 bounded notebook 的 `omega_for_unit_ball_support()` 实际返回半径平方，当前参数下为 **1.0**，既有 raw CSV 的 `omega` 列也全部为 1.0。本仓库保留这个实际生成论文图的代码参数，没有替换为经验二阶矩。这个设置仍使单位球训练经验分布满足二阶矩界。如果要研究 PDF 所述经验矩设置，应当另做实验并区分其输出。

**Appendix E.2：** 原代码和论文都设置 `Omega=d/(d+2)`，这是按缩放后的 Student-t 总体方差计算的二阶矩。有限训练样本的经验二阶矩可能大于或小于这个总体值；成对加入 `xi` 和 `-xi` 保证经验均值为零，却不会保证经验二阶矩不超过总体二阶矩。

当 `epsilon=0` 时，文中全决策损失约束给出 `E_Q[loss(x,xi)] <= E_P_hat[loss(x,xi)]`。若 `P_hat` 自身满足包括二阶矩界在内的所有 ambiguity-set 约束，则 `Q=P_hat` 可用于取得反向不等式，从而得到原 notebook 中的 `V_0(x)=E_P_hat[loss(x,xi)]`。在有限训练经验矩超过 `Omega` 时，这个“使用 `P_hat` 得到反向不等式”的步骤不能直接使用；这是等式推导的可行性前提，不能仅凭 `epsilon=0` 宣称等式恒成立。

成对对称样本与凸的 shifted loss 仍使 `x_star` 是训练经验目标的最优点。因此保存的数字和图可直接解释为 **sampled-dual solution 的 empirical objective gap**。以上区别没有改变现有图、数据或优化参数，也不据此断言图或求解失败。仅重抽原随机种子的训练样本而不求解，可以观察到 d=5、25、100 的 50 次重复中分别有 23、26、32 个经验矩超过总体 `Omega`。

## 运行与整理改动

在仓库根目录安装项目依赖后运行任一脚本：

```bash
python convergence/suboptimal_dual_gap_vs_M_dimension.py
python convergence/suboptimal_dual_gap_vs_M_dimension_student_t_xi.py
```

可先运行两个独立的小规模检查，输出与正式结果隔离：

```bash
python convergence/suboptimal_dual_gap_vs_M_dimension.py --quick --workers 1
python convergence/suboptimal_dual_gap_vs_M_dimension_student_t_xi.py --quick --workers 1
```

`--quick` 仅对脚本本次运行覆盖为 `d=5`、`M=[2,5]`、1 次重复、10 个历史样本；结果保存到 `results/convergence/quick/`，不会混入完整实验数据。`--workers` 覆盖环境变量指定的 worker 数量。

脚本只在显式调用 `main()` 或直接执行时运行实验。Notebook 可从仓库根目录或子目录启动。两种方式的新 CSV 写入 `results/convergence/`，新图写入其中 `suboptimal_gap_figures/` 或 `suboptimal_gap_figures_student_t_xi/`。完整实验需要 1200 次 SOCP 求解，每次重跑会覆盖同一实验的新输出文件；仓库中的参考结果保持独立。

并行 worker 默认从 64 上限调整为 4，使用 `IPM_DRO_WORKERS` 可覆盖，例如 `IPM_DRO_WORKERS=2 python convergence/suboptimal_dual_gap_vs_M_dimension.py`。这只影响并行调度，随机种子和数学设置保持原值。

Python 依赖为 NumPy、Pandas、Matplotlib、CVXPY。原 solver 顺序及数值容差保持为 ECOS、CLARABEL、SCS；没有可用 solver、重试失败或没有有限解时会报出明确错误。原代码接受 `optimal_inaccurate` 的 fallback 行为仍保留，实际状态写入 CSV。Matplotlib 默认使用 MathText，替换原 `bm` 图例和 `usetex=True`，不要求系统 LaTeX。

按原 notebook **0-based cell indices**，改动为：cell 0 增加论文映射及运行说明；Student-t 的 cell 0 补充经验分布可行性前提；cell 3 定位仓库并设置统一结果路径；cell 4 调整默认 workers、加入环境变量和 solver 存在检查；cell 6 检查重试状态及有限解；cells 8–10 统一 CSV 和绘图结果路径；cell 11 默认关闭 LaTeX 并改用 MathText。各 code cell 清除输出和 execution count，清除 cell 运行 metadata，notebook metadata 仅保留通用 Python kernelspec / language_info。数学模型、损失函数、`Omega`、采样方式、数据数量和求解容差未修改。

没有收录旧 `suboptimal_bound_vs_M_dimension.ipynb` 及其他探索性图表。原 `suboptimal_known_opt_inverse_sqrt_fit_check.csv` 对应维度 3/10/20/50，而当前论文图为 5/25/100；原 `suboptimal_known_opt_percentage_gap_vs_M_by_dim.png` 的拟合常数为 25.31，也不等于当前 Figure 1 的 26.98。因此这两个文件没有作为论文参考结果复制。

## 来源与 SHA-256

下表的原路径均相对于原工作目录 `/home/lanran/projects/`。原文件只读，整理版保存在新仓库。`.py` 是整理后对应 notebook code cells 的导出，计算部分放入 `main()`，增加独立的 `--quick` 与 `--workers` 命令行选项。

| 原 notebook | 原文件 SHA-256 |
| --- | --- |
| `suboptimal_dual_gap_vs_M_dimension.ipynb` | `49ff3dce6e10e12231bf3b631e809c030dec9e792ca6be404388743f7868f321` |
| `suboptimal_dual_gap_vs_M_dimension_student_t_xi.ipynb` | `6296082888fec699850b44a42c7f71c776007c3369ecff5e84ee512f55fb4ed7` |

参考结果复制后与原文件逐字节一致：

| 原文件 | 本仓库目标（`reference_results/convergence/` 下） | 原文件 SHA-256 |
| --- | --- | --- |
| `suboptimal_known_opt_gap_results.csv` | `bounded/suboptimal_known_opt_gap_results.csv` | `26ca495b667fa75a56de36006cdbbee5e275c941a6af6a14eee914cd63602e2b` |
| `suboptimal_known_opt_gap_summary.csv` | `bounded/suboptimal_known_opt_gap_summary.csv` | `59fb996ca83f82110c47ea500590e66b615f6f735cc44d68442fa5fc4b61d701` |
| `suboptimal_gap_figures/suboptimal_gap_summary_for_plot.csv` | `bounded/suboptimal_gap_summary_for_plot.csv` | `c2986810e87d0f2f06dc5a82ce9b104a02dd2711cf7e26509d63023df3d0ceb9` |
| `suboptimal_gap_figures/suboptimal_percentage_gap_vs_M_by_dim.pdf` | `bounded/suboptimal_percentage_gap_vs_M_by_dim.pdf` | `c99c44b432a0330d434b73a2ed22ca1dae7cfbc1504b4018c7d4deffce5dbc56` |
| `suboptimal_known_opt_gap_results_student_t_xi.csv` | `student_t/suboptimal_known_opt_gap_results_student_t_xi.csv` | `4948b0090dd41a3aad4b8cdcb02daebc9a6fe3544e043af1424bd7864f62251a` |
| `suboptimal_known_opt_gap_summary_student_t_xi.csv` | `student_t/suboptimal_known_opt_gap_summary_student_t_xi.csv` | `f8f5711dd48ce591b156cc844e864c4db1dc6d6ebce82927e01d839484e2d83c` |
| `suboptimal_gap_figures_student_t_xi/suboptimal_gap_summary_for_plot.csv` | `student_t/suboptimal_gap_summary_for_plot.csv` | `b954703605e9c6ddff90c21a14f366bf918cee8aaa2d158bdbbbb8e995c9eaa4` |
| `suboptimal_gap_figures_student_t_xi/suboptimal_percentage_gap_vs_M_by_dim.pdf` | `student_t/suboptimal_percentage_gap_vs_M_by_dim.pdf` | `cc16422134a0ae4386e4df89fc4092da2a5ad39c4dbdd7af42c844ca18003c9e` |

## 最低运行验证

整理后两个 `--quick --workers 1` 命令均实际运行完成，退出码为 0，各产生 2 行 `status=optimal` 的 raw CSV，并成功生成两种 PDF 图（默认 MathText）。另外检查了各 notebook code cell / 脚本语法、清空的运行信息、脚本导入时的 `__main__` 保护、从仓库根目录和 `convergence/` 子目录定位结果路径、参考结果逐字节复制及原 notebook SHA 保持不变。没有重跑完整 1200 次求解实验；快速检查结果不用于替换论文图。
