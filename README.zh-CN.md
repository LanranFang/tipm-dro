# TIPM-DRO 论文实验代码

本项目整理自论文 **Distributionally Robust Optimization via Targeted Integral Probability Metric for General Data Processes** 对应的已有实验代码，作者为 Lanran Fang、Jianqiang Cheng、Grani A. Hanasusanto 和 Yijie Wang。

PDF 包含数学模型、实验说明和图表，实际 Python 源码来自原项目中的 Notebook 与辅助模块。这里按论文图表收录了 7 个 Notebook、对应命令行入口、核心模型模块和原始参考结果；原项目文件保留不动。

| 目录 | 内容 | 论文对应 |
| --- | --- | --- |
| `convergence/` | 有界支撑与 Student-t 不确定性下的 Monte Carlo 实验 | 图 1、图 5；附录 E.1–E.2 |
| `newsvendor/` | 三商品报童模型，Gaussian、Chi-square、Lognormal、Pareto 四种需求分布 | 图 2；第 5.1 节、附录 E.3 |
| `regression/` | 异常值污染回归，维度和污染强度实验 | 图 3–4；第 5.2 节、附录 E.4 |
| `reference_results/` | 原实验的 CSV 和已生成的论文相关图 | 图 1–5 |
| `tools/` | 项目检查与参考结果重绘 | 无需求解器即可重绘 |
| `docs/` | 运行说明、源文件对应关系、参数差异和校验记录 | 复现说明 |

建议使用 Python 3.12 或 3.13。在本目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/check_project.py
python tools/plot_reference_results.py
```

最后一条命令使用保存的 CSV 重绘图 1–5，生成文件位于 `results/reference_plots/`。它不会重新计算优化实验。

小规模运行示例：

```bash
python newsvendor/run_gaussian_3d.py --quick --solver CLARABEL --workers 1
python regression/run_regression.py --quick --experiment all --jobs 1
python convergence/suboptimal_dual_gap_vs_M_dimension.py --quick --workers 1
python convergence/suboptimal_dual_gap_vs_M_dimension_student_t_xi.py --quick --workers 1
```

完整运行与 Notebook 的使用方法见[复现指南](docs/REPRODUCING.md)。原实验部分使用 Gurobi/MOSEK；本项目提供开放求解器和小规模验证入口。`--quick` 使用缩小参数，仅检查能否执行。完整实验保留原实验设置，计算量较大。

整理时已清空 Notebook 输出、去掉无关探索单元、修复本机绝对路径和失效启动方式。没有默默更改论文模型的实验参数：有界 Monte Carlo 的二阶矩界实际为 `Omega=1`，回归代码实际为经验二阶矩，分别与论文附录中的描述存在差异。详细记录见[收敛实验说明](docs/convergence_notes.md)、[回归实验说明](docs/regression_notes.md)和[报童实验说明](docs/newsvendor_notes.md)。参考 CSV 保留原运行结果，新运行的随机种子修复与求解器选择可能使结果不同。

原回归程序在拟合失败时会评价零预测；整理版改为明确记录失败方法和缺失风险值。保存的旧回归 CSV 保持原样，解读时需结合上述限制，具体情况见回归实验说明。

上传 GitHub 时只上传这个项目目录。虚拟环境、运行缓存和新生成的 `results/` 已加入 `.gitignore`。原 PDF、投资组合实验、Markov 对比实验、重复临时版本未收录。源文件来源和 SHA-256 见[源文件对应表](docs/SOURCE_MAP.md)及[来源清单](docs/source_manifest.json)。

在 GitHub 创建空仓库后，在本目录执行以下命令，把示例地址替换为自己的仓库地址：

```bash
git init -b main
git add .
git commit -m "Add TIPM-DRO paper experiments"
git remote add origin https://github.com/YOUR_USERNAME/tipm-dro.git
git push -u origin main
```

整理过程只准备本地文件；没有创建 GitHub 仓库或执行远程上传。
