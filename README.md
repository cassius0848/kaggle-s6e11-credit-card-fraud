# Kaggle S6E11：信用卡欺诈预测

从首次有效提交到浅层提升树、折内取值统计与融合的实验项目。

- 比赛：[Predicting Credit Card Fraud](https://www.kaggle.com/competitions/playground-series-s6e11)。
- 输入：24 个交易特征，训练集 699,960 行，测试集 299,984 行。
- 输出：每笔交易的欺诈概率；评分指标为 ROC-AUC。
- 最好公开成绩：**0.94910**，由参赛者于 **2026-11-17** 报告。
- **0.96000 是尚未达到的目标，不是已取得的成绩。**

## 实验结果

| 版本 | 方法 | 本地评估 | Public AUC |
|---|---|---:|---:|
| V1 | LightGBM 基线 | 单次留出 0.945182 | 0.94612 |
| V2 | 两版 LightGBM + CatBoost 等权融合 | 三折均值 0.946809 | 0.94743 |
| V3 | 取值统计 LightGBM + 浅层 XGBoost 各占 50% | 三折均值 0.948768 | **0.94910** |

V1 的单次留出成绩不能直接与 V2/V3 的三折均值比较。公开分数由参赛者提供，最终私榜成绩未知。早停、候选和融合权重使用开发验证数据，CV 成绩不是完全独立的测试估计。

V3 推荐方案各折 AUC：0.949162、0.949245、0.947897。LightGBM 单模型三折均值为 0.948453，XGBoost 为 0.948101。

## 方法

1. 分层划分保持欺诈比例；id 不参与训练。
2. 筛选使用 80/20 划分，种子 42；复核使用三折分层交叉验证，种子 2026。
3. V3 使用浅层提升树、数值具体取值和部分交互的目标统计。
4. 目标统计只在外层训练部分拟合，训练部分内部再做三折交叉拟合。
5. 比较小数位特征，以及原始公开数据的统计和源数据模型预测。
6. 入围模型使用全部训练数据重训，轮次取各折最佳轮次的中位数，再生成提交。

GPU CatBoost、样条逻辑回归及精确值独热逻辑回归也经过比较，并非所有候选都有正收益。记录见 [reports](reports/)。

## 项目结构

- run_pipeline.py：显式选择运行阶段。
- scripts/：基线、V2、V3、预测基准、数据下载和提交检查。
- notebooks/：清除运行输出的入门 Notebook。
- reports/：小体积历史指标、模型配置、公开成绩与数据指纹。
- docs/：待验证的研究路线。
- data/、outputs/、work/：本地数据、生成预测、模型与缓存，Git 忽略。

## 安装

历史环境为 Windows、Python 3.14。依赖固定在 requirements.txt。其他平台和 Python 版本需要确认兼容性。

    python -m venv .venv
    .venvScriptsActivate.ps1
    python -m pip install -r requirements.txt

建议将仓库、虚拟环境、数据与缓存放在空间充足的磁盘，例如 F:Kaggle。

## 数据

先在 Kaggle 加入比赛，下载 train.csv、test.csv、sample_submission.csv，放到 data/。仓库不包含比赛数据、预测概率、训练模型或凭据。

V3 需要官方链接的原始公开数据：

    python -m scripts.download_original

下载脚本只读取指定 CSV 并检查历史 SHA-256；数据版本变化时会停止。[来源与许可](data/README.md)

## 复现顺序

以下命令会实际训练。仅查看命令时添加 --dry-run。

    python run_pipeline.py --stage baseline
    python run_pipeline.py --stage v2
    python run_pipeline.py --stage v3

V2 依赖基线输出；V3 的历史比较依赖 V2 的验证预测和提交，应按顺序运行。

V3 默认尝试 CUDA。历史 GPU 为 RTX 4070 Laptop 8GB，CPU 与 GPU 的数值结果和成绩可能不同。仅 CPU 环境可以始终使用：

    python run_pipeline.py --stage all --cpu

数据可以留在别的目录，例如：

    python run_pipeline.py --stage baseline --data-dir F:Kaggleplayground-series-s6e11data
    python -m scripts.download_original --data-dir F:Kaggleplayground-series-s6e11data
    python run_pipeline.py --stage v2 --data-dir F:Kaggleplayground-series-s6e11data
    python run_pipeline.py --stage v3 --data-dir F:Kaggleplayground-series-s6e11data

--threads 设置 CPU 线程上限。默认保留历史配置：V1/V2 四线程，V3 八线程。代码、数据或选项改变时，缓存检查会要求采用新的 --run-id；同一轮各阶段使用相同 run-id，并重新生成前置阶段。

## 检查与提交

    python -m scripts.validate_submission outputs/submission_v3_recommended.csv --sample data/sample_submission.csv

脚本检查列名、行数、编号唯一性与顺序，以及概率是否有限且位于 0～1。随后手动上传 CSV 到 Kaggle；项目不会自动提交，也不会在导入模块时开始训练。

V2 的历史预测性能基准可在对应训练完成后运行：

    python run_pipeline.py --stage benchmark

[历史性能记录](reports/inference_benchmark_v2.json)来自特定电脑，不是跨硬件性能保证。

## 研究来源

- [公开浅层树研究 Notebook](https://www.kaggle.com/code/alexchilton/credit-card-fraud-lb-0-9542-shallow-tree-stack)，原文声明 Apache 2.0。本项目独立实现相关思路，未使用其他作者的预测文件。
- [官方链接的原始数据](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026)，CC0。
- 比赛数据须遵守[比赛规则](https://www.kaggle.com/competitions/playground-series-s6e11/rules)与数据许可。

待验证的论文方法见 [docs/research_roadmap.md](docs/research_roadmap.md)。数据及参考资料许可分别适用，本项目尚未选定开源许可证。
