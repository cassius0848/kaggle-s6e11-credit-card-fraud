# Kaggle S6E11：信用卡欺诈预测

从首次有效提交到浅层提升树、折内取值统计与融合的实验项目。

- 比赛：[Predicting Credit Card Fraud](https://www.kaggle.com/competitions/playground-series-s6e11)。
- 输入：24 个交易特征，训练集 699,960 行，测试集 299,984 行。
- 输出：每笔交易的欺诈概率；评分指标为 ROC-AUC。
- 自训模型的最好公开成绩：**0.94913**，由参赛者于 **2026-11-17** 报告。
- **0.96000 是尚未达到的目标，不是已取得的成绩。**

## 实验结果

| 版本 | 方法 | 本地评估 | Public AUC |
|---|---|---:|---:|
| V1 | LightGBM 基线 | 单次留出 0.945182 | 0.94612 |
| V2 | 两版 LightGBM + CatBoost 等权融合 | 三折均值 0.946809 | 0.94743 |
| V3 | 取值统计 LightGBM + 浅层 XGBoost 各占 50% | 三折均值 0.948768 | **0.94910** |
| V4 | V3 占 75% + 目标编码 TabM 占 25% | 三折均值 0.948906 | **0.94913** |

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
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt

建议将仓库、虚拟环境、数据与缓存放在空间充足的磁盘，例如 F:/Kaggle。

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

    python run_pipeline.py --stage baseline --data-dir F:/Kaggle/playground-series-s6e11/data
    python -m scripts.download_original --data-dir F:/Kaggle/playground-series-s6e11/data
    python run_pipeline.py --stage v2 --data-dir F:/Kaggle/playground-series-s6e11/data
    python run_pipeline.py --stage v3 --data-dir F:/Kaggle/playground-series-s6e11/data

--threads 设置 CPU 线程上限。默认保留历史配置：V1/V2 四线程，V3 八线程。代码、数据或选项改变时，缓存检查会要求采用新的 --run-id；同一轮各阶段使用相同 run-id，并重新生成前置阶段。

## 检查与提交

    python -m scripts.validate_submission outputs/submission_v3_recommended.csv --sample data/sample_submission.csv

脚本检查列名、行数、编号唯一性与顺序，以及概率是否有限且位于 0～1。随后手动上传 CSV 到 Kaggle；项目不会自动提交，也不会在导入模块时开始训练。

V2 的历史预测性能基准可在对应训练完成后运行：

    python run_pipeline.py --stage benchmark

[历史性能记录](reports/inference_benchmark_v2.json)来自特定电脑，不是跨硬件性能保证。

## 研究来源

- [公开浅层树研究 Notebook](https://www.kaggle.com/code/alexchilton/credit-card-fraud-lb-0-9542-shallow-tree-stack)，原文声明 Apache 2.0。V1–V4 独立实现相关思路，未使用其他作者的预测文件；V5 是下述有署名的公开预测融合复刻。
- [官方链接的原始数据](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026)，CC0。
- 比赛数据须遵守[比赛规则](https://www.kaggle.com/competitions/playground-series-s6e11/rules)与数据许可。

待验证的论文方法见 [docs/research_roadmap.md](docs/research_roadmap.md)。数据及参考资料许可分别适用，本项目尚未选定开源许可证。

## V4 论文方法实验

2026-11-17 开始按研究路线训练，实验协议见 [docs/research_v4_protocol.md](docs/research_v4_protocol.md)。

- scripts/research_features.py：各折的特征构造与目标编码。
- scripts/research_ablation.py：固定 LightGBM 参数的特征消融。
- scripts/research_neural.py：TabM 三种数值表示、RealMLP、BCE/AUC 对照及三折复验。
- scripts/research_evaluate.py：同一验证样本上的单模型与融合比较。

研究环境与原来的提升树环境分开，依赖见 requirements-research.txt。先安装该文件，再单独安装 libauc==2.0.1（--no-deps）；只使用其损失与优化器，无需图像模块。完整实测版本保存在本地 work/research_v4/environment.txt。

从仓库目录运行，例如：

    python -m scripts.research_ablation --workspace F:/Kaggle/playground-series-s6e11
    python -m scripts.research_neural --workspace F:/Kaggle/playground-series-s6e11 --stage screen
    python -m scripts.research_neural --workspace F:/Kaggle/playground-series-s6e11 --stage realmlp-default
    python -m scripts.research_neural --workspace F:/Kaggle/playground-series-s6e11 --stage loss

上述命令会实际训练。结果在工作区的 work/research_v4，分析表在 outputs。TabM 在每个 epoch 保存模型、优化器和历史记录，同配置重新运行可继续；RealMLP 在完成后保存完整估计器，默认规模实验另存每 16 轮的 Lightning 快照，原生恢复还需要适配其内部最佳轮次状态。

现有 V3 提交和模型保留。V4 本地实验结果不能当作 Kaggle 公开成绩。

V4 首轮实验已完成，结果见 [research_report_v4.md](reports/research_report_v4.md)。三个折均小幅改善，平均增量约 0.000138；本地与公开评估应分开看：用户报告 V4 公开分数为 0.94913，比 V3 高 0.00003，收益仍很小。

## V5 公开高分方案复刻

已重建 Asterios Panoras 的公开 0.95447 方案（V19），生成 submission_v5_replica_095447.csv；与原作者输出的完整 SHA-256 一致，299,984 行校验通过。[配方、运行方法和限制](docs/public_replica_v5.md)。

0.95447 是原作者分数；用户已报告本账号 V5 提交后排名第 25 名，具体公开分数尚未单独提供。V5 重用公开预测并按固定公式融合，不是重新训练底层模型；最终系数针对公开榜选择，私榜收益没有保证。原来的 V1–V4 自训结果单独保留。

## V6 互补性实验

已完成 60 组现有预测的融合与折间尺度对齐检查，没有组合通过预设条件，保留 V5 提交。[实验结论和下一条独立训练路线](reports/public_extension_v6.md)。

## 最新公开参照

2026-11-18 核验了另一份公开 V2 输出。该版本在 Kaggle 的 Public AUC 为 **0.95450**。本项目直接采用作者发布的 `submission.csv`，核对了全部 299,984 行的编号、顺序和预测范围。对应版本和文件指纹见[公开结果记录](reports/public_results.json)。这项成绩归属于来源版本，本项目的独立训练成绩见上表。
