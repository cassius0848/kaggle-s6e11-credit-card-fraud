# V4 研究实验结果

V3 的用户报告公开分数为 **0.94910**，V4 为 **0.94913**（2026-11-17 用户报告），提升 **0.00003**；**尚未达到 0.96**。

## 三折复验

| 方法 | 三折平均 AUC | 各折 AUC |
|---|---:|---|
| V3 树模型融合 | 0.94876771 | 0.94916154, 0.94924462, 0.94789697 |
| tabm_ple_encoded | 0.94773141 | 0.94795432, 0.94805119, 0.94718872 |
| tabm_ple_raw | 0.94514400 | 0.94573806, 0.94555379, 0.94414016 |
| 本轮候选融合 | 0.94890573 | 0.94928944, 0.94931570, 0.94811206 |

权重：{"v3": 0.75, "tabm_ple_encoded": 0.25}。
相对 V3 的平均增量：+0.00013803。各折增量：+0.00012791, +0.00007108, +0.00021509。
用户报告本候选 Public AUC 为 0.94913，比 V3 高 0.00003。单次公开分数的微小提升不足以证明稳定优势，私榜结果未知。

提交文件：F:\Kaggle\playground-series-s6e11\outputs\submission_v4_research.csv
行数：299984；列：id、is_fraud；编号顺序、概率范围、缺失值和重新读入检查均通过。

## 首轮固定留出实验

| 方法 | AUC | 最佳轮次 |
|---|---:|---:|
| loss_auc | 0.94362054 | 4 |
| loss_balanced_bce | 0.94000837 | 1 |
| loss_unbalanced_bce | 0.94229823 | 2 |
| loss_warmup | 0.94199961 | 8 |
| screen_realmlp | 0.94234578 | 30 |
| screen_realmlp_default | 0.94220307 | 52 |
| screen_tabm_periodic | 0.94386546 | 23 |
| screen_tabm_plain | 0.94205348 | 12 |
| screen_tabm_ple | 0.94452348 | 19 |
| screen_tabm_ple_encoded | 0.94674707 | 12 |
| screen_tabm_ple_encoded_seed123 | 0.94690108 | 13 |
| screen_tabm_ple_encoded_w256_k32_s42 | 0.94659280 | 12 |
| screen_tabm_ple_raw_w256_k32_s42 | 0.94459400 | 19 |
| screen_tabm_ple_seed123 | 0.94425058 | 13 |

## 特征消融

| 特征组 | 固定留出 AUC |
|---|---:|
| raw | 0.94584133 |
| digits | 0.94656085 |
| original | 0.94452598 |
| encoded | 0.94708474 |
| encoded_digits | 0.94726542 |
| encoded_original | 0.94627022 |
| all | 0.94676018 |

取值统计是本轮最明确的有效特征。小数位叠加后的三折增益很小；原始公开数据的统计在本次固定骨干上没有带来收益。

## 验证边界

TabM 三折先在每个外层训练部分内部选择 epoch，再用全部外层训练部分固定轮次重训，外层标签只在最终计算 AUC 时读取。所有目标编码与数值预处理都在训练部分拟合。

历史验证集已用于多轮开发；候选族和融合权重仍根据开发结果选择，因此这不是完全独立的盲测。V3 早停协议与本轮神经网络不同。配对区间只对固定预测作条件比较，不消除选择偏差。

两组损失对照共享 8 轮 BCE 预训练权重，并用相同的 10% 正类批次、32 轮训练预算。AUC 组同时改用了 PESG，不能把差异只归因于损失函数。另有原始类别比例的 BCE 对照。

RealMLP 默认规模实验使用官方默认的三层 256、256 轮及其优化方案；大数据训练批次调至 4096，并关闭标签平滑，以 AUC 选择模型。小规模筛选使用三层 128、64 轮。

TabM 较小配置：k=16、两层128；扩容对照：k=32、两层256。数值表示对照的其余骨干参数保持一致。随机种子 123 只作训练波动检查，没有用来选择超参数。

## 文件

代码：repository/scripts/research_features.py、research_ablation.py、research_neural.py、research_evaluate.py、research_finalize.py。
模型、预处理、逐轮记录和预测：work/research_v4。
全部环境版本：work/research_v4/environment.txt。
完整开发集对照：outputs/research_v4_development.csv；三折对照：outputs/research_v4_cv.csv。

本轮由用户提交 Kaggle 并报告公开分数；代码未自动推送 GitHub。
