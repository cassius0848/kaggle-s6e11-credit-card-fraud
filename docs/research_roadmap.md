# 研究路线与 V4 实验

当前 Public AUC 为 0.94910。以下方法已完成 V4 首轮实验，结果见 ../reports/research_report_v4.md；不能据此承诺达到 0.96。

1. 特征消融：固定模型、参数和验证划分，检查取值统计、小数位、原始数据及组合收益。
2. [TabM，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html)：参数共享神经网络集成，检查与提升树的互补性。
3. [数值嵌入，NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html)：固定网络，对照标准化数值、分段线性表示与周期嵌入。
4. [RealMLP，NeurIPS 2024](https://proceedings.nips.cc/paper_files/paper/2024/hash/2ee1c87245956e3eaa71aaba5f5753eb-Abstract-Conference.html)：以研究调优的默认配置补充另一种学习方式。
5. [AUC margin loss，ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Large-Scale_Robust_Deep_AUC_Maximization_A_New_Surrogate_Loss_and_ICCV_2021_paper.html)：固定网络后比较交叉熵与 AUC 损失；跨领域结论仍需本赛验证。
6. [模型选择偏差，JMLR 2010](https://jmlr.org/papers/v11/cawley10a.html)：内层选择方案，外层评估；报告各折成绩和训练随机波动。

2026-11-17 已开始按此路线安装并训练。实验配置、验证边界和复现方法见 research_v4_protocol.md；保留与否取决于固定验证方案下的结果及融合增益。
