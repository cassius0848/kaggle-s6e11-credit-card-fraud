# V5：公开最高分文件复刻

本次选择的是核实到的最高分、可复现公开方案：Asterios Panoras 的 **0.95447，Kaggle V19**（scriptVersionId=370693718），核实日期 2026-11-17。这不代表已经找到全站每个选手的最高私有方案。

[原作者方案](https://www.kaggle.com/code/asteriospanoras/s6e11-lb-0-95447-top-public-fully-explained?scriptVersionId=370693718) · [固定输入档案](https://www.kaggle.com/datasets/asteriospanoras/s6e11-public-board-files)

## 结果与提交

- 本地生成：`F:/Kaggle/playground-series-s6e11/outputs/submission_v5_replica_095447.csv`。
- 全部 **299,984 行**、编号顺序和预测值均已校验；与 V19 下载的原版 `submission.csv` **逐字节完全一致**。
- 完整 SHA-256：`c5c95aaffb517f0c9563bd8fb15a30926dcf5515efed477234630d6bbe5ad625`。
- **0.95447 是原作者该文件的官方公开分数**；用户现已报告本账号提交 V5 后排名 **第 25 名**，但尚未单独提供具体公开分数，因此不把作者分数直接记成本账号已确认分数。我们最后已知的自训方案公开分数仍是用户报告的 V4 **0.94913**。
- 文件已通过现有提交校验程序。用户已手动提交上述文件，并报告排名第 25 名。

## 具体做了什么

这是固定的公开预测融合，重用了他人公开模型的预测，未重新训练所有底层模型。没有根据本账号的提交反馈搜索新权重。

令 `U(x)=(rank(x)-0.5)/n`，并列值取平均名次；`L(x)=logit(U(x))`。原作者的最终公式为：

```text
t = L(T)
s = L(S_B)
submission = U(t + 1.5 * (s - t))
```

`T` 是 goodpjw2008 的 0.95442 文件，`S_B` 是其 Board Model + RealMLP 的 0.95446 文件。两个输入都从 Asterios Panoras 的固定档案读取，并验证完整 SHA-256；文件名与指纹见 `configs/public_replica_v5.json`。输出经过重新计算，而非把下载的最终 submission.csv 改名。

**泛化限制：**最终的 1.5 倍外推系数由公开榜选择，额外的半步没有独立交叉验证支持。因此不能保证私榜同样提升，也不能由此声称达到本地 CV 0.95447 或原来的 0.96 目标。原作者也提醒这是高度针对公开榜的文件。

## 代码与重跑

- `scripts/replicate_public_v5.py`：读取固定输入、排名变换、融合、提交及指纹校验。
- `configs/public_replica_v5.json`：来源、版本、输入和输出完整哈希及固定系数。
- `requirements-replica.txt`：本次实际验证的兼容版本。

从仓库目录，使用本机已有环境重跑：

```powershell
$env:PYTHONPATH = 'F:/Kaggle/playground-series-s6e11/work/replica_v5/compat_packages'
& 'F:/Kaggle/playground-series-s6e11/work/research_env/Scripts/python.exe' -m scripts.replicate_public_v5 --workspace 'F:/Kaggle/playground-series-s6e11' --reference 'F:/Kaggle/playground-series-s6e11/work/replica_v5/reference/asteriospanoras_v19/submission.csv'
```

新电脑可在所选 F 盘工作区建立单独环境，安装 `requirements-replica.txt`，然后运行：

```text
python -m scripts.replicate_public_v5 --workspace F:/Kaggle/playground-series-s6e11 --download-inputs
```

仍需已有比赛的 `data/sample_submission.csv` 和 `data/test.csv`。`--download-inputs` 仅通过官方公开接口下载并提取两个所需预测文件；不读取凭据。`--reference` 是可选的原版文件逐字节对比；即使不提供它，也会检查固定的完整输出哈希。模块导入不下载、不训练、不提交。

**兼容细节：**本地原环境 pandas 3.1.1 对 CSV 浮点数的读取改变了部分并列关系，第一次计算有 4,155 行名次差异，最大预测差约 0.00001333。独立安装 pandas **2.3.3** 后完整指纹匹配。脚本会明确拒绝其他 pandas 版本；旧训练环境没有被替换。输出明确使用 LF 换行，以匹配 Kaggle 原 CSV。

## 存储、来源和许可

所有数据、下载档案、兼容组件和结果都在项目 `work/replica_v5/`，不加入 Git：

- `sources/`：审查过的原 Notebook，以及先前候选方案源码；没有直接执行下载的 Notebook。
- `inputs/public-board-files/`：固定输入及其原始 SHA256SUMS。
- `reference/asteriospanoras_v19/`：原作者输出，仅用于验证与留存；其中 CV-led、CV-backed 文件也已保存，并未作为本地自训结果。
- `result.json`：本次运行环境、输入哈希、结果哈希及原版比对结果。
- `downloads/`：下载归档；`compat_packages/` 与 `pip_cache/`：独立兼容组件及其缓存。

来源配方与公开输入按 Apache-2.0 标注，署名与修改说明见 `third_party/public_replica_v5/NOTICE.txt`，许可文本保存在同目录。本地新脚本是对最终固定配方的独立、精简适配。底层公开模型的贡献者和预测链路详见原 Notebook 的 credits。其他项目文件与比赛数据各自的许可仍分别适用。

V1–V4 的模型、提交和结果保持原状。本次没有自动提交 Kaggle，也没有推送 GitHub。
