# Research training V4

Authorized by the user's request to start training according to the research plan.

## Protocol

- Development split: stratified 80/20, random_state=42, identical to V3.
- Outer folds: StratifiedKFold(3, shuffle=True, random_state=2026).
- No held-out fold labels enter target encodings or numerical preprocessing.
- Neural outer-fold experiments select epochs on a separate inner split (20%, seed=1731+fold), then refit on the entire outer training partition for that fixed number of epochs.
- The dataset and these folds have already been used in previous model development. These results are development evidence, not a pristine independent estimate.
- Kaggle public score remains user-reported 0.94910 for V3. No V4 public score is available.
- No automatic Kaggle submission.

## Experiments

1. Fixed LightGBM backbone (4 leaves) with raw, digits, original source, target encodings, and their combinations. Seven development fits; encoded control and development winner in 3 folds.
2. Official TabM with identical backbone across plain, piecewise-linear, and periodic numeric representations. k=16, two blocks of width 128, dropout=0.1; AdamW lr=0.002, weight_decay=0.0003; batch=2048, max48 epochs, patience8/min12. These are resource-conscious research configurations, not a full reproduction of paper benchmarks.
3. Official RealMLP, CUDA, 64 epochs, hidden_sizes=[128,128,128], batch1024, no label smoothing; use AUC to select checkpoint.
4. Loss comparison: same single-member TabM/PLE, identical 8-epoch BCE warmup checkpoint; BCE and AUC-Margin/PESG fine-tuning with exactly the same deterministic class-controlled batches (10% positives). This compares training recipes, since the optimizer also changes.
5. Advance candidates based on both individual AUC and marginal contribution to the matched V3 tree predictions. Confirm shortlisted models with inner-selected epochs and outer-fold predictions.

## Checkpoints and resume

All datasets, environments, model checkpoints, logs, caches, predictions and outputs stay under F:/Kaggle/playground-series-s6e11.

Neural runs save last.pt (model, optimizer, loss parameters, epoch and history), best.pt and preprocessing. Re-running the same command continues from the checkpoint or reuses the completed result. Run signatures reject a changed configuration under the same name.

## Primary sources

- TabM, ICLR 2025: https://github.com/yandex-research/tabm
- Numerical embeddings, NeurIPS 2022: https://github.com/yandex-research/rtdl-num-embeddings
- RealMLP, NeurIPS 2024: https://github.com/dholzmueller/pytabkit
- Deep AUC maximization, ICCV 2021: https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Large-Scale_Robust_Deep_AUC_Maximization_A_New_Surrogate_Loss_and_ICCV_2021_paper.html
- LibAUC: https://docs.libauc.org/api/libauc.losses.html

## Additional confirmation checks

- Capacity check: repeat PLE/raw and PLE/encoded with k=32 and width=256. This changes model capacity and ensemble size together, so it is a practical capacity check, not an isolated causal ablation.
- Within 0.00002 of the best development blend AUC, prefer lower k*width^2. Freeze one raw and one encoded candidate before outer evaluation.
- Repeat each frozen candidate with seed123 on the same development split to measure training variation; retain seed42 for the declared outer comparison.
- RealMLP default-scale follow-up: 3x256 hidden layers and 256 epochs, official optimization/preprocessing; batch4096 for this large dataset, use_ls=False for AUC. The first 3x128/64-epoch experiment is only a small screen.
- Add an original-class-ratio BCE continuation control to the matched 10%-positive BCE/AUC fine-tuning arms.
- TabM checkpoints support automatic continuation. RealMLP saves its completed estimator and a Lightning snapshot every 16 epochs; resuming its internal best-parameter buffers requires an adapter, so the TabM resume guarantee does not extend to RealMLP.
- A final blend must improve mean fold AUC by at least 0.00005, improve at least two folds and lose less than 0.0002 on any fold. Among candidates within 0.00002 of the best, prefer fewer networks and lower neural weight. This rule is a development heuristic; it is not independent confirmation.
