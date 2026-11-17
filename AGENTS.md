# Project rules

- Launch training or submit to Kaggle only when the user explicitly requests it.
- Importing a training module must never launch training.
- Keep datasets, credentials, predictions, trained models and runtime caches out of Git.
- Distinguish local CV scores from user-reported Kaggle public scores.
- Fit target encodings within training folds; validation labels must not enter training features.
- Use syntax/import checks for packaging edits; full retraining needs explicit authorization.
- Store runtime files in the selected project workspace.
