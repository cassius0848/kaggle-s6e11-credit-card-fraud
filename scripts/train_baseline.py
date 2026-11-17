"""Portable experiment entry point."""

def main():
    import os
    if __package__:
        from .project_runtime import prepare_runtime,experiment_dir
    else:
        from project_runtime import prepare_runtime,experiment_dir
    RUNTIME_ROOT,DATA_DIR=prepare_runtime('baseline',__file__)
    import sys
    from pathlib import Path
    import json
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    root = RUNTIME_ROOT
    out = root / 'outputs'
    out.mkdir(exist_ok=True)
    train = pd.read_csv(DATA_DIR / 'train.csv')
    test = pd.read_csv(DATA_DIR / 'test.csv')
    sample = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    y = train['is_fraud'].astype(int)
    X = train.drop(columns=['id', 'is_fraud']).copy()
    T = test.drop(columns='id').copy()
    assert list(X.columns) == list(T.columns)
    for col in X.select_dtypes(include=['object', 'str']).columns:
        cats = sorted(X[col].dropna().unique().tolist())
        X[col] = pd.Categorical(X[col], categories=cats)
        T[col] = pd.Categorical(T[col], categories=cats)
    summary = dict(train_rows=len(train), test_rows=len(test), features=X.shape[1], fraud_count=int(y.sum()), fraud_rate=float(y.mean()), train_missing=int(train.isna().sum().sum()), test_missing=int(test.isna().sum().sum()), duplicate_train_ids=int(train.id.duplicated().sum()))
    print(json.dumps(summary), flush=True)
    fit_idx, val_idx = train_test_split(np.arange(len(y)), test_size=.2, random_state=42, stratify=y)
    params = dict(objective='binary', learning_rate=.05, num_leaves=31, max_depth=-1, min_child_samples=100, colsample_bytree=.9, reg_lambda=2., n_jobs=int(os.environ.get("KAGGLE_THREADS","4")), random_state=42, verbosity=-1)
    model = lgb.LGBMClassifier(n_estimators=2500, **params)
    model.fit(X.iloc[fit_idx], y.iloc[fit_idx], eval_set=[(X.iloc[val_idx], y.iloc[val_idx])], eval_metric='auc', callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
    p = model.predict_proba(X.iloc[val_idx])[:,1]
    summary.update(validation_auc=float(roc_auc_score(y.iloc[val_idx], p)), best_iteration=int(model.best_iteration_), validation_rows=len(val_idx))
    print(json.dumps(summary), flush=True)
    final_model = lgb.LGBMClassifier(n_estimators=model.best_iteration_, **params)
    final_model.fit(X, y)
    pred = final_model.predict_proba(T)[:,1]
    assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
    assert len(test) == len(sample) and np.array_equal(test.id.to_numpy(), sample.id.to_numpy())
    submission = pd.DataFrame({'id': test.id, 'is_fraud': pred})
    submission.to_csv(out / 'submission_baseline.csv', index=False)
    check = pd.read_csv(out / 'submission_baseline.csv')
    assert check.shape == sample.shape and list(check.columns) == list(sample.columns)
    assert np.array_equal(check.id, sample.id) and check.is_fraud.notna().all()
    summary.update(prediction_min=float(pred.min()), prediction_max=float(pred.max()), prediction_mean=float(pred.mean()))
    (root / 'work' / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    pd.DataFrame({'feature':X.columns,'importance':final_model.feature_importances_}).sort_values('importance',ascending=False).to_csv(root/'work'/'feature_importance.csv', index=False)
    print('DONE: '+str(out / 'submission_baseline.csv'), flush=True)


if __name__ == "__main__":
    main()
