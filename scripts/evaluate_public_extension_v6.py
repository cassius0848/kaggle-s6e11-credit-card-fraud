"""Measure marginal value of frozen local predictions against public OOF stacks.

No base-model training, leaderboard fitting or automatic submission. Import is inert.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

from .replicate_public_v5 import logit_rank, rank_fraction, read_prediction, sha256

CONFIG = Path(__file__).resolve().parents[1] / 'configs/public_extension_v6.json'


def auc(y, scores):
    positives = y == 1
    npos = int(positives.sum())
    nneg = len(y) - npos
    if not npos or not nneg:
        raise ValueError('AUC needs both classes')
    ranks = rankdata(scores, method='average')
    return float((ranks[positives].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def paired_auc(y, a, b):
    pos = y == 1
    neg = ~pos
    def placements(score):
        pp, pn = np.sort(score[pos]), np.sort(score[neg])
        vp = (np.searchsorted(pn, score[pos], side='left') + np.searchsorted(pn, score[pos], side='right')) / (2 * len(pn))
        vn = 1 - (np.searchsorted(pp, score[neg], side='left') + np.searchsorted(pp, score[neg], side='right')) / (2 * len(pp))
        return vp, vn
    ap, an = placements(a)
    bp, bn = placements(b)
    delta = float(np.mean(ap - bp))
    se = float(np.sqrt(np.var(ap - bp, ddof=1) / len(ap) + np.var(an - bn, ddof=1) / len(an)))
    return {'delta': delta, 'conditional_se': se, 'conditional_95_interval': [delta - 1.96 * se, delta + 1.96 * se]}


def checked_array(path, n, expected_hash, manifest):
    digest = sha256(path)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f'Input version changed: {path}')
    values = np.load(path, allow_pickle=False).astype(np.float64)
    if values.shape != (n,) or not np.isfinite(values).all():
        raise ValueError(f'Invalid fixed prediction array: {path}')
    manifest[str(path)] = digest
    return values


def metrics(y, scores, partitions):
    return {'pooled_auc': auc(y, scores), 'fold_auc': [auc(y[idx], scores[idx]) for idx in partitions]}


def write_candidate(path, ids, score):
    frame = pd.DataFrame({'id': ids, 'is_fraud': rank_fraction(score)})
    payload = frame.to_csv(index=False, lineterminator='\n').encode()
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists() and sha256(path) != digest:
        raise ValueError(f'Refusing to overwrite a different candidate: {path}')
    path.write_bytes(payload)
    read_prediction(path, digest, ids)
    return {'file': str(path), 'sha256': digest, 'rows': len(ids), 'public_score': None}


def run(root):
    started = time.perf_counter()
    if pd.__version__ != '2.3.3':
        raise RuntimeError('Use the documented pandas 2.3.3 compatibility environment')
    protocol = json.loads(CONFIG.read_text(encoding='utf-8'))
    work = root / 'work/extension_v6'
    output = root / 'outputs'
    work.mkdir(parents=True, exist_ok=True)
    manifest = {'protocol': sha256(CONFIG), 'script': sha256(Path(__file__))}
    train = pd.read_csv(root / 'data/train.csv', usecols=['id', 'is_fraud'])
    y = train.is_fraud.to_numpy(np.int8)
    ids = pd.read_csv(root / 'data/test.csv', usecols=['id']).id.to_numpy()
    sample = pd.read_csv(root / 'data/sample_submission.csv', usecols=['id'])
    if not train.id.is_unique or not sample.id.is_unique or not np.array_equal(ids, sample.id.to_numpy()):
        raise ValueError('Dataset ID alignment failed')
    manifest['train_csv'] = sha256(root / 'data/train.csv')
    bundle_path = work / 'inputs/s6e11-rank1-q17-meta-bundle/rank1_q17_streams.npz'
    if sha256(bundle_path) != protocol['public_sha256']['bundle']:
        raise ValueError('Public bundle changed')
    z = np.load(bundle_path, allow_pickle=False)
    if not np.array_equal(y, z['y_train']):
        raise ValueError('Public OOF labels are not in train.csv row order')
    public_folds = [np.flatnonzero(z['folds'] == k) for k in range(5)]
    local_folds = [v for _, v in StratifiedKFold(3, shuffle=True, random_state=2026).split(y, y)]
    public_dir = work / 'inputs/s6e11-fraud-series-artifacts'
    bases, base_test = {}, {}
    for name, key in [('wide_lm_stack', 'wide_lm'), ('wide_stack', 'wide')]:
        bases[name] = logit_rank(checked_array(public_dir / f'{name}_nested.npy', len(y), protocol['public_sha256'][key + '_oof'], manifest))
        base_test[name] = logit_rank(checked_array(public_dir / f'{name}_test.npy', len(ids), protocol['public_sha256'][key + '_test'], manifest))
    local_tree = root / 'work/experiments_v3'
    lgb = checked_array(local_tree / 'oof_lgb_leaf4_encoded.npy', len(y), None, manifest)
    xgb = checked_array(local_tree / 'oof_xgb_depth2_full.npy', len(y), None, manifest)
    v3 = 0.5 * lgb + 0.5 * xgb
    neural_dir = root / 'work/research_v4/neural'
    encoded = checked_array(neural_dir / 'oof_tabm_ple_encoded.npy', len(y), None, manifest)
    raw = checked_array(neural_dir / 'oof_tabm_ple_raw.npy', len(y), None, manifest)
    own_raw = {'v3_tree_blend': v3, 'v4_tree_tabm_blend': 0.75 * v3 + 0.25 * encoded,
               'tabm_ple_encoded': encoded, 'tabm_ple_raw': raw}
    own = {name: logit_rank(v) for name, v in own_raw.items()}
    own_test = {}
    for name, filename, expected in [('v3_tree_blend', 'submission_v3_recommended.csv', None),
                                      ('v4_tree_tabm_blend', 'submission_v4_research.csv', protocol['v4_sha256'])]:
        path = output / filename
        digest = expected or sha256(path)
        own_test[name] = logit_rank(read_prediction(path, digest, ids))
        manifest[str(path)] = digest
    for name in ['tabm_ple_encoded', 'tabm_ple_raw']:
        own_test[name] = logit_rank(checked_array(neural_dir / f'test_{name}.npy', len(ids), None, manifest))
    summary = {name: {'public_partition': metrics(y, p, public_folds), 'local_partition': metrics(y, p, local_folds)} for name, p in bases.items()}
    own_summary = {name: {'pooled_auc': auc(y, p), 'rank_correlation_with_public_stack': float(np.corrcoef(rankdata(p), rankdata(bases['wide_lm_stack']))[0, 1]),
                          'test_rank_correlation_with_public_stack': float(np.corrcoef(rankdata(own_test[name]), rankdata(base_test['wide_lm_stack']))[0, 1])} for name, p in own.items()}
    public_members = [{'name': str(name), 'oof_auc': auc(y, z['oof_' + str(name)])} for name in z['stream_names']]
    rows = []
    for base_name, baseline in bases.items():
        base_public = summary[base_name]['public_partition']
        base_local = summary[base_name]['local_partition']
        for name, addition in own.items():
            for w in protocol['weights'][1:]:
                score = (1 - w) * baseline + w * addition
                public = metrics(y, score, public_folds)
                local = metrics(y, score, local_folds)
                fd = (np.array(public['fold_auc']) - base_public['fold_auc']).tolist()
                ld = (np.array(local['fold_auc']) - base_local['fold_auc']).tolist()
                rows.append({'base': base_name, 'own_model': name, 'own_weight': w,
                             'pooled_auc': public['pooled_auc'], 'pooled_delta': public['pooled_auc'] - base_public['pooled_auc'],
                             'mean_five_fold_delta': float(np.mean(fd)), 'public_fold_deltas': fd,
                             'positive_public_folds': sum(v > 0 for v in fd), 'worst_public_fold_delta': min(fd),
                             'local_fold_deltas': ld, 'positive_local_folds': sum(v > 0 for v in ld)})
            best = max((r for r in rows if r['base'] == base_name and r['own_model'] == name), key=lambda r: r['pooled_auc'])
            print(base_name, name, 'best weight', best['own_weight'], 'delta', f"{best['pooled_delta']:+.8f}", 'folds', best['positive_public_folds'], flush=True)
    gate = protocol['selection_gate']
    eligible = []
    for row in rows:
        if row['base'] != 'wide_lm_stack':
            continue
        secondary = next(r for r in rows if r['base'] == 'wide_stack' and r['own_model'] == row['own_model'] and r['own_weight'] == row['own_weight'])
        row['secondary_base_pooled_delta'] = secondary['pooled_delta']
        preliminary = (row['mean_five_fold_delta'] >= gate['primary_mean_five_fold_delta_at_least']
                       and row['positive_public_folds'] >= gate['primary_positive_five_folds_at_least']
                       and row['worst_public_fold_delta'] >= gate['primary_worst_fold_delta_at_least']
                       and row['positive_local_folds'] == gate['positive_local_three_folds_required']
                       and secondary['pooled_delta'] > 0)
        row['diagnostic_gate_passed'] = False
        if preliminary:
            w = row['own_weight']
            score = (1 - w) * bases['wide_lm_stack'] + w * own[row['own_model']]
            row['paired_interval'] = paired_auc(y, score, bases['wide_lm_stack'])
            if row['paired_interval']['conditional_95_interval'][0] > 0:
                row['diagnostic_gate_passed'] = True
                eligible.append(row)
    selected = None
    submissions = []
    if eligible:
        best_auc = max(r['pooled_auc'] for r in eligible)
        selected = min([r for r in eligible if r['pooled_auc'] >= best_auc - 0.000002], key=lambda r: r['own_weight'])
        w, name = selected['own_weight'], selected['own_model']
        submissions.append({'role': 'OOF-backed development candidate', **write_candidate(output / 'submission_v6_cv_own_contribution.csv', ids, (1 - w) * base_test['wide_lm_stack'] + w * own_test[name])})
        v5_path = output / 'submission_v5_replica_095447.csv'
        v5 = logit_rank(read_prediction(v5_path, protocol['v5_sha256'], ids))
        submissions.append({'role': 'Experimental transfer onto V5; no valid local V5 CV score', **write_candidate(output / 'submission_v6_v5_own_experimental.csv', ids, (1 - w) * v5 + w * own_test[name])})
    primary = bases['wide_lm_stack']
    for name in own_summary:
        baseline = output / 'submission_v5_replica_095447.csv'
        v5 = read_prediction(baseline, protocol['v5_sha256'], ids)
        own_summary[name]['test_rank_correlation_with_v5'] = float(np.corrcoef(rankdata(own_test[name]), rankdata(v5))[0, 1])
    result = {'method': protocol['objective'], 'protocol': protocol, 'baselines': summary, 'own_models': own_summary,
              'public_members': public_members, 'candidates': rows, 'selected': selected, 'submissions': submissions,
              'input_hashes': manifest, 'seconds': time.perf_counter() - started,
              'v5_cv_score': None, 'v5_public_rank': 25, 'rank_source': 'user-reported',
              'limitations': protocol['limits'], 'no_new_base_training': True, 'no_kaggle_submission': True}
    (work / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    report_copy = Path(__file__).resolve().parents[1] / 'reports/public_extension_v6.json'
    report_copy.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    table = pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, (list, dict))} for row in rows])
    table.to_csv(output / 'public_extension_v6_grid.csv', index=False, lineterminator='\n')
    print('SELECTED', json.dumps(selected, ensure_ascii=False), 'seconds', result['seconds'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    run(args.workspace.resolve())


if __name__ == '__main__':
    main()
