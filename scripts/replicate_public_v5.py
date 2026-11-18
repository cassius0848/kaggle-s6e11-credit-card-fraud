"""Rebuild Asterios Panoras' scored S6E11 V19 file from public predictions.

SPDX-License-Identifier: Apache-2.0
Recipe credit: Asterios Panoras and goodpjw2008. See docs/public_replica_v5.md.
This is a fixed public-prediction ensemble, not base-model retraining.
Importing the module does not download, train, write files or submit to Kaggle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.special import logit
from scipy.stats import rankdata

CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'public_replica_v5.json'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rank_fraction(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Scores must be a finite one-dimensional array')
    return (rankdata(values, method='average') - 0.5) / len(values)


def logit_rank(values: np.ndarray) -> np.ndarray:
    return logit(rank_fraction(values))


def read_prediction(path: Path, expected_hash: str, ids: np.ndarray) -> np.ndarray:
    actual = sha256(path)
    if actual != expected_hash:
        raise ValueError(f'Input version changed: {path.name}: {actual}')
    frame = pd.read_csv(path)
    if frame.columns.tolist() != ['id', 'is_fraud'] or len(frame) != len(ids):
        raise ValueError(f'Wrong prediction schema: {path.name}')
    if not frame.id.is_unique or not np.array_equal(frame.id.to_numpy(), ids):
        raise ValueError(f'Missing, repeated or reordered IDs: {path.name}')
    scores = frame.is_fraud.to_numpy(np.float64)
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError(f'Invalid probabilities: {path.name}')
    return scores


def download_inputs(directory: Path, config: dict) -> None:
    """Use the official public download endpoint; never use account credentials."""
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory.parent / 'downloads' / 'public-board-files.zip'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        partial = archive.with_suffix('.zip.partial')
        with urllib.request.urlopen(config['dataset_download_url'], timeout=30) as response:
            if 'zip' not in response.headers.get('Content-Type', ''):
                raise ValueError('Kaggle did not return a ZIP archive')
            with partial.open('wb') as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
        partial.replace(archive)
    # Only extract the two pinned prediction files; do not execute archive content.
    with zipfile.ZipFile(archive) as source:
        for spec in config['inputs'].values():
            filename = spec['filename']
            contents = source.read(filename)
            if hashlib.sha256(contents).hexdigest() != spec['sha256']:
                raise ValueError(f'Downloaded input differs: {filename}')
            destination = directory / filename
            if destination.exists() and sha256(destination) != spec['sha256']:
                raise ValueError(f'Refusing to overwrite a different file: {destination}')
            destination.write_bytes(contents)


def rebuild(workspace: Path, input_dir: Path, output: Path, config: dict, reference: Path | None = None) -> dict:
    started = time.perf_counter()
    if pd.__version__ != config['required_pandas_version']:
        raise RuntimeError('This byte-exact recipe needs pandas ' + config['required_pandas_version'] + '; see requirements-replica.txt. Other CSV parsers can change ties.')
    sample = pd.read_csv(workspace / 'data' / 'sample_submission.csv')
    test_ids = pd.read_csv(workspace / 'data' / 'test.csv', usecols=['id']).id.to_numpy()
    ids = sample.id.to_numpy()
    if len(ids) != config['expected_rows'] or not sample.id.is_unique:
        raise ValueError('Unexpected sample submission IDs')
    if not np.array_equal(ids, test_ids):
        raise ValueError('Sample and test IDs differ')
    vectors = {}
    for name, spec in config['inputs'].items():
        vectors[name] = logit_rank(read_prediction(input_dir / spec['filename'], spec['sha256'], ids))
    # Preserve the author's expression and operation order; do not search new weights.
    scores = vectors['T'] + config['extrapolation'] * (vectors['S_B'] - vectors['T'])
    frame = sample[['id']].assign(is_fraud=rank_fraction(scores))
    # Linux line endings reproduce the source Kaggle CSV on Windows too.
    payload = frame.to_csv(index=False, lineterminator='\n').encode('utf-8')
    digest = hashlib.sha256(payload).hexdigest()
    if digest != config['expected_output_sha256']:
        raise ValueError(f'Rebuild differs from author fingerprint: {digest}')
    if reference is not None and reference.read_bytes() != payload:
        raise ValueError('Rebuilt predictions differ from the original reference file')
    if output.exists() and sha256(output) != digest:
        raise ValueError(f'Refusing to overwrite a different submission: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    # Re-read the actual deliverable and validate its complete ID order and probabilities.
    read_prediction(output, digest, ids)
    return {
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'method': 'Fixed ensemble of two public prediction files; no base-model training',
        'source_url': config['source_url'],
        'author_public_score': config['author_public_score'],
        'our_kaggle_public_score': None,
        'local_cv_score': None,
        'author_fingerprint_matches': True,
        'author_fingerprint_bits': 256,
        'reference_bytes_equal': True if reference is not None else None,
        'reference': str(reference) if reference is not None else None,
        'output': str(output), 'sha256': digest, 'rows': len(frame),
        'columns': frame.columns.tolist(), 'id_order_matches_test': True,
        'min_prediction': float(frame.is_fraud.min()),
        'max_prediction': float(frame.is_fraud.max()),
        'unique_predictions': int(frame.is_fraud.nunique()),
        'inputs': {k: {'path': str(input_dir / v['filename']), 'sha256': v['sha256']} for k,v in config['inputs'].items()},
        'seconds': time.perf_counter() - started,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__},
        'limitation': 'The final 1.5 multiplier was selected on the public leaderboard. No independent CV supports this extra half-step or guarantees the private score.',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--download-inputs', action='store_true')
    parser.add_argument('--reference', type=Path, help='Optional original author output for a byte-for-byte comparison')
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    inputs = args.input_dir or workspace / 'work' / 'replica_v5' / 'inputs' / 'public-board-files'
    output = args.output or workspace / 'outputs' / 'submission_v5_replica_095447.csv'
    if args.download_inputs:
        download_inputs(inputs, config)
    result = rebuild(workspace, inputs, output, config, args.reference)
    report = workspace / 'work' / 'replica_v5' / 'result.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
