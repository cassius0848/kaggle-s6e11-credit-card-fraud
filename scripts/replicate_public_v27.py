"""Reproduce the V27 test-set ensemble from frozen public model predictions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .replicate_public_v5 import logit_rank, rank_fraction, sha256


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "public_replica_v27.json"


def prepare_inputs(directory, sources, download):
    def prepare(item):
        group, dataset, version, filename, expected = item
        path = directory / group / filename
        if path.exists():
            if sha256(path) != expected:
                raise ValueError(f"Unexpected input fingerprint: {path}")
            return
        if not download:
            raise FileNotFoundError(f"{path} is missing. Use --download-inputs.")
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset}/{filename}?datasetVersionNumber={version}"
        with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temp = Path(stream.name)
        try:
            with urlopen(url, timeout=90) as response, temp.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            if sha256(temp) != expected:
                raise ValueError(f"Downloaded input fingerprint differs: {filename}")
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)

    tasks = [(group, spec["dataset"], spec["version"], name, digest)
             for group, spec in sources.items() for name, digest in spec["files"].items()]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(prepare, tasks))


def csv_scores(path, ids, column="is_fraud"):
    frame = pd.read_csv(path)
    if "id" in frame and not np.array_equal(frame["id"].to_numpy(), ids):
        raise ValueError(f"ID order differs: {path}")
    if len(frame) != len(ids):
        raise ValueError(f"Row count differs: {path}")
    return frame[column].to_numpy(np.float64)


def member_step(inputs, config, ids, base):
    """Replay the frozen 152-member adjustment in the published summation order."""
    matrix_dir = inputs / "matrix"
    matrix = np.load(matrix_dir / "test.npy", mmap_mode="r")
    names = pd.read_csv(matrix_dir / "manifest.csv")["name"].tolist()
    named = config["named_members"]

    def prediction(key):
        if key.startswith("s:"):
            return bundle["test_" + key[2:]]
        if key.startswith("b:"):
            return np.load(inputs / "bank" / f"test_{key[2:]}.npy")
        if key.startswith("x:b33_"):
            return matrix[:, names.index(key[6:])]
        if key.startswith("x:weni_"):
            return np.load(inputs / "pinned" / f"test_{key[2:]}.npy")
        group, name = named[key]
        path = inputs / group / name
        if path.suffix == ".npy":
            return np.load(path)
        return csv_scores(path, ids, "pred" if key == "x:davut" else "is_fraud")

    # Rank float32 model files in double precision across SciPy versions.
    weights = config["weights"]
    with np.load(inputs / "bundle" / "rank1_q17_streams.npz") as bundle:
        adjustment = sum(
            weight * logit_rank(np.asarray(prediction(key), dtype=np.float64))
            for key, weight in weights.items()
        )
    return logit_rank((1 - sum(weights.values())) * base + adjustment) - base


def residual(direction, moves):
    matrix = np.column_stack(moves)
    coefficients = np.linalg.lstsq(matrix, direction, rcond=None)[0]
    return direction - matrix @ coefficients


def low_risk_mask(test):
    velocity = test["velocity_score"]
    risk = test["merchant_risk_score"]
    flags = ["is_new_merchant", "is_foreign_transaction", "used_vpn", "ip_country_mismatch",
             "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
    clean = (test["cvv_retry_count"] == 0) & (test["prior_disputes"] == 0) & ~test[flags].any(axis=1)
    return ((risk < 13.1) | ((velocity <= 4.1) & (risk <= 20)) |
            (clean & (((velocity <= 4.1) & (risk <= 33.8)) |
                      ((velocity <= 6.0) & (risk <= 25))))).to_numpy()


def rebuild(test, inputs, config):
    ids = test["id"].to_numpy()

    def pinned_rank(name):
        return logit_rank(csv_scores(inputs / "pinned" / name, ids))

    steps = {key: pinned_rank(name) for key, name in config["scored_steps"].items()}
    anchors = {key: pinned_rank(name) for key, name in config["anchors"].items()}

    historical = [steps[k] for k in "jklmnopq"] + [anchors["round3"]]
    moves = [later - earlier for earlier, later in zip(historical, historical[1:])]
    orthogonal = residual(anchors["orthogonal_base"] - steps["o"], moves)
    fusion = residual(anchors["round4"] - steps["q"],
                      [steps["r"] - steps["q"], steps["q"] - steps["p"], steps["p"] - steps["o"]])
    base = logit_rank(steps["r"] + 0.35 * fusion)
    adjustment = member_step(inputs, config, ids, base)
    latest = anchors["round5"]
    scores = latest + 0.75 * (latest - base) + 0.2 * adjustment + 0.15 * orthogonal
    return rank_fraction(rank_fraction(scores) - 2 * low_risk_mask(test))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--input-dir", type=Path, default=Path("work/v27_inputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/submission_v27_095553.csv"))
    parser.add_argument("--download-inputs", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    test = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    if (len(test) != config["rows"] or not test["id"].is_unique or
            not np.array_equal(test["id"].to_numpy(), sample["id"].to_numpy())):
        raise ValueError("Competition IDs do not match the expected test set")
    prepare_inputs(args.input_dir, config["sources"], args.download_inputs)
    scores = rebuild(test, args.input_dir, config)
    frame = sample[["id"]].assign(is_fraud=scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=args.output.parent, delete=False) as stream:
        temp = Path(stream.name)
    try:
        frame.to_csv(temp, index=False, lineterminator="\n")
        digest = sha256(temp)
        if digest != config["output_sha256"]:
            raise ValueError(f"Reproduction differs from the scored file: {digest}")
        if args.output.exists() and sha256(args.output) != digest:
            raise FileExistsError(f"Refusing to replace a different file: {args.output}")
        temp.replace(args.output)
    finally:
        temp.unlink(missing_ok=True)
    print(f"{len(frame):,} rows verified. SHA-256 {digest}. Output: {args.output}")


if __name__ == "__main__":
    main()
