"""Check a prediction CSV against the official sample submission."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("submission",type=Path)
    parser.add_argument("--sample",type=Path,required=True)
    args=parser.parse_args()
    frame=pd.read_csv(args.submission); sample=pd.read_csv(args.sample)
    if frame.columns.tolist()!=["id","is_fraud"] or frame.shape!=sample.shape:
        raise ValueError("Wrong columns or row count")
    if not frame.id.is_unique or not np.array_equal(frame.id.to_numpy(),sample.id.to_numpy()):
        raise ValueError("IDs are duplicated, missing, or reordered")
    scores=frame.is_fraud.to_numpy(dtype=float)
    if not np.isfinite(scores).all() or not ((scores>=0)&(scores<=1)).all():
        raise ValueError("Probabilities must be finite and between 0 and 1")
    print("Valid submission:",len(frame),"rows")

if __name__=="__main__": main()
