"""Portable paths and cache guards. Importing never trains."""
from pathlib import Path
import hashlib,json,os,re

def experiment_dir(root,version):
    run_id=os.environ.get("KAGGLE_RUN_ID","default")
    if not re.fullmatch(r"[A-Za-z0-9_-]+",run_id):
        raise ValueError("Invalid run-id")
    return root/"work"/("experiments_"+version+("" if run_id=="default" else "_"+run_id))

def prepare_runtime(stage,script_file):
    root=Path(__file__).resolve().parents[1]
    data=Path(os.environ.get("KAGGLE_DATA_DIR",str(root/"data"))).resolve()
    required=[data/"test.csv"]
    if stage!="benchmark":
        required+=[data/"train.csv",data/"sample_submission.csv"]
    if stage=="v2":
        required+=[root/"outputs/submission_baseline.csv"]
    if stage=="v3":
        required+=[data/"original/credit_card_fraud_2026.csv",root/"outputs/submission_v2_recommended.csv"]
        selection=experiment_dir(root,"v2")/"selection.json"
        required.append(selection)
        if selection.is_file():
            weights=json.loads(selection.read_text(encoding="utf-8"))["selected_weights"]
            required+=[experiment_dir(root,"v2")/("oof_"+n+".npy") for n in weights]
    missing=[str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing prerequisites; see README:\n"+"\n".join(missing))
    for directory in [root/"work",root/"outputs",root/"work/temp"]:
        directory.mkdir(parents=True,exist_ok=True)
    os.environ["TEMP"]=os.environ["TMP"]=str(root/"work/temp")
    if stage=="benchmark": return root,data
    cache=experiment_dir(root,stage) if stage in ("v2","v3") else root/"work"
    cache.mkdir(parents=True,exist_ok=True)
    signature={
        "entry_sha256":hashlib.sha256(Path(script_file).read_bytes()).hexdigest(),
        "runtime_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs":{str(p.relative_to(root)) if p.is_relative_to(root) else p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in required},
        "threads":os.environ.get("KAGGLE_THREADS","historical_default"),
        "device":os.environ.get("KAGGLE_DEVICE","GPU").upper() if stage=="v3" else "CPU"
    }
    marker=cache/("."+stage+"_signature.json")
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8"))!=signature:
        raise RuntimeError("Code, data or options changed. Use a new --run-id and regenerate prerequisite stages.")
    marker.write_text(json.dumps(signature,indent=2),encoding="utf-8")
    return root,data
