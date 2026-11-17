"""Run only an explicitly selected experiment stage."""
from pathlib import Path
import argparse,os,subprocess,sys

def main():
    parser=argparse.ArgumentParser(description="Reproduce Kaggle S6E11 experiments.")
    parser.add_argument("--stage",required=True,choices=["baseline","v2","v3","all","benchmark"])
    parser.add_argument("--data-dir",type=Path,help="CSV directory; default ./data")
    parser.add_argument("--threads",type=int,help="CPU thread limit")
    parser.add_argument("--cpu",action="store_true",help="Use CPU for v3; results can differ")
    parser.add_argument("--run-id",default="default",help="Separate caches for changed code/data/options")
    parser.add_argument("--dry-run",action="store_true",help="Print commands without executing them")
    args=parser.parse_args()
    if args.threads is not None and args.threads<1: parser.error("--threads must be positive")
    root=Path(__file__).resolve().parent
    env=os.environ.copy()
    if args.data_dir: env["KAGGLE_DATA_DIR"]=str(args.data_dir.resolve())
    if args.threads: env["KAGGLE_THREADS"]=str(args.threads)
    env["KAGGLE_DEVICE"]="CPU" if args.cpu else "GPU"
    env["KAGGLE_RUN_ID"]=args.run_id
    stages=["baseline","v2","v3"] if args.stage=="all" else [args.stage]
    modules={"baseline":"scripts.train_baseline","v2":"scripts.experiment_v2","v3":"scripts.experiment_v3","benchmark":"scripts.benchmark_inference_v2"}
    for stage in stages:
        command=[sys.executable,"-m",modules[stage]]
        print("Run:",subprocess.list2cmdline(command),flush=True)
        if not args.dry_run: subprocess.run(command,cwd=root,env=env,check=True)

if __name__=="__main__": main()
