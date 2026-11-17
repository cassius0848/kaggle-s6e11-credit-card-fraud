"""Download the host-linked CC0 original CSV; no archive code is executed."""
from pathlib import Path
import argparse,hashlib,json,urllib.request,zipfile,io

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=Path,default=Path(__file__).resolve().parents[1]/"data")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    provenance=json.loads((root/"reports/data_provenance.json").read_text(encoding="utf-8"))
    with urllib.request.urlopen(provenance["download_url"],timeout=60) as response:
        payload=response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names=[n for n in archive.namelist() if Path(n).name=="credit_card_fraud_2026.csv"]
        if len(names)!=1: raise RuntimeError("Expected exactly one original CSV")
        content=archive.read(names[0])
    digest=hashlib.sha256(content).hexdigest()
    if digest!=provenance["sha256"]:
        raise RuntimeError("Public dataset version changed; inspect before updating its expected hash")
    destination=args.data_dir.resolve()/"original"
    destination.mkdir(parents=True,exist_ok=True)
    (destination/"credit_card_fraud_2026.csv").write_bytes(content)
    (destination/"provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    print("Downloaded original CSV; SHA-256",digest)

if __name__=="__main__": main()
