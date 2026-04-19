import argparse
import json
from pathlib import Path
from src.metrics.utils import calc_cer, calc_wer

parser = argparse.ArgumentParser()
parser.add_argument("--pred_dir", type=str, required=True)
args = parser.parse_args()

pred_dir = Path(args.pred_dir)
cers, wers = [], []

for f in pred_dir.glob("*.json"):
    with open(f) as fp:
        data = json.load(fp)
    pred = data.get("prediction", "")
    target = data.get("target", "")
    cers.append(calc_cer(target, pred))
    wers.append(calc_wer(target, pred))

print(f"CER: {sum(cers)/len(cers):.4f}")
print(f"WER: {sum(wers)/len(wers):.4f}")
