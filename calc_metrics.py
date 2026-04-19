import os
import sys
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

os.chdir("/content/asr")
sys.path.insert(0, '/content/asr')

from src.metrics.utils import calc_cer, calc_wer

def load_predictions(pred_dir):
    pred_dir = Path(pred_dir)
    results = []
    for json_file in pred_dir.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        results.append({
            'file': json_file.stem,
            'prediction': data.get('prediction', ''),
            'target': data.get('target', '')
        })
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, default="/content/asr/custom_predictions")
    parser.add_argument("--verbose", action="store_true")
    
    # Используем parse_known_args() вместо parse_args()
    # Это игнорирует неизвестные аргументы (например, -f от Jupyter)
    args, unknown = parser.parse_known_args()
    
    if unknown:
        print(f"Ignoring Jupyter arguments: {unknown}")
    
    print(f"Loading predictions from: {args.pred_dir}")
    results = load_predictions(args.pred_dir)
    print(f"Found {len(results)} prediction files")
    
    print("\nRECOGNITION RESULTS")
    print("="*60)
    
    total_cer, total_wer = 0.0, 0.0
    cer_list, wer_list = [], []
    
    for res in results:
        cer = calc_cer(res['target'], res['prediction'])
        wer = calc_wer(res['target'], res['prediction'])
        total_cer += cer
        total_wer += wer
        cer_list.append(cer)
        wer_list.append(wer)
        if args.verbose:
            print(f"{res['file']}: CER={cer:.2%}, WER={wer:.2%}")
    
    if not args.verbose:
        print("(Use --verbose to see per-file results)")
    
    print("-"*50)
    print(f"Average CER: {total_cer/len(results):.2%}")
    print(f"Average WER: {total_wer/len(results):.2%}")
    
    cer_list.sort()
    wer_list.sort()
    print(f"Median CER: {cer_list[len(cer_list)//2]:.2%}")
    print(f"Median WER: {wer_list[len(wer_list)//2]:.2%}")
    print("="*60)
    
    print("\nRECOGNITION EXAMPLES:")
    print("-"*60)
    
    wer_list_sorted = [(wer, idx) for idx, wer in enumerate(wer_list)]
    wer_list_sorted.sort()
    
    best_idx = wer_list_sorted[0][1]
    worst_idx = wer_list_sorted[-1][1]
    
    print(f"\nBest (WER={wer_list[best_idx]:.2%}):")
    print(f"   Target: {results[best_idx]['target']}")
    print(f"   Pred:   {results[best_idx]['prediction']}")
    
    print(f"\nWorst (WER={wer_list[worst_idx]:.2%}):")
    print(f"   Target: {results[worst_idx]['target']}")
    print(f"   Pred:   {results[worst_idx]['prediction']}")

if __name__ == "__main__":
    main()
