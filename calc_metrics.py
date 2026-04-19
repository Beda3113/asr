import argparse
import json
from pathlib import Path
from collections import defaultdict

from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder.ctc_text_encoder import CTCTextEncoder


def load_predictions(pred_dir: Path) -> dict:
    """
    Load predictions from JSON files created by inference.py
    
    Args:
        pred_dir: Path to directory with prediction JSON files
    
    Returns:
        dict: {file_stem: {"prediction": str, "target": str}}
    """
    predictions = {}
    for json_file in pred_dir.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
            predictions[json_file.stem] = {
                "prediction": data.get("prediction", ""),
                "target": data.get("target", "")
            }
    return predictions


def load_transcriptions(transcription_dir: Path, text_encoder) -> dict:
    """
    Load ground truth transcriptions from .txt files
    
    Args:
        transcription_dir: Path to directory with .txt transcription files
        text_encoder: CTCTextEncoder for text normalization
    
    Returns:
        dict: {file_stem: normalized_text}
    """
    transcriptions = {}
    for txt_file in transcription_dir.glob("*.txt"):
        with open(txt_file, "r") as f:
            text = f.read().strip()
            normalized = text_encoder.normalize_text(text)
            transcriptions[txt_file.stem] = normalized
    return transcriptions


def main():
    parser = argparse.ArgumentParser(description="Calculate CER/WER from predictions")
    parser.add_argument(
        "--pred_dir", 
        type=str, 
        required=True,
        help="Path to directory with prediction JSON files"
    )
    parser.add_argument(
        "--target_dir", 
        type=str, 
        required=True,
        help="Path to directory with ground truth transcriptions (.txt files)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Path to save results (optional)"
    )
    parser.add_argument(
        "--detailed", 
        action="store_true",
        help="Print detailed results for each file"
    )
    
    args = parser.parse_args()
    
    pred_dir = Path(args.pred_dir)
    target_dir = Path(args.target_dir)
    
    # Initialize text encoder for normalization
    text_encoder = CTCTextEncoder()
    
    # Load data
    print(f"Loading predictions from: {pred_dir}")
    predictions = load_predictions(pred_dir)
    print(f"Found {len(predictions)} prediction files")
    
    print(f"Loading transcriptions from: {target_dir}")
    transcriptions = load_transcriptions(target_dir, text_encoder)
    print(f"Found {len(transcriptions)} transcription files")
    
    # Calculate metrics
    results = []
    all_cers = []
    all_wers = []
    
    for file_id, pred_data in predictions.items():
        pred_text = pred_data["prediction"]
        target_text = pred_data.get("target", "")
        
        # If target not in prediction file, try to load from transcription dir
        if not target_text and file_id in transcriptions:
            target_text = transcriptions[file_id]
        
        if not target_text:
            print(f"Warning: No target text for {file_id}")
            continue
        
        cer = calc_cer(target_text, pred_text)
        wer = calc_wer(target_text, pred_text)
        
        all_cers.append(cer)
        all_wers.append(wer)
        
        results.append({
            "file": file_id,
            "target": target_text,
            "prediction": pred_text,
            "cer": cer,
            "wer": wer
        })
    
    # Calculate averages
    avg_cer = sum(all_cers) / len(all_cers) if all_cers else 1.0
    avg_wer = sum(all_wers) / len(all_wers) if all_wers else 1.0
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Total files processed: {len(results)}")
    print(f"Average CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print(f"Average WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
    print("="*60)
    
    # Print detailed results if requested
    if args.detailed:
        print("\n" + "-"*60)
        print("DETAILED RESULTS")
        print("-"*60)
        for r in results:
            print(f"\nFile: {r['file']}")
            print(f"  Target:     {r['target']}")
            print(f"  Prediction: {r['prediction']}")
            print(f"  CER: {r['cer']:.4f} ({r['cer']*100:.2f}%)")
            print(f"  WER: {r['wer']:.4f} ({r['wer']*100:.2f}%)")
    
    # Save results if output path provided
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump({
                "summary": {
                    "avg_cer": avg_cer,
                    "avg_wer": avg_wer,
                    "total_files": len(results)
                },
                "details": results
            }, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
