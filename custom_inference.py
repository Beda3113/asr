import os
import sys
import re
import torch
import torchaudio
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

os.chdir("/content/asr")
sys.path.insert(0, '/content/asr')

from src.model.deepspeech2 import DeepSpeech2
from src.text_encoder.ctc_text_encoder import CTCTextEncoder
from src.datasets.custom_dir_audio_dataset import CustomDirAudioDataset
from src.metrics.utils import calc_cer, calc_wer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEAM_SIZE = 15
N_TOKENS = 29

print(f"Device: {DEVICE}")
print(f"Beam Size: {BEAM_SIZE}")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


print("\nLoading model...")

model = DeepSpeech2(
    n_tokens=N_TOKENS,
    n_feats=80, 
    dim=512, 
    n_channels=48,
    gru_layers=5, 
    dropout=0.3
)

checkpoint = torch.load("checkpoint-epoch50.pth", map_location=DEVICE)
print(f"Checkpoint: epoch {checkpoint.get('epoch', 'unknown')}")

# Weight adaptation
checkpoint_state = checkpoint['state_dict']
if checkpoint_state['fc.weight'].shape[0] == 28 and model.fc.weight.shape[0] == 29:
    print("Adapting weights: 28 -> 29 classes")
    new_state_dict = model.state_dict()
    for key in checkpoint_state:
        if key not in ['fc.weight', 'fc.bias']:
            new_state_dict[key] = checkpoint_state[key]
    new_state_dict['fc.weight'][:28, :] = checkpoint_state['fc.weight']
    new_state_dict['fc.bias'][:28] = checkpoint_state['fc.bias']
    model.load_state_dict(new_state_dict)
else:
    model.load_state_dict(checkpoint_state)

model.to(DEVICE)
model.eval()
print("Model loaded")


text_encoder = CTCTextEncoder()


AUDIO_DIR = "/content/asr/custom_dir/audio"
TRANS_DIR = "/content/asr/custom_dir/transcriptions"

print("\nLoading dataset...")
dataset = CustomDirAudioDataset(
    audio_dir=AUDIO_DIR,
    transcription_dir=TRANS_DIR,
    text_encoder=text_encoder,
    instance_transforms={
        "get_spectrogram": torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=80, n_fft=400, hop_length=160, power=1.0
        )
    }
)
print(f"Dataset: {len(dataset)} examples")

def ctc_decode(log_probs, text_encoder, blank_idx=0):
    """
    CTC greedy decode (argmax + remove blanks + remove duplicates)
    
    Args:
        log_probs: torch.Tensor [T, n_tokens] - log probabilities
        text_encoder: CTCTextEncoder instance
        blank_idx: int - index of blank token (default 0)
    
    Returns:
        str: decoded text
    """
    # Argmax over last dimension
    pred_ids = torch.argmax(log_probs, dim=-1)  # [T]
    
    # Remove consecutive duplicates
    pred_ids = torch.unique_consecutive(pred_ids, dim=-1)
    
    # Remove blank tokens
    pred_ids = pred_ids[pred_ids != blank_idx]
    
    # Convert tensor to Python list
    pred_ids_list = pred_ids.cpu().tolist()
    
    # Decode to text
    decoded = ''.join([text_encoder.ind2char[idx] for idx in pred_ids_list])
    
    return decoded


results = []
print("\nRunning inference with CTC greedy decode...\n")

with torch.no_grad():
    for i, item in enumerate(dataset):
        spectrogram = item['spectrogram']
        
        if spectrogram.dim() == 3:
            spectrogram = spectrogram.squeeze(0)
        spectrogram = spectrogram.unsqueeze(0).to(DEVICE)
        spectrogram_length = torch.tensor([spectrogram.shape[-1]]).to(DEVICE)
        
        outputs = model(
            spectrogram=spectrogram, 
            spectrogram_length=spectrogram_length
        )
        
        log_probs = outputs['log_probs'][0].cpu()  # [T, n_tokens]
        
        # CTC decode (greedy)
        predicted_text = ctc_decode(log_probs, text_encoder, blank_idx=0)
        
        target_text = normalize_text(item['text'])
        
        results.append({
            'file': Path(item['audio_path']).stem,
            'prediction': predicted_text,
            'target': target_text,
        })
        
        if (i + 1) % 5 == 0 or i == 0 or i == len(dataset) - 1:
            print(f"[{i+1}/{len(dataset)}] {Path(item['audio_path']).name}")
            print(f"   Target: {target_text[:80]}")
            print(f"   Pred:   {predicted_text[:80]}\n")


SAVE_DIR = Path("/content/asr/custom_predictions")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

for res in results:
    with open(SAVE_DIR / f"{res['file']}.json", 'w', encoding='utf-8') as f:
        json.dump({
            "prediction": res['prediction'],
            "target": res['target']
        }, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(results)} predictions to {SAVE_DIR}")


print("RECOGNITION RESULTS")

total_cer, total_wer = 0.0, 0.0
cer_list, wer_list = [], []

for res in results:
    cer = calc_cer(res['target'], res['prediction'])
    wer = calc_wer(res['target'], res['prediction'])
    total_cer += cer
    total_wer += wer
    cer_list.append(cer)
    wer_list.append(wer)
    print(f"{res['file']}: CER={cer:.2%}, WER={wer:.2%}")

print(f"Average CER: {total_cer/len(results):.2%}")
print(f"Average WER: {total_wer/len(results):.2%}")

# Median values
cer_list.sort()
wer_list.sort()
print(f"Median CER: {cer_list[len(cer_list)//2]:.2%}")
print(f"Median WER: {wer_list[len(wer_list)//2]:.2%}")


print("\nRECOGNITION EXAMPLES:")

# Best and worst examples
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
