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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TOKENS = 29

print(f"Device: {DEVICE}")

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def ctc_decode(log_probs, text_encoder, blank_idx=0):
    pred_ids = torch.argmax(log_probs, dim=-1)
    pred_ids = torch.unique_consecutive(pred_ids, dim=-1)
    pred_ids = pred_ids[pred_ids != blank_idx]
    if len(pred_ids) == 0:
        return ""
    pred_ids_list = pred_ids.cpu().tolist()
    decoded = ''.join([text_encoder.ind2char[idx] for idx in pred_ids_list])
    return decoded

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

results = []
print("\nRunning inference with CTC greedy decode...\n")

with torch.no_grad():
    for i, item in enumerate(dataset):
        spectrogram = item['spectrogram']
        if spectrogram.dim() == 3:
            spectrogram = spectrogram.squeeze(0)
        spectrogram = spectrogram.unsqueeze(0).to(DEVICE)
        spectrogram_length = torch.tensor([spectrogram.shape[-1]]).to(DEVICE)
        
        outputs = model(spectrogram=spectrogram, spectrogram_length=spectrogram_length)
        log_probs = outputs['log_probs'][0].cpu()
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
