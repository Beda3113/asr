import pyarrow.parquet as pq
from pathlib import Path
import soundfile as sf
import numpy as np

print("Loading parquet files directly...")

# Путь к кэшу datasets
cache_path = Path.home() / ".cache/huggingface/datasets/librispeech_asr/clean/1.0.0"

save_dir = Path("./data/datasets/librispeech/train-clean-100")
save_dir.mkdir(parents=True, exist_ok=True)

# Ищем parquet файлы для train.100
parquet_files = list(cache_path.glob("*.parquet"))

for parquet_file in parquet_files:
    if "train.100" in str(parquet_file):
        print(f"Processing {parquet_file}...")
        table = pq.read_table(parquet_file)
        df = table.to_pandas()
        
        for idx, row in df.iterrows():
            file_id = row["id"]
            text = row["text"].lower()
            audio_path = row["audio"]["path"]
            
            parts = file_id.split("-")
            if len(parts) >= 2:
                speaker_dir = save_dir / parts[0]
                chapter_dir = speaker_dir / parts[1]
                chapter_dir.mkdir(parents=True, exist_ok=True)
                
                txt_path = chapter_dir / f"{file_id}.txt"
                with open(txt_path, "w") as f:
                    f.write(text)
                
                # Копируем или создаём заглушку для аудио
                # Так как аудио нет, создаём пустой файл
                audio_file = chapter_dir / f"{file_id}.flac"
                audio_file.touch()
            
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1} samples in this file...")

print("Done! Note: audio files are empty placeholders.")