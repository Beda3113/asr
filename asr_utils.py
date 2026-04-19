import shutil
from pathlib import Path

def process_dataset():
    """Обрабатывает скачанный датасет"""
    temp_path = Path("temp_download")
    audio_dir = Path("custom_dir/audio")
    trans_dir = Path("custom_dir/transcriptions")
    
    audio_dir.mkdir(parents=True, exist_ok=True)
    trans_dir.mkdir(parents=True, exist_ok=True)
    
    audio_count = 0
    for file in temp_path.rglob("*.flac"):
        shutil.move(str(file), audio_dir / file.name)
        audio_count += 1
    
    for file in temp_path.rglob("*.trans.txt"):
        with open(file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    with open(trans_dir / f"{parts[0]}.txt", 'w') as f_out:
                        f_out.write(parts[1])
        file.unlink()
    
    shutil.rmtree(temp_path, ignore_errors=True)
    print(f"   Аудио: {audio_count}, Транскрипции: {len(list(trans_dir.glob('*.txt')))}")

if __name__ == "__main__":
    process_dataset()
    print("\n Датасет готов к использованию!")
