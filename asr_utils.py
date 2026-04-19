# asr_utils.py
import shutil
from pathlib import Path

def process_downloaded_dataset(temp_path: str, target_dir: str):
    """Обрабатывает скачанный датасет: перемещает аудио и транскрипции"""
    temp_path = Path(temp_path)
    target_path = Path(target_dir)
    
    audio_count = 0
    text_count = 0
    
    for file in temp_path.rglob("*"):
        if file.is_file():
            if file.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
                shutil.move(str(file), target_path / "audio" / file.name)
                audio_count += 1
            elif file.suffix.lower() == '.txt':
                shutil.move(str(file), target_path / "transcriptions" / file.name)
                text_count += 1
    
    shutil.rmtree(temp_path)
    
    # Преобразуем общие файлы транскрипций (.trans.txt) в отдельные файлы
    trans_dir = target_path / "transcriptions"
    for trans_file in trans_dir.glob("*.trans.txt"):
        with open(trans_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    audio_id, text = parts[0], parts[1]
                    with open(trans_dir / f"{audio_id}.txt", 'w') as f_out:
                        f_out.write(text)
        trans_file.unlink()
    
    print(f"   Аудио файлов: {audio_count}")
    print(f"   Транскрипций: {text_count}")
