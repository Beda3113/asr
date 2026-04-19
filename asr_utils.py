# asr_utils.py
import shutil
from pathlib import Path

def process_dataset(temp_path: str, target_dir: str):
    """Обрабатывает скачанный датасет: перемещает аудио и транскрипции"""
    temp_path = Path(temp_path)
    target_dir = Path(target_dir)
    
    # Создаём папки
    audio_dir = target_dir / "audio"
    trans_dir = target_dir / "transcriptions"
    audio_dir.mkdir(parents=True, exist_ok=True)
    trans_dir.mkdir(parents=True, exist_ok=True)
    
    audio_count = 0
    text_count = 0
    
    # Перемещаем файлы
    for file in temp_path.rglob("*"):
        if file.is_file():
            if file.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
                shutil.move(str(file), audio_dir / file.name)
                audio_count += 1
            elif file.suffix.lower() == '.txt':
                shutil.move(str(file), trans_dir / file.name)
                text_count += 1
    
    # Удаляем временную папку
    shutil.rmtree(temp_path, ignore_errors=True)
    
    # Обрабатываем общий файл транскрипций
    for trans_file in trans_dir.glob("*.trans.txt"):
        with open(trans_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    audio_id, text = parts[0], parts[1]
                    with open(trans_dir / f"{audio_id}.txt", 'w') as f_out:
                        f_out.write(text)
        trans_file.unlink()
    
    print(f"   Аудио: {audio_count}, Транскрипции: {len(list(trans_dir.glob('*.txt')))}")
    return audio_count, text_count
