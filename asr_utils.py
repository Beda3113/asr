# asr_utils.py
import shutil
import os
from pathlib import Path

def process_downloaded_dataset(temp_path: str, target_dir: str):
    """Обрабатывает скачанный датасет: перемещает аудио и транскрипции"""
    temp_path = Path(temp_path)
    target_path = Path(target_dir)
    
    # Создаём папки
    audio_dir = target_path / "audio"
    trans_dir = target_path / "transcriptions"
    audio_dir.mkdir(parents=True, exist_ok=True)
    trans_dir.mkdir(parents=True, exist_ok=True)
    
    audio_count = 0
    text_count = 0
    
    # Обходим все файлы во временной папке
    for file in temp_path.rglob("*"):
        if file.is_file():
            # Аудио файлы
            if file.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
                dest = audio_dir / file.name
                shutil.move(str(file), str(dest))
                audio_count += 1
                print(f"   Аудио: {file.name} -> {dest}")
            
            # Транскрипции
            elif file.suffix.lower() == '.txt':
                dest = trans_dir / file.name
                shutil.move(str(file), str(dest))
                text_count += 1
                print(f"   Транскрипция: {file.name} -> {dest}")
    
    # Удаляем временную папку
    shutil.rmtree(temp_path, ignore_errors=True)
    
    # Обрабатываем общий файл транскрипций (84-121123.trans.txt)
    trans_file = trans_dir / "84-121123.trans.txt"
    if trans_file.exists():
        print(f"\n Обработка {trans_file.name}...")
        with open(trans_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    audio_id = parts[0]
                    text = parts[1]
                    output_file = trans_dir / f"{audio_id}.txt"
                    with open(output_file, 'w') as f_out:
                        f_out.write(text)
                    print(f"   Создан: {audio_id}.txt")
        # Удаляем старый общий файл
        trans_file.unlink()
        print(f"   Удалён: {trans_file.name}")
    
    print(f"\n Результат:")
    print(f"   Аудио файлов: {audio_count}")
    print(f"   Транскрипций: {len(list(trans_dir.glob('*.txt')))}")
    
    return audio_count, text_count
