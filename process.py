import os
import shutil
from pathlib import Path

def process_downloaded():
    os.chdir("/content/asr")
    
    # Создаём целевые папки
    os.makedirs("custom_dir/audio", exist_ok=True)
    os.makedirs("custom_dir/transcriptions", exist_ok=True)
    
    # Папка со скачанными файлами
    source_dir = Path("downloaded_dataset")
    
    if not source_dir.exists():
        print("Папка downloaded_dataset не найдена")
        print("Сначала запустите download.py")
        return
    
    audio_count = 0
    text_count = 0
    
    # Ищем файлы рекурсивно
    for file in source_dir.rglob("*"):
        if file.is_file():
            if file.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
                shutil.move(str(file), f"custom_dir/audio/{file.name}")
                audio_count += 1
                print(f"Аудио: {file.name}")
            elif file.suffix.lower() == '.txt':
                shutil.move(str(file), f"custom_dir/transcriptions/{file.name}")
                text_count += 1
                print(f"Транскрипция: {file.name}")
    
    # Обрабатываем общий файл транскрипций
    trans_dir = Path("custom_dir/transcriptions")
    for trans_file in trans_dir.glob("*.trans.txt"):
        print(f"\nОбработка {trans_file.name}...")
        with open(trans_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    audio_id, text = parts[0], parts[1]
                    with open(trans_dir / f"{audio_id}.txt", 'w') as f_out:
                        f_out.write(text)
                    print(f"Создан: {audio_id}.txt")
        trans_file.unlink()
    
    # Удаляем исходную папку
    shutil.rmtree(source_dir, ignore_errors=True)
    
    audio_files = list(Path("custom_dir/audio").glob("*.flac"))
    trans_files = list(Path("custom_dir/transcriptions").glob("*.txt"))
    
    print(f"\nИтог:")
    print(f"   Аудио: {len(audio_files)} файлов")
    print(f"   Транскрипции: {len(trans_files)} файлов")
    print("\nДатасет готов к использованию!")

if __name__ == "__main__":
    process_downloaded()
