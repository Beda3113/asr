import os
import re
import shutil
from pathlib import Path

def extract_folder_id(url):
    # Извлекаем ID из полной ссылки
    match = re.search(r'/folders/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    # Если уже ID
    return url

def process_dataset(FOLDER_URL):
    os.chdir("/content/asr")
    
    # Извлекаем ID
    FOLDER_ID = extract_folder_id(FOLDER_URL)
    print(f" ID папки: {FOLDER_ID}")
    
    # Создаём папки
    os.makedirs("custom_dir/audio", exist_ok=True)
    os.makedirs("custom_dir/transcriptions", exist_ok=True)
    
    print("\n Скачивание датасета...")
    os.system(f"gdown --folder {FOLDER_ID} -O temp_download --remaining-ok 2>/dev/null")
    
    # Перемещаем файлы
    temp_path = Path("temp_download")
    audio_count = 0
    text_count = 0
    
    for file in temp_path.rglob("*"):
        if file.is_file():
            if file.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
                shutil.move(str(file), f"custom_dir/audio/{file.name}")
                audio_count += 1
            elif file.suffix.lower() == '.txt':
                shutil.move(str(file), f"custom_dir/transcriptions/{file.name}")
                text_count += 1
    
    shutil.rmtree("temp_download", ignore_errors=True)
    
    print(f"\n📊 Результат:")
    print(f"   Аудио файлов: {audio_count}")
    print(f"   Транскрипций: {text_count}")
    
    # Обрабатываем транскрипции
    trans_dir = Path("custom_dir/transcriptions")
    for trans_file in trans_dir.glob("*.trans.txt"):
        print(f"\n📄 Обработка {trans_file.name}...")
        with open(trans_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    audio_id, text = parts[0], parts[1]
                    with open(trans_dir / f"{audio_id}.txt", 'w') as f_out:
                        f_out.write(text)
                    print(f"   Создан: {audio_id}.txt")
        trans_file.unlink()
    
    # Финальная проверка
    audio_files = list(Path("custom_dir/audio").glob("*.flac"))
    trans_files = list(Path("custom_dir/transcriptions").glob("*.txt"))
    print(f"\n Итог:")
    print(f"   Аудио: {len(audio_files)} файлов")
    print(f"   Транскрипции: {len(trans_files)} файлов")
    print("\n Датасет готов к использованию!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        process_dataset(sys.argv[1])
    else:
        print(" Укажите ссылку на папку Google Drive")
