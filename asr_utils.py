import os
import shutil
from pathlib import Path

def process_dataset(FOLDER_ID):
    os.chdir("/content/asr")
    
    # Создаём папки
    os.makedirs("custom_dir/audio", exist_ok=True)
    os.makedirs("custom_dir/transcriptions", exist_ok=True)
    
    # Скачиваем (БЕЗ подавления ошибок)
    print("\n Скачивание датасета...")
    result = os.system(f"gdown --folder {FOLDER_ID} -O temp_download --remaining-ok")
    
    if result != 0:
        print(" Ошибка при скачивании!")
        return
    
    # Проверяем, что скачалось
    temp_path = Path("temp_download")
    print(f"\nСодержимое temp_download:")
    for item in temp_path.rglob("*"):
        if item.is_file():
            print(f"   {item}")
    
    # Ищем .flac файлы (в любом месте)
    audio_count = 0
    for file in temp_path.rglob("*.flac"):
        shutil.move(str(file), f"custom_dir/audio/{file.name}")
        audio_count += 1
        print(f"   Аудио: {file.name}")
    
    # Ищем .trans.txt файлы
    for trans_file in temp_path.rglob("*.trans.txt"):
        print(f"\n Обработка {trans_file.name}...")
        with open(trans_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    audio_id, text = parts[0], parts[1]
                    with open(f"custom_dir/transcriptions/{audio_id}.txt", 'w') as f_out:
                        f_out.write(text)
                    print(f"   Создан: {audio_id}.txt")
        trans_file.unlink()
    
    # Очистка
    shutil.rmtree("temp_download", ignore_errors=True)
    
    print(f"\n Аудио: {audio_count}")
    print(f" Транскрипции: {len(list(Path('custom_dir/transcriptions').glob('*.txt')))}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        process_dataset(sys.argv[1])
    else:
        print(" Укажите FOLDER_ID")
