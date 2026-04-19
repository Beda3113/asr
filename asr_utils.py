import os
import sys
import shutil
from pathlib import Path

def main():
    # Получаем FOLDER_ID из аргументов командной строки
    if len(sys.argv) < 2:
        print(" Ошибка: укажите FOLDER_ID")
        print("Пример: !python asr_utils.py 15WKxg15iBH6SAZDV71PTqY3FMs5Eq8wh")
        sys.exit(1)
    
    FOLDER_ID = sys.argv[1]
    
    os.chdir("/content/asr")
    print(f"Текущая директория: {os.getcwd()}")
    
    # Создаём папки
    os.makedirs("custom_dir/audio", exist_ok=True)
    os.makedirs("custom_dir/transcriptions", exist_ok=True)
    
    # Скачиваем датасет с Google Drive
    print("\n📥 Скачивание датасета...")
    os.system(f"gdown --folder {FOLDER_ID} -O temp_download --remaining-ok 2>/dev/null")
    
    # Файлы лежат в temp_download/121123/
    source_dir = Path("temp_download/121123")
    
    if source_dir.exists():
        # Перемещаем аудио файлы
        audio_count = 0
        for file in source_dir.glob("*.flac"):
            shutil.move(str(file), f"custom_dir/audio/{file.name}")
            audio_count += 1
            print(f"   Аудио: {file.name}")
        
        # Обрабатываем транскрипции
        trans_file = source_dir / "84-121123.trans.txt"
        if trans_file.exists():
            print(f"\n📄 Обработка транскрипций...")
            with open(trans_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        audio_id, text = parts[0], parts[1]
                        with open(f"custom_dir/transcriptions/{audio_id}.txt", 'w') as f_out:
                            f_out.write(text)
                        print(f"   Создан: {audio_id}.txt")
            trans_file.unlink()
        
        # Удаляем временную папку
        shutil.rmtree("temp_download", ignore_errors=True)
        
        print(f"\n Результат:")
        print(f"   Аудио файлов: {audio_count}")
        print(f"   Транскрипций: {len(list(Path('custom_dir/transcriptions').glob('*.txt')))}")
    else:
        print(" Папка temp_download/121123 не найдена")
    
    # Финальная проверка
    audio_files = list(Path("custom_dir/audio").glob("*.flac"))
    trans_files = list(Path("custom_dir/transcriptions").glob("*.txt"))
    print(f"\n Итог:")
    print(f"   Аудио: {len(audio_files)} файлов")
    print(f"   Транскрипции: {len(trans_files)} файлов")
    print("\n Датасет готов к использованию!")

if __name__ == "__main__":
    main()
