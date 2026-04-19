import os
import shutil
from pathlib import Path

os.chdir("/content/asr")
print(f"Текущая директория: {os.getcwd()}")

# Создаём папки
os.makedirs("custom_dir/audio", exist_ok=True)
os.makedirs("custom_dir/transcriptions", exist_ok=True)

# Скачиваем датасет с Google Drive
FOLDER_ID = "15WKxg15iBH6SAZDV71PTqY3FMs5Eq8wh"
print("\n Скачивание датасета...")
!gdown --folder {FOLDER_ID} -O temp_download --remaining-ok 2>/dev/null

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

# Удаляем временную папку
shutil.rmtree("temp_download")

print(f"\n Результат:")
print(f"   Аудио файлов: {audio_count}")
print(f"   Транскрипций: {text_count}")

# Проверяем
print(f"\n custom_dir/audio: {len(list(Path('custom_dir/audio').glob('*')))} файлов")
print(f" custom_dir/transcriptions: {len(list(Path('custom_dir/transcriptions').glob('*')))} файлов")
