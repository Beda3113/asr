%%writefile asr_utils.py
import os
import shutil
from pathlib import Path

def main(folder_id):
    os.chdir("/content/asr")
    
    print(" Скачивание датасета...")
    os.system(f"gdown --folder {folder_id} -O temp_download --remaining-ok 2>/dev/null")
    
    audio_dir = Path("custom_dir/audio")
    trans_dir = Path("custom_dir/transcriptions")
    audio_dir.mkdir(parents=True, exist_ok=True)
    trans_dir.mkdir(parents=True, exist_ok=True)
    
    temp_path = Path("temp_download")
    audio_count = 0
    
    for file in temp_path.rglob("*.flac"):
        shutil.move(str(file), audio_dir / file.name)
        audio_count += 1
        print(f"   Аудио: {file.name}")
    
    for trans_file in temp_path.rglob("*.trans.txt"):
        print(f"\n📄 Обработка {trans_file.name}...")
        with open(trans_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    audio_id, text = parts[0], parts[1]
                    output_file = trans_dir / f"{audio_id}.txt"
                    with open(output_file, 'w') as f_out:
                        f_out.write(text)
                    print(f"   Создан: {audio_id}.txt")
        trans_file.unlink()
    
    shutil.rmtree("temp_download", ignore_errors=True)
    
    print(f"\n Результат:")
    print(f"   Аудио: {audio_count}")
    print(f"   Транскрипции: {len(list(trans_dir.glob('*.txt')))}")
    print("\n Датасет готов к использованию!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print(" Ошибка: укажите FOLDER_ID")
        print("Пример: !python asr_utils.py 15WKxg15iBH6SAZDV71PTqY3FMs5Eq8wh")
