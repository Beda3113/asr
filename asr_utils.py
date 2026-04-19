import os
import shutil
from pathlib import Path

def process_dataset(FOLDER_ID):
    os.chdir("/content/asr")
    
 
    os.makedirs("custom_dir/audio", exist_ok=True)
    os.makedirs("custom_dir/transcriptions", exist_ok=True)
    
 
    print("\n Скачивание датасета...")
    os.system(f"gdown --folder {FOLDER_ID} -O temp_download --remaining-ok 2>/dev/null")
    

    source_dir = Path("temp_download/121123")
    
    if source_dir.exists():
 
        audio_count = 0
        for file in source_dir.glob("*.flac"):
            shutil.move(str(file), f"custom_dir/audio/{file.name}")
            audio_count += 1
            print(f"   Аудио: {file.name}")
        
     
        trans_file = source_dir / "84-121123.trans.txt"
        if trans_file.exists():
            with open(trans_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        with open(f"custom_dir/transcriptions/{parts[0]}.txt", 'w') as out:
                            out.write(parts[1])
                        print(f"   Транскрипция: {parts[0]}.txt")
            trans_file.unlink()
        
     
        shutil.rmtree("temp_download", ignore_errors=True)
        
        print(f"\n Аудио: {audio_count}, Транскрипции: {len(list(Path('custom_dir/transcriptions').glob('*.txt')))}")
    else:
        print(" Ошибка: папка не найдена")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        process_dataset(sys.argv[1])
    else:
        print(" Укажите FOLDER_ID")
