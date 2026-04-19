import os
import sys
import gdown
from pathlib import Path

def download_dataset(folder_url):
    # Создаём папку для скачивания
    download_dir = Path("/content/asr/downloaded_dataset")
    download_dir.mkdir(parents=True, exist_ok=True)
    
    os.chdir(download_dir)
    print(f"Скачивание в: {os.getcwd()}")
    
    # Скачиваем
    gdown.download_folder(folder_url, quiet=False, use_cookies=False)
    
    print("Скачивание завершено")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        download_dataset(sys.argv[1])
    else:
        print("Укажите ссылку")
