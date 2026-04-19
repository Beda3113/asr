import torchaudio
from pathlib import Path
from src.datasets.base_dataset import BaseDataset


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, audio_dir, transcription_dir=None, *args, **kwargs):
        data = []
        
        # Проходим по всем аудиофайлам
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        
        for path in Path(audio_dir).iterdir():
            if path.suffix.lower() in audio_extensions:
                # Получаем ID файла (без расширения)
                file_id = path.stem
                
                # Ищем транскрипцию
                text = ""
                if transcription_dir and Path(transcription_dir).exists():
                    # Сначала ищем файл с таким же именем
                    trans_path = Path(transcription_dir) / f"{file_id}.txt"
                    if trans_path.exists():
                        with open(trans_path, 'r') as f:
                            text = f.read().strip()
                    else:
                        # Пробуем найти в общих файлах *.trans.txt
                        for trans_file in Path(transcription_dir).glob("*.trans.txt"):
                            with open(trans_file, 'r') as f:
                                for line in f:
                                    parts = line.strip().split(' ', 1)
                                    if len(parts) == 2 and parts[0] == file_id:
                                        text = parts[1]
                                        break
                            if text:
                                break
                
                # Получаем длину аудио
                try:
                    t_info = torchaudio.info(str(path))
                    audio_len = t_info.num_frames / t_info.sample_rate
                except:
                    # Если не удалось получить информацию, пробуем загрузить
                    try:
                        waveform, sr = torchaudio.load(str(path))
                        audio_len = waveform.shape[1] / sr
                    except:
                        audio_len = 0.0
                
                # Добавляем запись
                data.append({
                    "path": str(path.absolute().resolve()),
                    "text": text,
                    "audio_len": audio_len,
                })
        
        super().__init__(data, *args, **kwargs)
