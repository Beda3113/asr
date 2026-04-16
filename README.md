# asr

# Структура шаблона
```
pytorch_project_template/                    # Корень проекта
│
├── inference.py                             # Запуск распознавания (инференс)
├── train.py                                 # Запуск обучения модели
│
├── src/                                     # Весь исходный код
│   │
│   ├── datasets/                            # Работа с датасетами
│   │   ├── base_dataset.py                  # Базовый класс для всех датасетов
│   │   ├── librispeech_dataset.py           # LibriSpeech (основной датасет)
│   │   ├── common_voice.py                  # Mozilla Common Voice
│   │   ├── custom_dir_audio_dataset.py      # Пользовательский датасет (для сдачи)
│   │   ├── collate.py                       # Склеивание батчей с паддингом
│   │   └── data_utils.py                    # Создание даталоадеров
│   │
│   ├── model/                               # Архитектуры нейросетей
│   │   ├── baseline_model.py                # Простая MLP для отладки
│   │   └── __init__.py                      # Экспорт моделей
│   │
│   ├── text_encoder/                        # Токенизация текста
│   │   ├── ctc_text_encoder.py              # Символьный энкодер + CTC декодирование
│   │   └── __init__.py
│   │
│   ├── loss/                                # Функции потерь
│   │   ├── ctc_loss.py                      # CTC Loss для обучения
│   │   └── __init__.py
│   │
│   ├── metrics/                             # Метрики качества
│   │   ├── base_metric.py                   # Абстрактный класс метрики
│   │   ├── cer.py                           # Character Error Rate
│   │   ├── wer.py                           # Word Error Rate
│   │   ├── tracker.py                       # Агрегация метрик по эпохе
│   │   └── utils.py                         # calc_cer, calc_wer через editdistance
│   │
│   ├── trainer/                             # Циклы обучения и инференса
│   │   ├── base_trainer.py                  # Базовый класс (сохранение/загрузка)
│   │   ├── trainer.py                       # Полный цикл обучения
│   │   └── inferencer.py                    # Логика инференса
│   │
│   ├── transforms/                          # Аугментации
│   │   ├── wav_augs/                        # Аугментации аудио (временная область)
│   │   │   ├── gain.py                      # Изменение громкости
│   │   │   └── __init__.py
│   │   └── spec_augs/                       # Аугментации спектрограмм
│   │       └── __init__.py
│   │
│   ├── logger/                              # Логирование экспериментов
│   │   ├── logger.py                        # Настройка logging
│   │   ├── wandb.py                         # Weights & Biases
│   │   ├── cometml.py                       # CometML
│   │   ├── utils.py                         # plot_spectrogram, plot_images
│   │   └── __init__.py
│   │
│   └── utils/                               # Вспомогательные функции
│       ├── init_utils.py                    # seed, логи, сохранение чекпоинтов
│       ├── io_utils.py                      # ROOT_PATH, read_json, write_json
│       └── __init__.py
│
├── configs/                                 # Hydra конфиги (НЕ ВЫВЕЛИСЬ, но должны быть)
├── requirements.txt                         # Зависимости
├── README.md                                # Документация
├── LICENSE                                  # Лицензия
├── .pre-commit-config.yaml                  # Pre-commit хуки (black, isort)
├── .flake8                                  # Линтер flake8
└── .gitignore                               # Игнорируемые файлы
```

PyTorch Project Template — это готовый промышленный шаблон для тренировки и инференса нейросетей с организацией кода по стандартам ML-инженерии.
Это система распознавания речи (ASR) на основе DeepSpeech2 с CTC-декодированием. Модель принимает на вход аудиофайл и выдаёт текстовую транскрипцию.

## Основные возможности

| Компонент | Назначение |
|-----------|------------|
| **Hydra** | Управление конфигами (YAML) — меняй параметры без правки кода |
| **PyTorch Lightning-подобный Trainer** | Циклы обучения/валидации, логирование, чекпоинты |
| **Абстрактные классы** | `BaseDataset`, `BaseMetric`, `BaseTrainer` — легко расширять |
| **Готовые ASR-компоненты** | CTC Loss, WER/CER, LibriSpeech, Beam Search |
| **Логирование** | W&B, CometML, TensorBoard из коробки |
| **Pre-commit** | Автоформатирование (black, isort) перед каждым коммитом |

## Ключевые файлы

| Файл | Назначение |
|------|------------|
| `train.py` | Запуск обучения |
| `inference.py` | Запуск распознавания |
| `src/trainer/trainer.py` | Логика одного шага (forward, backward, оптимизация) |
| `src/datasets/base_dataset.py` | Загрузка аудио → спектрограмма → текст |
| `src/text_encoder/ctc_text_encoder.py` | Символьная токенизация + Beam Search |
| `configs/` | Все гиперпараметры (модель, lr, датасет, аугментации) |

## Структура модулей

| Папка | Ответственность |
|-------|------------------|
| `datasets/` | Загрузка данных, аугментации, паддинг батчей |
| `model/` | Архитектура нейросети (DeepSpeech2, Baseline) |
| `text_encoder/` | Преобразование текст ↔ числа, CTC/Beam Search декодинг |
| `loss/` | Функция потерь (CTC Loss) |
| `metrics/` | Character Error Rate, Word Error Rate |
| `trainer/` | Цикл обучения, валидация, сохранение чекпоинтов |
| `transforms/` | Аугментации спектрограмм и аудио (SpecAugment) |
| `logger/` | Логирование в W&B/Comet/TensorBoard |
| `utils/` | Вспомогательное: seed, пути, чтение JSON |
