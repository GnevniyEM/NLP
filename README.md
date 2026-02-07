# NLP Getting Started (Kaggle): Text Classification

Проект решает задачу соревнования Kaggle **“NLP Getting Started”** (`nlp-getting-started`): бинарная классификация твитов на английском языке (`target ∈ {0, 1}`).

## Структура проекта

```
data/
  raw/
  processed/
logs/
scripts/
  __init__.py
  download_data.py
  baseline.py
  preprocess.py
  train_emb_lr.py
src/
  data/
    init.py
    io.py
  model/
    init.py
    emb_lr.py
  utils/
    init.py
    run_logger.py
    preprocess.py
requirements.txt
```

## Как запустить проект

Команды выполнять из корня проекта.

### 0) Установить зависимости

```
pip install -r requirements.txt
```

### 1) Скачать и распаковать датасет

Скрипт скачает архив соревнования в `data/raw/`, распакует и удалит `.zip`.

```
python -m scripts.download_data
```

После выполнения должны быть файлы:

```
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

### 2) Classic NLP / Classic ML baseline (TF-IDF → LogisticRegression → GridSearchCV)

```
python -m scripts.baseline
```

Что делает:
- лёгкая очистка текста (URL/упоминания/числа/пробелы)
- TF-IDF: word ngrams + char ngrams
- LogisticRegression + GridSearchCV
- метрики: **F1**, **Accuracy**
- артефакты в `logs/`:
  - `run_<run_id>.log`
  - `run_<run_id>_cv_results.csv`
  - `run_<run_id>_summary.json`
- `submission.csv` в корне проекта

### 3) Preprocess для DL+classic ML (извлечение эмбеддингов → .npy)

```
python -m scripts.preprocess
```

Что делает:
- берёт только колонку `text`
- получает эмбеддинги pretrained модели (текущая конфигурация: **768-dim**, MPNet)
- сохраняет:
  - `data/processed/train_emb.npy`
  - `data/processed/test_emb.npy`

Проверка размерностей:

```
python -c "import numpy as np; print(np.load('data/processed/train_emb.npy').shape); print(np.load('data/processed/test_emb.npy').shape)"
```

Ожидаемо (для 768-dim): `(7613, 768)` и `(3263, 768)`.

### 4) Обучение на эмбеддингах (LogisticRegression → GridSearchCV → submission)

```
python -m scripts.train_emb_lr
```

Что делает:
- читает `data/processed/train_emb.npy`, `data/processed/test_emb.npy`
- берёт `target` из `data/raw/train.csv` и `id` из `data/raw/test.csv`
- LogisticRegression + GridSearchCV по `C`
- метрики: **F1**, **Accuracy** (refit по F1)
- сохраняет:
  - `submission_emb_lr.csv` в корне
  - артефакты в `logs/emb_lr/<timestamp>/`:
    - `run.log`
    - `config.json`
    - `cv_results.csv`
    - `summary.json`
    - копия `submission_emb_lr.csv`

## Что делали в частях проекта

### Часть 1 — Classic NLP / Classic ML
Pipeline:
1) очистка/нормализация текста (регэксп-замены + нормализация пробелов)
2) TF-IDF (word + char)
3) LogisticRegression
4) GridSearchCV (подбор гиперпараметров)
5) логирование и сохранение артефактов

Запуск: `python -m scripts.baseline`.

### Часть 2 — DL + classic ML (эмбеддинги → классический ML)
Pipeline:
1) pretrained модель (через Hugging Face / sentence-transformers) как feature extractor
2) эмбеддинги для каждого текста (матрица признаков)
3) LogisticRegression на эмбеддингах
4) GridSearchCV по `C`
5) логирование и сохранение артефактов

Запуски:
- эмбеддинги: `python -m scripts.preprocess`
- обучение: `python -m scripts.train_emb_lr`

## Результаты и как их посмотреть

### Classic ML (TF-IDF baseline)
Числа зависят от конкретного запуска и сохраняются в `logs/run_<run_id>_summary.json`.

Быстро вывести метрики **последнего** baseline-запуска:

```
python -c "import json; from pathlib import Path; p=max(Path('logs').glob('run_*_summary.json'), key=lambda x: x.stat().st_mtime); d=json.loads(p.read_text(encoding='utf-8')); print('summary=', p); print('best_params=', d.get('best_params')); print('f1=', d.get('cv_mean_f1')); print('accuracy=', d.get('cv_mean_accuracy'))"
```

### DL + classic ML (эмбеддинги + LR)
1) Исторический результат для варианта **384-dim** (mean CV при refit по F1):
- **Best CV F1 (mean)**: `0.774175`
- **Best CV Accuracy (mean)**: `0.812820`

2) Текущие эмбеддинги в проекте — **768-dim**. Результаты каждого запуска сохраняются в `logs/emb_lr/<timestamp>/summary.json` и выводятся в терминал.

Быстро вывести метрики **последнего** запуска `train_emb_lr.py`:

```
python -c "import json; from pathlib import Path; p=max(Path('logs/emb_lr').glob('*/*summary.json'), key=lambda x: x.stat().st_mtime); d=json.loads(p.read_text(encoding='utf-8')); print('summary=', p); print('best_params=', d.get('best_params')); print('f1=', d.get('cv_mean_f1')); print('accuracy=', d.get('cv_mean_accuracy'))"
```

## Выводы (на текущем этапе)
- Classic ML (TF-IDF) — сильный и быстрый baseline: легко тюнить, быстро запускать, хорошо интерпретируется.
- DL + classic ML (эмбеддинги → LR) — альтернативный baseline без ручной генерации признаков, но зависит от pretrained модели и затрат на извлечение эмбеддингов.
- Все эксперименты сохраняют артефакты (конфиги/метрики/результаты CV/сабмиты) в `logs/`.
