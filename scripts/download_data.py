from pathlib import Path
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "nlp-getting-started"
OUT_DIR = Path("data/raw")

OUT_DIR.mkdir(parents=True, exist_ok=True)

api = KaggleApi()
api.authenticate()

# Скачает архив nlp-getting-started.zip в data/raw
api.competition_download_files(COMPETITION, path=str(OUT_DIR))

# Распаковать архив в data/raw
zip_path = OUT_DIR / f"{COMPETITION}.zip"
with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(OUT_DIR)
zip_path.unlink()  # удалить архив