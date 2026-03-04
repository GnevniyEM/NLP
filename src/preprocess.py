from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME_768 = "sentence-transformers/all-mpnet-base-v2"  # 768-dim


def extract_embeddings(
    texts: Iterable[str],
    model_name: str = MODEL_NAME_768,
    batch_size: int = 32,  
    device: Optional[str] = None,  # "cpu" / "cuda"
) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return emb.astype(np.float32)


def save_embeddings(path: str | Path, embeddings: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
