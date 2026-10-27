from sentence_transformers import SentenceTransformer

from raglab1.settings import settings


class Embedder:
    """Turn texts into normalized vectors with a sentence-transformers model."""

    def __init__(self, model_id: str = settings.embed_model_id) -> None:
        self._model = SentenceTransformer(model_id)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 32,
        )
        return vectors.tolist()
