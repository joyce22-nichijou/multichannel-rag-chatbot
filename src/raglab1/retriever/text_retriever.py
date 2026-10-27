from raglab1.settings import settings
from raglab1.storage.embedder import Embedder
from raglab1.storage.vector_store import VectorStore


class TextRetriever:
    """Find the rulebook chunks that are most relevant to a question."""

    def __init__(self) -> None:
        self._embedder = Embedder()
        self._store = VectorStore(settings.vector_db_path)

    def retrieve(self, question: str, k: int = settings.retrieval_k) -> list[dict]:
        query_vector = self._embedder.embed([question])[0]
        return self._store.search(query_vector, k=k)
