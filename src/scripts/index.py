import json

from raglab1.settings import settings
from raglab1.storage.embedder import Embedder
from raglab1.storage.vector_store import VectorStore


def main() -> None:
    with settings.chunks_path.open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    vectors = Embedder().embed([chunk["text_for_embedding"] for chunk in chunks])

    with VectorStore(settings.vector_db_path) as store:
        store.rebuild(chunks, vectors)

    print(f"Indexed {len(chunks)} chunks into {settings.vector_db_path}")


if __name__ == "__main__":
    main()
