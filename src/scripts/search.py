import sys

from raglab1.settings import settings
from raglab1.storage.embedder import Embedder
from raglab1.storage.vector_store import VectorStore


def main() -> None:
    question = " ".join(sys.argv[1:]) or "How many points is a Strike Force game?"
    query_vector = Embedder().embed([question])[0]

    with VectorStore(settings.vector_db_path) as store:
        results = store.search(query_vector, k=5)

    print(f"Question: {question}\n")
    for r in results:
        print(f"{r['score']:.3f} | {' > '.join(r['headings'])} | page {r['pages']}")
        print(f"      {r['text'][:150]!r}\n")


if __name__ == "__main__":
    main()
