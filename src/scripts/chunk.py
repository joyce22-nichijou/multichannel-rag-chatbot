from pathlib import Path

from raglab1.chunker.docling_chunker import chunk_document
from raglab1.settings import settings


def main() -> None:
    n = chunk_document(
        json_path=Path("data") / "processed" / "Core Rules.json",
        output_path=settings.chunks_path,
    )
    print(f"Saved {n} chunks to {settings.chunks_path}")


if __name__ == "__main__":
    main()
