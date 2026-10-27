from pathlib import Path

from raglab1.chunker.docling_chunker import chunk_document


def main() -> None:
    n = chunk_document(
        json_path=Path("data") / "processed" / "Core Rules.json",
        output_path=Path("data") / "processed" / "chunks.jsonl",
    )
    print(f"Saved {n} chunks")


if __name__ == "__main__":
    main()
