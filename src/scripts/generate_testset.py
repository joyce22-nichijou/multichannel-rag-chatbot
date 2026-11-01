import argparse
import json

from raglab1.evaluation.testset import generate_testset, save_jsonl, select_chunks
from raglab1.llm.gemini import GeminiLLM
from raglab1.settings import settings
from raglab1.storage.embedder import Embedder


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the RAGAS test set.")
    parser.add_argument("--size", type=int, default=15, help="number of questions")
    parser.add_argument("--chunks", type=int, default=20, help="chunks used as sources")
    parser.add_argument("--seed", type=int, default=42, help="seed of the chunk choice")
    parser.add_argument("--force", action="store_true", help="overwrite the test set")
    args = parser.parse_args()

    # Generating costs many Gemini calls: never overwrite the test set by accident
    if settings.testset_path.exists() and not args.force:
        print(f"{settings.testset_path} already exists. Use --force to regenerate it.")
        return

    with settings.chunks_path.open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    selected = select_chunks(chunks, args.chunks, args.seed)
    print(f"Generating {args.size} questions from {len(selected)} chunks...")

    samples = generate_testset(selected, args.size, GeminiLLM(), Embedder())
    save_jsonl(samples, settings.testset_path)
    print(f"Saved {len(samples)} questions to {settings.testset_path}")


if __name__ == "__main__":
    main()
