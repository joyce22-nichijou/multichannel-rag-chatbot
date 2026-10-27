import json
from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument
from transformers import AutoTokenizer

from raglab1.settings import settings


def clean_text(text: str) -> str:
    """Docling reads the rulebook's bullet glyph as the control character \\x07."""
    return text.replace("\x07", "-")


def chunk_document(json_path: Path, output_path: Path) -> int:
    """Read a Docling JSON, split it into chunks and save them as JSONL. Returns the chunk count."""
    doc = DoclingDocument.load_from_json(json_path)

    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(settings.embed_model_id),
        max_tokens=settings.chunk_max_tokens,
    )
    chunker = HybridChunker(tokenizer=tokenizer)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunker.chunk(dl_doc=doc):
            text = clean_text(chunk.text)
            # Skip fragments without any letter, e.g. the step numbers "2", "3", "6"
            if not any(ch.isalpha() for ch in text):
                continue
            pages = sorted(
                {p.page_no for item in chunk.meta.doc_items for p in item.prov}
            )
            record = {
                "chunk_id": f"{json_path.stem}_{count:04d}",
                "text": text,
                "text_for_embedding": clean_text(chunker.contextualize(chunk=chunk)),
                "headings": chunk.meta.headings or [],
                "pages": pages,
                "source": json_path.stem,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
