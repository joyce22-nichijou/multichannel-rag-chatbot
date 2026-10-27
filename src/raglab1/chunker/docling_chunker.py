import json
from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument
from transformers import AutoTokenizer

# 必须和之后用的 embedding 模型一致
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 256


def chunk_document(json_path: Path, output_path: Path) -> int:
    """读取 Docling 的 JSON，切分成 chunk，保存为 JSONL。返回 chunk 数量。"""
    doc = DoclingDocument.load_from_json(json_path)

    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
        max_tokens=MAX_TOKENS,
    )
    chunker = HybridChunker(tokenizer=tokenizer)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunker.chunk(dl_doc=doc)):
            pages = sorted({p.page_no for item in chunk.meta.doc_items for p in item.prov})
            record = {
                "chunk_id": f"{json_path.stem}_{i:04d}",
                "text": chunk.text,
                "text_for_embedding": chunker.contextualize(chunk=chunk),
                "headings": chunk.meta.headings or [],
                "pages": pages,
                "source": json_path.stem,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
