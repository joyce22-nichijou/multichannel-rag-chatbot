from docling_core.types.doc import DoclingDocument

doc = DoclingDocument.load_from_json("data/processed/Core Rules.json")
PAGE = 55
for item, _ in doc.iterate_items(page_no=PAGE):
    b = item.prov[0].bbox
    text = getattr(item, "text", "")[:40]
    print(f"{item.label.value:<15} l={b.l:5.0f} t={b.t:5.0f} | {text}")