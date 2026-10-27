from pathlib import Path

from raglab1.parser.pdf_parser import DoclingPdfParser


def main() -> None:
    source_path = Path("data") / "raw" / "Core Rules.pdf"
    output_dir = Path("data") / "processed"

    parser = DoclingPdfParser(
        use_ocr=False,
        ocr_languages=("iso:en",),
        ocr_scale=3.0,
        images_scale=2.0,
        keep_picture_images=False
    )

    artifacts = parser.parse(
        source_path=source_path,
        output_dir=output_dir,
    )

    print(f"Status:   {artifacts.status}")
    print(f"JSON:     {artifacts.json_path}")
    print(f"Markdown: {artifacts.markdown_path}")


if __name__ == "__main__":
    main()