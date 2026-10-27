from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    OcrAutoOptions,
    OcrMode
)
from docling.document_converter import DocumentConverter, PdfFormatOption


@dataclass(frozen=True)
class ParsedArtifacts:
    """Files generated from one parsed PDF."""

    source_path: Path
    json_path: Path
    markdown_path: Path
    status: str


class DoclingPdfParser:
    """Parse PDF documents with Docling and export inspectable artifacts."""

    def __init__(self,
                 *, 
                 use_ocr: bool = False,
                 ocr_languages: tuple[str, ...] = ("iso:en",),
                 ocr_scale: float = 3.0,
                 images_scale: float = 2.0,
                 keep_picture_images: bool = False
    ) -> None:
        pipeline_options = PdfPipelineOptions()

        # Enable this only when the PDF needs OCR.
        pipeline_options.do_ocr = use_ocr

        if use_ocr:
            pipeline_options.ocr_options = OcrAutoOptions(
                ocr_mode=OcrMode.DEFAULT,
                lang=list(ocr_languages),
                scale=ocr_scale,
            )

        # Useful because rulebooks may contain tables and structured rule boxes.
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=True
        )

        # Scale images to improve OCR quality.
        pipeline_options.images_scale = images_scale

        # Do not generate large page/picture images for the first version.
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = keep_picture_images
        pipeline_options.generate_table_images = False

        self._converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            },
        )

    def parse(
        self,
        source_path: Path,
        output_dir: Path,
        *,
        page_range: tuple[int, int] | None = None,
    ) -> ParsedArtifacts:
        """Parse one PDF and export JSON and Markdown."""

        source_path = source_path.resolve()
        output_dir = output_dir.resolve()
        
        output_dir.mkdir(parents=True, exist_ok=True)

        convert_kwargs = {}
        if page_range is not None:
            convert_kwargs["page_range"] = page_range

        result = self._converter.convert(
            source=source_path,
            raises_on_error=True,
            **convert_kwargs,
        )

        document = result.document
        output_stem = source_path.stem

        json_path = output_dir / f"{output_stem}.json"
        markdown_path = output_dir / f"{output_stem}.md"

        json_path.write_text(
            json.dumps(
                document.export_to_dict(),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        markdown_path.write_text(
            document.export_to_markdown(traverse_pictures=True),
            encoding="utf-8",
        )


        return ParsedArtifacts(
            source_path=source_path,
            json_path=json_path,
            markdown_path=markdown_path,
            status=str(result.status),
        )