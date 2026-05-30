"""
PDF → Markdown conversion.

Steps (all in memory — no cleaned_pdfs/ folder written to disk):
  1. PyMuPDF  — strip headers/footers, remove hyperlinks, redact credentials
  2. Docling  — convert to Markdown with table detection

Exposes:
  convert_pdf(pdf_path, md_dir, converter) -> Path | None
  build_converter() -> DocumentConverter
"""

import re
import logging
import tempfile
from pathlib import Path
from typing import Optional

import fitz  # pymupdf
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

log = logging.getLogger(__name__)

HEADER_RATIO = 0.07
FOOTER_RATIO = 0.07

_CRED_PATTERNS = [
    re.compile(r"(username|user[\s_-]*name|login|user[\s_-]*id)\s*[:=]\s*\S+", re.I),
    re.compile(r"(password|passwd|pwd)\s*[:=]\s*\S+", re.I),
    re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
]


# ---------------------------------------------------------------------------
# Cleaning helpers (operate on fitz.Document in place)
# ---------------------------------------------------------------------------
def _redact_zone(page: fitz.Page, zone: fitz.Rect) -> None:
    for b in page.get_text("blocks", clip=zone):
        page.add_redact_annot(fitz.Rect(b[:4]), fill=(1, 1, 1))


def _redact_credentials(page: fitz.Page) -> None:
    for block in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE).get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if any(p.search(span.get("text", "")) for p in _CRED_PATTERNS):
                    page.add_redact_annot(fitz.Rect(span["bbox"]), fill=(1, 1, 1))


def _clean_doc(doc: fitz.Document) -> None:
    """Clean headers/footers/links/credentials in place."""
    for page in doc:
        h, w = page.rect.height, page.rect.width
        _redact_zone(page, fitz.Rect(0, 0, w, h * HEADER_RATIO))
        _redact_zone(page, fitz.Rect(0, h * (1 - FOOTER_RATIO), w, h))
        for link in page.get_links():
            page.delete_link(link)
        _redact_credentials(page)
        page.apply_redactions()


# ---------------------------------------------------------------------------
# Docling converter
# ---------------------------------------------------------------------------
def build_converter() -> DocumentConverter:
    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = False
    opts.table_structure_options.do_cell_matching = True

    # Force CPU — MPS (Apple Silicon) doesn't support float64 in RT-DETRv2
    try:
        from docling.datamodel.pipeline_options import AcceleratorOptions
        from docling.datamodel.accelerator_options import AcceleratorDevice
        opts.accelerator_options = AcceleratorOptions(
            num_threads=4, device=AcceleratorDevice.CPU
        )
    except ImportError:
        pass

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


# ---------------------------------------------------------------------------
# Main function (importable by main.py)
# ---------------------------------------------------------------------------
def convert_pdf(
    pdf_path: Path,
    md_dir: Path,
    converter: DocumentConverter,
) -> Optional[Path]:
    """
    Clean *pdf_path* in memory and convert to Markdown via Docling.
    Writes one .md file to *md_dir*.  No intermediate files saved to disk.

    Returns the Path to the written markdown file, or None on failure.
    """
    md_out = md_dir / (pdf_path.stem + ".md")

    # Skip if already converted
    if md_out.exists():
        log.info("[skip]    %s — markdown already exists", pdf_path.name)
        return md_out

    try:
        # --- clean in memory ---
        doc = fitz.open(str(pdf_path))
        _clean_doc(doc)

        # Write cleaned bytes to a temp file (Docling needs a file path)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            doc.save(str(tmp_path), garbage=4, deflate=True)
        doc.close()

        # --- convert via Docling ---
        result   = converter.convert(str(tmp_path))
        tmp_path.unlink(missing_ok=True)          # delete temp file immediately

        document = result.document
        markdown = document.export_to_markdown()

        # Count tables
        n_tables = 0
        try:
            from docling_core.types.doc import DocItemLabel
            n_tables = sum(
                1 for item, _ in document.iterate_items()
                if getattr(item, "label", None) == DocItemLabel.TABLE
            )
        except Exception:
            n_tables = markdown.count("\n|")

        md_dir.mkdir(parents=True, exist_ok=True)
        md_out.write_text(markdown, encoding="utf-8")
        log.info("[done]    %s  | tables: %d | chars: %d",
                 pdf_path.name, n_tables, len(markdown))
        return md_out

    except Exception as exc:
        log.error("[fail]    %s — %s", pdf_path.name, exc)
        tmp_path = locals().get("tmp_path")
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
        return None
