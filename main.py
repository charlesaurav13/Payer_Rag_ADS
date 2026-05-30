"""
Full pipeline — 1 PDF end to end.

  PDF
   │
   ▼  STEP 1 — pdf_to_markdown.py
  Markdown
   │
   ▼  STEP 2 — chunk_and_store.py
  Chunks → ChromaDB (ephemeral, in memory for this run)
   │
   ▼  STEP 3 — brand_detection.py + llm_client.py
  Brands relevant to PsO
   │
   ▼  STEP 4 — param_extraction.py + llm_client.py
  12 parameters per brand
   │
   ▼  STEP 5
  extraction_output.csv

Run:
    cd /path/to/Payer_policy
    uv run python Final_Code/main.py
"""

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Make Final_Code importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

from pdf_to_markdown import build_converter, convert_pdf
from chunk_and_store import chunk_markdown, PolicyStore
from brand_detection import detect_brands
from param_extraction import extract_params, write_csv
from llm_client import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths  (all relative to the project root, one level above Final_Code/)
# ---------------------------------------------------------------------------
_HERE        = Path(__file__).parent
PDF_DIR      = _HERE / "pdfs" / "Sample_PsO_ADS_Track"
MARKDOWN_DIR = _HERE / "markdown_output"
OUTPUT_CSV   = _HERE / "extraction_output.csv"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(pdf_path: Path) -> None:
    log.info("=" * 65)
    log.info("PDF  : %s", pdf_path.name)
    log.info("=" * 65)

    # ------------------------------------------------------------------
    # STEP 1 — PDF → Markdown
    # ------------------------------------------------------------------
    log.info("STEP 1 — PDF → Markdown")
    converter = build_converter()
    md_path   = convert_pdf(pdf_path, MARKDOWN_DIR, converter)

    if md_path is None:
        log.error("Markdown conversion failed — aborting")
        return
    log.info("  Markdown: %s", md_path)

    # ------------------------------------------------------------------
    # STEP 2 — Chunk & Store (ephemeral ChromaDB for this run)
    # ------------------------------------------------------------------
    log.info("STEP 2 — Chunk & Store")
    pdf_name = pdf_path.stem + ".pdf"
    md_text  = md_path.read_text(encoding="utf-8")
    chunks   = chunk_markdown(md_text, pdf_name)
    log.info("  Chunks produced: %d", len(chunks))

    tmp_dir = tempfile.mkdtemp(prefix="chroma_main_")
    try:
        store = PolicyStore(chroma_dir=Path(tmp_dir))
        store.add_chunks(chunks)

        # ------------------------------------------------------------------
        # STEP 3 — Brand Detection
        # ------------------------------------------------------------------
        log.info("STEP 3 — Brand Detection")
        provider   = os.getenv("LLM_PROVIDER", "openrouter")
        llm        = LLMClient(provider=provider)
        brand_data = detect_brands(md_path, llm)

        has_pso = brand_data.get("policy_has_pso", "No")
        brands  = brand_data.get("brands_relevant_to_pso", [])
        log.info("  policy_has_pso : %s", has_pso)
        log.info("  brands found   : %s", [b["brand"] for b in brands])

        if has_pso != "Yes" or not brands:
            log.info("  No PsO brands — nothing to extract")
            return

        # ------------------------------------------------------------------
        # STEP 4 — 12-Parameter Extraction
        # ------------------------------------------------------------------
        log.info("STEP 4 — Parameter Extraction")
        rows = extract_params(md_path, brand_data, store, llm)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # STEP 5 — Write CSV
    # ------------------------------------------------------------------
    log.info("STEP 5 — Write CSV")
    if rows:
        write_csv(rows, OUTPUT_CSV, append=True)
        log.info("  Wrote %d rows → %s", len(rows), OUTPUT_CSV)
        for r in rows:
            log.info("    %-30s  steps_brand=%-4s  reauth=%-4s  TB=%s",
                     r["brand"], r["Number of Steps through Brands"],
                     r["Reauthorization Required"], r["TB Test required"])
    else:
        log.info("  No rows extracted")

    log.info("LLM cache: %d entries", llm.cache_size)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        log.error("No PDFs found in %s", PDF_DIR)
        raise SystemExit(1)

    # Run on the first PDF — change index to try another
    for pdf in pdfs:
        run_pipeline(pdf)
