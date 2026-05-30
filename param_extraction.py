"""
12-parameter extraction pipeline.

Flow per brand:
  brand name + preferred_status
       │
       ▼  hybrid search (BM25 + dense + reranker)
  brand-specific chunks from ChromaDB
       │
       ▼  LLM (one call per brand)
  12 parameters + evidence
       │
       ▼  CSV row
"""

import csv
import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from chunk_and_store import chunk_markdown, PolicyStore
from brand_detection import detect_brands
from llm_client import LLMClient

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE        = Path(__file__).parent
MARKDOWN_DIR = _HERE / "../markdown_output"
OUTPUT_CSV   = _HERE / "extraction_output.csv"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
PARAMS = [
    "Age",
    "Step Therapy Requirements Documented in Policy",
    "Number of Steps through Brands",
    "Number of Steps through Generic",
    "Step through-Phototherapy",
    "TB Test required",
    "Initial Authorization Duration(in-months)",
    "Reauthorization Duration(in-months)",
    "Reauthorization Required",
    "Reauthorization Requirements Documented in Policy",
    "Specialist Types",
    "Quantity Limits",
]

CSV_COLUMNS = ["filename", "brand"] + PARAMS + ["access_score"]

# ---------------------------------------------------------------------------
# Prompt — one brand at a time
# ---------------------------------------------------------------------------
_PARAM_PROMPT = """\
You are an expert in extracting structured prior authorization policy data from payer policy documents.

Extract 12 PsO-specific parameters for the brand below using ONLY the provided policy chunks.

BRAND:
  Name             : {brand_name}
  Preferred status : {preferred_status}

INSTRUCTIONS:
- Extract for plaque psoriasis (PsO) only. Ignore other indications.
- If moderate-to-severe and severe PsO are distinguished, use moderate-to-severe criteria only.
- Universal criteria that apply to all brands must be combined with brand-specific criteria using AND logic.
- If OR statements exist, choose the least restrictive valid path.
- Count only what is explicitly stated. Do not infer.
- Use "NA" for any value not mentioned, unless rules below specify otherwise.
- Output strict JSON only — no explanation.

PARAMETERS:

1. Age: Age threshold for eligibility. Output "FDA labelled age" if only FDA labelling is mentioned. If two age groups listed, capture the youngest.

2. Step Therapy Requirements Documented in Policy: Full free-text of all step therapy language relevant to PsO for this brand (universal + brand-specific).

3. Number of Steps through Brands: Count of branded/biologic steps required before this brand is approved. Choose least restrictive OR path. Exclude phototherapy. "NA" if none.

4. Number of Steps through Generic: Count of non-biologic/generic/topical steps required. Exclude phototherapy. "NA" if none.

5. Step through-Phototherapy: "Yes" if phototherapy is a mandatory step. "No" if not required. "N/A" if no criteria at all.

6. TB Test required: "Y" if required. "N" if explicitly not required. "NA" if not mentioned.

7. Initial Authorization Duration(in-months): Numeric months. "Unspecified" if required but not stated numerically.

8. Reauthorization Duration(in-months): Numeric months. "Unspecified" if required but not stated numerically.

9. Reauthorization Required: "Yes" if reauth is documented. "No" if explicitly not required. "NA" otherwise.

10. Reauthorization Requirements Documented in Policy: Actual continuation/renewal criteria text for PsO.

11. Specialist Types: Specialist type(s) acceptable for prescribing/managing PsO treatment.

12. Quantity Limits: Only explicitly stated quantity limits. Do NOT extract dosage language. "NA" if not stated.

OUTPUT FORMAT — strict JSON only:
{{
  "brand": "{brand_name}",
  "preferred_status": "{preferred_status}",
  "Age": "",
  "Step Therapy Requirements Documented in Policy": "",
  "Number of Steps through Brands": "",
  "Number of Steps through Generic": "",
  "Step through-Phototherapy": "",
  "TB Test required": "",
  "Initial Authorization Duration(in-months)": "",
  "Reauthorization Duration(in-months)": "",
  "Reauthorization Required": "",
  "Reauthorization Requirements Documented in Policy": "",
  "Specialist Types": "",
  "Quantity Limits": "",
  "evidence": {{
    "Age": "",
    "Step Therapy Requirements Documented in Policy": "",
    "Number of Steps through Brands": "",
    "Number of Steps through Generic": "",
    "Step through-Phototherapy": "",
    "TB Test required": "",
    "Initial Authorization Duration(in-months)": "",
    "Reauthorization Duration(in-months)": "",
    "Reauthorization Required": "",
    "Reauthorization Requirements Documented in Policy": "",
    "Specialist Types": "",
    "Quantity Limits": ""
  }}
}}

RELEVANT POLICY CHUNKS:
{chunks}"""

# ---------------------------------------------------------------------------
# Per-brand chunk retrieval queries
# ---------------------------------------------------------------------------
_COMMON_QUERIES = [
    "step therapy prior authorization criteria plaque psoriasis PsO",
    "initial authorization duration months reauthorization renewal continuation criteria",
    "TB test tuberculosis quantity limit specialist prescriber dermatologist",
]
RETRIEVAL_TOP_K = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_brand_chunks(
    store: PolicyStore,
    pdf_name: str,
    brand_name: str,
) -> str:
    """
    Retrieve chunks relevant to a specific brand.
    Runs one brand-specific query + common parameter queries, deduplicates.
    """
    seen:  set       = set()
    texts: List[str] = []

    # Brand-specific query first (highest priority)
    brand_query = (
        f"{brand_name} prior authorization criteria plaque psoriasis "
        f"step therapy reauthorization approval"
    )
    for r in store.hybrid_search(brand_query, top_k=RETRIEVAL_TOP_K):
        if r["metadata"]["pdf"] == pdf_name and r["chunk_id"] not in seen:
            seen.add(r["chunk_id"])
            texts.append(r["text"])

    # Common parameter queries
    for query in _COMMON_QUERIES:
        for r in store.hybrid_search(query, top_k=RETRIEVAL_TOP_K):
            if r["metadata"]["pdf"] == pdf_name and r["chunk_id"] not in seen:
                seen.add(r["chunk_id"])
                texts.append(r["text"])

    return "\n\n---\n\n".join(texts)


def _parse_brand_json(raw: str, brand_name: str) -> Dict[str, Any]:
    try:
        start = raw.index("{")
        end   = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        log.error("JSON parse failed for brand '%s':\n%s", brand_name, raw[:200])
        return {}


def _score_age(val: str) -> float:
    v = val.strip().lower()
    if v in ("na", "", "fda labelled age", "fda labeled age"):
        return 1.0
    m = re.search(r"(\d+)", v)
    if not m:
        return 0.7
    age = int(m.group(1))
    if age <= 6:   return 1.0
    if age <= 12:  return 0.9
    if age <= 18:  return 0.7
    return 0.4


def _score_steps(val: str) -> float:
    v = val.strip().lower()
    if v in ("na", "", "0"):  return 1.0
    m = re.search(r"(\d+)", v)
    if not m:                  return 1.0
    n = int(m.group(1))
    return max(0.0, 1.0 - n * 0.3)


def _score_duration(val: str) -> float:
    v = val.strip().lower()
    if v in ("na", ""):        return 0.5
    if v == "unspecified":     return 0.5
    m = re.search(r"(\d+)", v)
    if not m:                  return 0.5
    months = int(m.group(1))
    if months >= 12:  return 1.0
    if months >= 6:   return 0.7
    if months >= 3:   return 0.4
    return 0.2


def _score_yesno(val: str, yes_score: float = 0.3, no_score: float = 1.0) -> float:
    v = val.strip().lower()
    if v in ("yes", "y"):  return yes_score
    if v in ("no", "n"):   return no_score
    return 0.7  # NA / unspecified


def _score_text_present(val: str) -> float:
    """No text / NA = easier access (1.0); text present = more restrictive (0.3)."""
    v = val.strip().lower()
    return 0.3 if v and v != "na" else 1.0


_WEIGHTS = {
    "Number of Steps through Brands":                20,
    "Initial Authorization Duration(in-months)":     15,
    "TB Test required":                              15,
    "Age":                                           10,
    "Number of Steps through Generic":               10,
    "Step through-Phototherapy":                      5,
    "Step Therapy Requirements Documented in Policy":  5,
    "Reauthorization Required":                       5,
    "Reauthorization Duration(in-months)":            5,
    "Specialist Types":                               4,
    "Reauthorization Requirements Documented in Policy": 3,
    "Quantity Limits":                                3,
}


def compute_access_score(row: Dict[str, str]) -> int:
    scorers = {
        "Age":                                            lambda v: _score_age(v),
        "Step Therapy Requirements Documented in Policy": lambda v: _score_text_present(v),
        "Number of Steps through Brands":                 lambda v: _score_steps(v),
        "Number of Steps through Generic":                lambda v: _score_steps(v),
        "Step through-Phototherapy":                      lambda v: _score_yesno(v),
        "TB Test required":                               lambda v: _score_yesno(v, yes_score=0.3, no_score=1.0),
        "Initial Authorization Duration(in-months)":      lambda v: _score_duration(v),
        "Reauthorization Duration(in-months)":            lambda v: _score_duration(v),
        "Reauthorization Required":                       lambda v: _score_yesno(v),
        "Reauthorization Requirements Documented in Policy": lambda v: _score_text_present(v),
        "Specialist Types":                               lambda v: _score_text_present(v),
        "Quantity Limits":                                lambda v: _score_text_present(v),
    }
    total = sum(
        _WEIGHTS[p] * scorers[p](row.get(p, "NA"))
        for p in _WEIGHTS
    )
    return round(total)


def _flatten_row(filename: str, brand_result: Dict) -> Dict[str, str]:
    row = {
        "filename": filename,
        "brand":    brand_result.get("brand", ""),
    }
    for p in PARAMS:
        row[p] = str(brand_result.get(p, "NA"))
    row["access_score"] = str(compute_access_score(row))
    return row


# ---------------------------------------------------------------------------
# Core extraction — one LLM call per brand (used by main.py)
# ---------------------------------------------------------------------------
def extract_params(
    md_path: Path,
    brand_data: Dict[str, Any],
    store: PolicyStore,
    llm: LLMClient,
) -> List[Dict[str, str]]:
    """
    For each brand: retrieve brand-specific chunks → LLM → 12 params.
    One focused LLM call per brand instead of one big call for all brands.
    """
    pdf_name = md_path.stem + ".pdf"
    brands   = brand_data.get("brands_relevant_to_pso", [])

    if not brands:
        log.info("  No brands — skipping")
        return []

    rows: List[Dict[str, str]] = []

    for brand in brands:
        brand_name = brand["brand"]
        preferred  = brand.get("preferred_status", "Unspecified")
        log.info("  Extracting — %s (%s)", brand_name, preferred)

        # Retrieve chunks specific to this brand
        chunks = _get_brand_chunks(store, pdf_name, brand_name)
        if not chunks.strip():
            log.warning("    No chunks retrieved for %s — skipping", brand_name)
            continue

        log.info("    Chunks: %d chars → LLM", len(chunks))

        messages = [{
            "role": "user",
            "content": _PARAM_PROMPT.format(
                brand_name=brand_name,
                preferred_status=preferred,
                chunks=chunks,
            ),
        }]

        raw    = llm.complete(messages, temperature=0.0, max_tokens=1024)
        result = _parse_brand_json(raw, brand_name)

        if result:
            rows.append(_flatten_row(pdf_name, result))

    log.info("  Total rows extracted: %d", len(rows))
    return rows


# ---------------------------------------------------------------------------
# Convenience wrapper — builds its own ephemeral store internally
# ---------------------------------------------------------------------------
def process_markdown(md_path: Path, llm: LLMClient) -> List[Dict[str, str]]:
    """Standalone entry: brand detection + store + extraction in one call."""
    pdf_name   = md_path.stem + ".pdf"
    brand_data = detect_brands(md_path, llm)

    if brand_data.get("policy_has_pso") != "Yes":
        log.info("  No PsO policy — skipping")
        return []

    tmp_dir = tempfile.mkdtemp(prefix="chroma_param_")
    try:
        store  = PolicyStore(chroma_dir=Path(tmp_dir))
        chunks = chunk_markdown(md_path.read_text(encoding="utf-8"), pdf_name)
        store.add_chunks(chunks)
        return extract_params(md_path, brand_data, store, llm)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------
def write_csv(rows: List[Dict], output_path: Path, append: bool = True) -> None:
    existing: set = set()
    if append and output_path.exists():
        with open(output_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing.add((r.get("filename", ""), r.get("brand", "")))

    new_rows = [r for r in rows if (r["filename"], r["brand"]) not in existing]
    skipped  = len(rows) - len(new_rows)
    if skipped:
        log.info("  Skipped %d duplicate row(s) already in CSV", skipped)
    if not new_rows:
        log.info("  No new rows to write")
        return

    mode         = "a" if (append and output_path.exists()) else "w"
    write_header = not (append and output_path.exists())
    with open(output_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)
    log.info("CSV updated — %d new row(s) → %s", len(new_rows), output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    provider = os.getenv("LLM_PROVIDER", "openrouter")
    llm      = LLMClient(provider=provider)

    md_files = sorted(MARKDOWN_DIR.glob("*.md"))
    if not md_files:
        log.error("No .md files found in %s", MARKDOWN_DIR)
        raise SystemExit(1)

    rows = process_markdown(md_files[0], llm)
    if rows:
        write_csv(rows, OUTPUT_CSV, append=True)
        for r in rows:
            print(f"  {r['brand']} | steps={r['Number of Steps through Brands']} "
                  f"| reauth={r['Reauthorization Required']} | TB={r['TB Test required']}")
