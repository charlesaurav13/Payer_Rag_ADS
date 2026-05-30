"""
Brand detection from a single markdown file.

Flow:
  Read markdown file → truncate to fit LLM context → LLM → brands JSON
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict

from llm_client import LLMClient

log = logging.getLogger(__name__)

# Max characters sent to the LLM — keeps us well within the 8K token window
MAX_CHARS = 6000

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
_BRAND_PROMPT = """\
You are an expert at extracting structured prior authorization policy information from payer policy documents.

Your task is to identify all brands/products in this policy document that are relevant to plaque psoriasis (PsO).

Instructions:
1. Read the full policy text carefully.
2. Identify all products listed in the Applicable Drug List or equivalent drug list section.
3. Determine whether the policy contains coverage criteria for plaque psoriasis (PsO).
4. Return every product/brand that is relevant to PsO extraction.
5. Include preferred/non-preferred status if explicitly stated.
6. Do not infer brands that are not explicitly listed.
7. If the policy has multiple indications, only identify brands relevant to PsO extraction.

Return strict JSON only in this format:
{{
  "policy_has_pso": "Yes | No",
  "brands_relevant_to_pso": [
    {{
      "brand": "",
      "preferred_status": "Preferred | Non-preferred | Unspecified"
    }}
  ]
}}

Policy Text:
{policy_text}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _truncate(text: str, max_chars: int = MAX_CHARS) -> str:
    """
    If the markdown is longer than max_chars, keep the first 2/3 and
    last 1/3 — brand lists appear near the top and the drug section
    often appears at the end.
    """
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.67)
    tail = max_chars - head
    return text[:head] + "\n\n[...truncated...]\n\n" + text[-tail:]


def _parse_json(raw: str, pdf_name: str) -> Dict[str, Any]:
    try:
        start = raw.index("{")
        end   = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        log.error("Brand JSON parse failed for %s:\n%s", pdf_name, raw[:300])
        return {"policy_has_pso": "No", "brands_relevant_to_pso": []}


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def detect_brands(md_path: Path, llm: LLMClient) -> Dict[str, Any]:
    """
    Detect PsO-relevant brands from a single markdown file.

    Reads the markdown directly and passes it to the LLM — no chunk retrieval.

    Returns:
        {
          "policy_has_pso": "Yes" | "No",
          "brands_relevant_to_pso": [{"brand": ..., "preferred_status": ...}, ...]
        }
    """
    pdf_name    = md_path.stem + ".pdf"
    policy_text = _truncate(md_path.read_text(encoding="utf-8"))

    log.info("Brand detection — %s  (%d chars sent to LLM)", pdf_name, len(policy_text))

    messages = [{"role": "user", "content": _BRAND_PROMPT.format(policy_text=policy_text)}]
    raw      = llm.complete(messages, temperature=0.0, max_tokens=2048)
    result   = _parse_json(raw, pdf_name)

    log.info("  policy_has_pso=%s  brands=%d",
             result.get("policy_has_pso", "?"),
             len(result.get("brands_relevant_to_pso", [])))
    return result
