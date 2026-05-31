# Payer Policy PA Parameter Extraction Pipeline

End-to-end RAG pipeline that extracts **12 Prior Authorization (PA) parameters** per brand from PsO (Plaque Psoriasis) payer policy PDFs and outputs a structured CSV with an access score per brand.

---

## Python Version

Requires **Python 3.13**. Python 3.14+ is not supported (tokenizers build fails).

```bash
python --version  # should be 3.13.x
```

---

## Installation

Clone the repo and create a virtual environment:

```bash
git clone https://github.com/charlesaurav13/Payer_Rag_ADS.git
cd Payer_Rag_ADS
python3.13 -m venv RAG_System
source RAG_System/bin/activate
pip install -r requirements.txt
```

Set your Groq API key:

```bash
export GROQ_API_KEY=gsk_...
```

On Kaggle, add `GROQ_API_KEY` via **Add-ons > Secrets** and enable **Attach to session**.

---

## Architecture

```
PDF
 │
 ├─ PyMuPDF         Clean headers, footers, links, credentials in memory
 │
 ├─ Docling         Convert cleaned PDF → Markdown with table detection (CPU)
 │
 ├─ Chunker         Recursive character split — 700 chars / 100 overlap
 │
 ├─ ChromaDB        Store chunks as 384-dim dense vectors (ephemeral per PDF)
 │
 ├─ Hybrid Search   BM25 sparse + BGE-384 dense → RRF fusion (k=60)
 │                  → bge-reranker-v2-m3 cross-encoder reranking
 │
 ├─ Brand Detection [8B LLM]
 │   ├─ Search for drug-list sections → pass top chunks to 8B
 │   └─ Per-brand anchor chunk IDs collected for downstream retrieval
 │
 ├─ Param Extraction [70B LLM]
 │   ├─ Tier 1: anchor chunks (guaranteed relevant)
 │   ├─ Tier 2: brand-specific hybrid search (fills to max 4 chunks)
 │   └─ One 70B call per brand → 12 PA parameters
 │
 └─ CSV Output      filename, brand, 12 parameters, access_score
```

---

## LLMs

| Model | Purpose | Why |
|-------|---------|-----|
| `llama-3.1-8b-instant` (Groq) | Brand detection | Simple pattern matching — finds drug names from drug-list chunks. Fast and token-efficient. |
| `llama-3.3-70b-versatile` (Groq) | Parameter extraction | Complex reasoning across contradictory policy text — needs larger model for accuracy. |

Both models are served via **Groq's free tier**. A built-in `RateLimiter` tracks requests and tokens in a 60-second sliding window, staying just below the free-tier limits (12K TPM for 70B is the binding constraint).

---

## Embeddings

| Component | Model | Dimension | Purpose |
|-----------|-------|-----------|---------|
| Dense encoder | `BAAI/bge-small-en-v1.5` | 384 | Chunk embeddings stored in ChromaDB for cosine similarity search |
| Cross-encoder reranker | `BAAI/bge-reranker-v2-m3` | — | Re-scores BM25 + dense fusion candidates for final top-k selection |

384-dim was chosen over 768-dim (bge-base) for faster encoding and lower memory with minimal accuracy loss on this domain.

---

## Access Score

The access score (1–100) is a weighted sum across all 12 parameters. **Higher score = easier patient access** to the drug.

| Parameter | Weight | Scoring Logic |
|-----------|--------|---------------|
| Number of Steps through Brands | 20 | 0 steps → 1.0 · each step subtracts 0.3 |
| Initial Authorization Duration | 15 | ≥12 months → 1.0 · 6–11 → 0.7 · 3–5 → 0.4 · <3 → 0.2 |
| TB Test required | 15 | Not required → 1.0 · Required → 0.3 |
| Age | 10 | ≤6 yrs → 1.0 · ≤12 → 0.9 · ≤18 → 0.7 · >18 → 0.4 |
| Number of Steps through Generic | 10 | Same as steps through brands |
| Step through-Phototherapy | 5 | Not required → 1.0 · Required → 0.3 |
| Step Therapy text present | 5 | No text / NA → 1.0 · Text present → 0.3 |
| Reauthorization Required | 5 | Not required → 1.0 · Required → 0.3 |
| Reauthorization Duration | 5 | Same as initial auth duration |
| Specialist Types | 4 | No restriction → 1.0 · Restriction present → 0.3 |
| Reauth Requirements text | 3 | No text / NA → 1.0 · Text present → 0.3 |
| Quantity Limits | 3 | No limits → 1.0 · Limits present → 0.3 |

**Formula:**

```
access_score = round( Σ weight_i × score_i )   where score_i ∈ [0.0, 1.0]
```

Maximum possible score = 100 (all parameters indicate no restrictions).
