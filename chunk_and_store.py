"""
Markdown → recursive chunks → ChromaDB + BM25 hybrid search.

Chunk schema
------------
  chunk_id  : "<pdf_stem>_<index:04d>"
  text      : str
  columns   : list[str]   table column headers; [] for prose
  metadata  : {
      table  : bool
      pdf    : str   source PDF filename
      header : str   nearest section header above this chunk
  }

Hybrid search: BM25 (sparse) + cosine (dense, 768-dim BGE) fused with RRF.
"""

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MARKDOWN_DIR  = Path("../markdown_output")
CHROMA_DIR    = Path("chroma_store")
EMBED_MODEL   = "BAAI/bge-base-en-v1.5"      # 768-dim
RERANK_MODEL  = "BAAI/bge-reranker-v2-m3"   # cross-encoder reranker
COLLECTION    = "payer_policy"
CHUNK_SIZE    = 900
CHUNK_OVERLAP = 150
RRF_K         = 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id: str
    text: str
    columns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Recursive text splitter
# ---------------------------------------------------------------------------
_SEPARATORS = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""]


def _merge_splits(splits: List[str], sep: str, size: int, overlap: int) -> List[str]:
    """
    Merge atomic splits into chunks <= size chars.
    Sliding-window: flush when the next split would exceed size,
    pop from the front until retained context <= overlap.
    """
    chunks: List[str] = []
    window: List[str] = []
    window_len = 0
    sep_len = len(sep)

    def flush() -> None:
        if not window:
            return
        chunk = sep.join(window)
        if len(chunk) <= size:
            chunks.append(chunk)
        else:
            step = max(1, size - overlap)
            for i in range(0, len(chunk), step):
                piece = chunk[i: i + size]
                if piece.strip():
                    chunks.append(piece)

    for s in splits:
        s_len = len(s)
        add_len = s_len + (sep_len if window else 0)

        if window_len + add_len > size:
            flush()
            while window and window_len > overlap:
                removed = window.pop(0)
                window_len -= len(removed) + (sep_len if window else 0)

        window.append(s)
        window_len += s_len + (sep_len if len(window) > 1 else 0)

    flush()
    return chunks


def _recursive_split(text: str, size: int = CHUNK_SIZE,
                     overlap: int = CHUNK_OVERLAP,
                     seps: List[str] = _SEPARATORS) -> List[str]:
    """
    Recursively split text using separators from coarsest to finest,
    then merge into chunks of at most `size` chars with `overlap`.
    """
    if not text.strip():
        return []
    if len(text) <= size:
        return [text.strip()]

    sep = None
    remaining_seps: List[str] = []
    for i, s in enumerate(seps):
        if s == "" or s in text:
            sep = s
            remaining_seps = seps[i + 1:]
            break

    if sep is None:
        return [text.strip()]

    if sep == "":
        step = max(1, size - overlap)
        return [text[i: i + size].strip()
                for i in range(0, len(text), step)
                if text[i: i + size].strip()]

    flat: List[str] = []
    for p in text.split(sep):
        p = p.strip()
        if not p:
            continue
        if len(p) <= size:
            flat.append(p)
        else:
            flat.extend(_recursive_split(p, size, overlap, remaining_seps))

    join_sep = sep.strip() or " "
    return [c.strip() for c in _merge_splits(flat, join_sep, size, overlap) if c.strip()]


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------
_TABLE_RE = re.compile(
    r"(\|[^\n]+\|\n\|[-| :]+\|\n(?:\|[^\n]+\|\n)*)",
    re.MULTILINE,
)
_HEADER_RE = re.compile(r"^#{1,4}\s+(.+)", re.MULTILINE)


def _extract_columns(table_text: str) -> List[str]:
    first_line = table_text.strip().splitlines()[0]
    return [c.strip() for c in first_line.split("|") if c.strip()]


def _split_table(table_text: str, columns: List[str],
                 size: int, meta: Dict) -> List[Chunk]:
    """Split an oversized table into row-batches, each prefixed with the header row."""
    lines = table_text.strip().splitlines()
    if len(lines) < 3:
        return []

    header_block = lines[0] + "\n" + lines[1] + "\n"
    chunks: List[Chunk] = []
    buf = header_block

    for row in lines[2:]:
        candidate = buf + row + "\n"
        if len(candidate) <= size:
            buf = candidate
        else:
            if buf.strip() != header_block.strip():
                chunks.append(Chunk(text=buf.strip(), columns=columns,
                                    metadata=meta, chunk_id=""))
            buf = header_block + row + "\n"

    if buf.strip() and buf.strip() != header_block.strip():
        chunks.append(Chunk(text=buf.strip(), columns=columns,
                            metadata=meta, chunk_id=""))
    return chunks


# ---------------------------------------------------------------------------
# Markdown chunker
# ---------------------------------------------------------------------------
def chunk_markdown(md_text: str, pdf_name: str) -> List[Chunk]:
    """Parse one markdown document into Chunk objects."""
    chunks: List[Chunk] = []
    current_header = "Introduction"
    idx = 0

    def _next_id() -> str:
        nonlocal idx
        cid = f"{Path(pdf_name).stem}_{idx:04d}"
        idx += 1
        return cid

    segments: List[Dict] = []
    last_end = 0
    for m in _TABLE_RE.finditer(md_text):
        if m.start() > last_end:
            segments.append({"type": "prose", "text": md_text[last_end:m.start()]})
        segments.append({"type": "table", "text": m.group(0)})
        last_end = m.end()
    if last_end < len(md_text):
        segments.append({"type": "prose", "text": md_text[last_end:]})

    for seg in segments:
        # ---- TABLE --------------------------------------------------------
        if seg["type"] == "table":
            columns = _extract_columns(seg["text"])
            meta = {"table": True, "pdf": pdf_name, "header": current_header}
            if len(seg["text"]) <= CHUNK_SIZE:
                chunks.append(Chunk(
                    chunk_id=_next_id(),
                    text=seg["text"].strip(),
                    columns=columns,
                    metadata=meta,
                ))
            else:
                for c in _split_table(seg["text"], columns, CHUNK_SIZE, meta):
                    c.chunk_id = _next_id()
                    chunks.append(c)

        # ---- PROSE --------------------------------------------------------
        else:
            prose = seg["text"]
            for hdr in _HEADER_RE.finditer(prose):
                current_header = hdr.group(1).strip()

            for split in _recursive_split(prose):
                for hdr in _HEADER_RE.finditer(split):
                    current_header = hdr.group(1).strip()
                chunks.append(Chunk(
                    chunk_id=_next_id(),
                    text=split,
                    columns=[],
                    metadata={"table": False, "pdf": pdf_name, "header": current_header},
                ))

    return chunks


# ---------------------------------------------------------------------------
# ChromaDB store + hybrid search
# ---------------------------------------------------------------------------
class PolicyStore:
    def __init__(self, chroma_dir: Path = CHROMA_DIR, embed_model: str = EMBED_MODEL,
                 rerank_model: str = RERANK_MODEL):
        log.info("Loading embedding model: %s", embed_model)
        self.encoder = SentenceTransformer(embed_model)
        log.info("Embedding dim: %d", self.encoder.get_sentence_embedding_dimension())

        log.info("Loading reranker: %s", rerank_model)
        self.reranker = CrossEncoder(rerank_model)

        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.col = self.client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        log.info("Collection '%s' — %d docs stored", COLLECTION, self.col.count())

        self._bm25: Optional[BM25Okapi] = None
        self._bm25_ids: List[str] = []
        self._bm25_texts: List[str] = []

    def add_chunks(self, chunks: List[Chunk], batch_size: int = 64) -> int:
        existing = set(self.col.get(include=[])["ids"])
        new = [c for c in chunks if c.chunk_id not in existing]
        if not new:
            return 0

        for i in range(0, len(new), batch_size):
            batch = new[i: i + batch_size]
            texts = [c.text for c in batch]
            embeddings = self.encoder.encode(
                texts, batch_size=32, show_progress_bar=False,
                normalize_embeddings=True,
            ).tolist()
            self.col.add(
                ids=[c.chunk_id for c in batch],
                documents=texts,
                embeddings=embeddings,
                metadatas=[
                    {
                        **c.metadata,
                        "columns": "|".join(c.columns),
                        "table": str(c.metadata.get("table", False)),
                    }
                    for c in batch
                ],
            )

        self._bm25 = None
        log.info("Stored %d new chunks (%d already present)", len(new), len(existing))
        return len(new)

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        result = self.col.get(include=["documents"])
        self._bm25_ids = result["ids"]
        self._bm25_texts = result["documents"]
        self._bm25 = BM25Okapi([t.lower().split() for t in self._bm25_texts])
        log.info("BM25 index built over %d docs", len(self._bm25_ids))

    def hybrid_search(self, query: str, top_k: int = 10,
                      rerank_candidates: int = 30) -> List[Dict[str, Any]]:
        """
        Three-stage retrieval:
          1. BM25 + dense cosine → RRF → top `rerank_candidates`
          2. bge-reranker-v2-m3 cross-encoder → re-scores each candidate
          3. Return top `top_k` by reranker score
        """
        self._ensure_bm25()
        n_candidates = min(rerank_candidates, max(self.col.count(), 1))

        # --- Stage 1a: BM25 sparse ---
        bm25_scores = self._bm25.get_scores(query.lower().split())
        sparse_ranks = np.argsort(bm25_scores)[::-1][:n_candidates].tolist()

        # --- Stage 1b: dense cosine ---
        q_vec = self.encoder.encode([query], normalize_embeddings=True).tolist()
        dense = self.col.query(
            query_embeddings=q_vec,
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"],
        )
        dense_ids: List[str] = dense["ids"][0]

        # --- Stage 1c: RRF fusion ---
        rrf: Dict[str, float] = {}
        for rank, arr_idx in enumerate(sparse_ranks):
            cid = self._bm25_ids[arr_idx]
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, cid in enumerate(dense_ids):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

        candidate_ids = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:n_candidates]

        # Fetch full text for all candidates
        fetched = self.col.get(ids=candidate_ids, include=["documents", "metadatas"])
        id_map = {
            cid: (doc, meta)
            for cid, doc, meta in zip(
                fetched["ids"], fetched["documents"], fetched["metadatas"]
            )
        }

        # --- Stage 2: rerank ---
        pairs = [(query, id_map[cid][0]) for cid in candidate_ids if cid in id_map]
        rerank_scores = self.reranker.predict(pairs)   # shape: (n_candidates,)

        ranked = sorted(
            zip(candidate_ids, rerank_scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        # --- Stage 3: build results ---
        return [
            {
                "chunk_id": cid,
                "text": id_map[cid][0],
                "columns": [c for c in id_map[cid][1].get("columns", "").split("|") if c],
                "metadata": {
                    "table":  id_map[cid][1].get("table") == "True",
                    "pdf":    id_map[cid][1].get("pdf", ""),
                    "header": id_map[cid][1].get("header", ""),
                },
                "rrf_score":    round(rrf.get(cid, 0.0), 6),
                "rerank_score": round(float(score), 4),
            }
            for cid, score in ranked if cid in id_map
        ]


# ---------------------------------------------------------------------------
# Entry point — processes ONE markdown file
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    md_files = sorted(MARKDOWN_DIR.glob("*.md"))
    if not md_files:
        log.error("No .md files found in %s", MARKDOWN_DIR)
        raise SystemExit(1)

    # Pick the first file; change index or pass a name to use a different one
    md_path = md_files[0]
    log.info("Processing: %s", md_path.name)

    text   = md_path.read_text(encoding="utf-8")
    chunks = chunk_markdown(text, md_path.stem + ".pdf")
    log.info("Chunks produced: %d", len(chunks))
    log.info("  tables : %d", sum(1 for c in chunks if c.metadata["table"]))
    log.info("  prose  : %d", sum(1 for c in chunks if not c.metadata["table"]))
    log.info("  max len: %d", max(len(c.text) for c in chunks))

    store = PolicyStore()
    store.add_chunks(chunks)