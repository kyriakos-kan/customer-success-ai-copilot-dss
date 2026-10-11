"""RAG over per-account context (emails, call notes, contract terms, tickets).

* Retrieval is always filtered by account_id — the writer must never see
  another customer's history.
* Embeddings come from a HashingVectorizer: deterministic, no model
  download, tiny image. Swap `_embed` for a sentence-transformer or a
  hosted embedding API when you move past the MVP — nothing else changes.
* If CHROMA_HOST is unset (local dev / tests) an in-memory store is used.
"""
import logging

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from tenacity import retry, stop_after_attempt, wait_exponential

from copilot.config import get_settings

log = logging.getLogger(__name__)
COLLECTION = "account_context"
_vec = HashingVectorizer(n_features=512, alternate_sign=False, norm="l2")


def _embed(texts: list[str]) -> list[list[float]]:
    return _vec.transform(texts).toarray().tolist()


class _InMemoryStore:
    def __init__(self):
        self.docs: dict[str, tuple[str, dict, np.ndarray]] = {}

    def upsert(self, ids, documents, metadatas):
        for i, d, m, e in zip(ids, documents, metadatas, _embed(documents)):
            self.docs[i] = (d, m, np.asarray(e))

    def query(self, account_id: str, text: str, k: int):
        q = np.asarray(_embed([text])[0])
        hits = [(float(e @ q), d, m) for d, m, e in self.docs.values()
                if m["account_id"] == account_id]
        hits.sort(key=lambda h: -h[0])
        return [{"text": d, "metadata": m, "score": round(s, 3)} for s, d, m in hits[:k]]


class _ChromaStore:
    def __init__(self, host: str, port: int):
        self.col = self._connect(host, port)

    @staticmethod
    @retry(stop=stop_after_attempt(10), wait=wait_exponential(min=1, max=15), reraise=True)
    def _connect(host, port):
        import chromadb
        client = chromadb.HttpClient(host=host, port=port)
        client.heartbeat()                                     # fails until Chroma is ready
        return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    def upsert(self, ids, documents, metadatas):
        self.col.upsert(ids=ids, documents=documents, metadatas=metadatas,
                        embeddings=_embed(documents))

    def query(self, account_id: str, text: str, k: int):
        res = self.col.query(query_embeddings=_embed([text]), n_results=k,
                             where={"account_id": account_id})
        return [{"text": d, "metadata": m, "score": round(1 - dist, 3)}
                for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]


class ContextRetriever:
    def __init__(self):
        s = get_settings()
        self.store = _ChromaStore(s.chroma_host, s.chroma_port) if s.chroma_host else _InMemoryStore()
        log.info("RAG store: %s", type(self.store).__name__)

    def index(self, docs: list[dict]) -> None:
        """docs: [{id, account_id, doc_type, date, text}]"""
        if docs:
            self.store.upsert(
                ids=[d["id"] for d in docs], documents=[d["text"] for d in docs],
                metadatas=[{"doc_id": d["id"], "account_id": d["account_id"], "doc_type": d["doc_type"],
                            "date": d["date"]} for d in docs])

    def retrieve(self, account_id: str, query: str, k: int = 4,
                 min_score: float = 0.05) -> list[dict]:
        return [h for h in self.store.query(account_id, query, k) if h["score"] >= min_score]


def synthetic_context(gold) -> list[dict]:
    """Demo context documents. Replace with your CRM / email / ticket export.
    Roughly 15% of accounts get no documents, so the writer's
    'insufficient data' guardrail is exercised in the demo."""
    rng = np.random.default_rng(7)
    out, i = [], 0
    for row in gold.itertuples():
        if rng.random() < 0.15:
            continue
        items = []
        if row.quality_complaints_12m >= 2:
            items.append(("ticket", f"Complaint about shrinkage and colour inconsistency on the last "
                                    f"{row.quality_complaints_12m} cotton-blend batches; customer asked for lab reports."))
        if row.late_payments_12m >= 2:
            items.append(("call_note", "Finance contact mentioned cash-flow pressure and asked about "
                                       "extending payment terms from 30 to 60 days."))
        if row.avg_lead_time_days > 30:
            items.append(("email", f"Buyer raised concern that {row.avg_lead_time_days:.0f}-day lead times "
                                   "are hurting their seasonal collection planning."))
        if row.order_volume_change_pct < -15:
            items.append(("call_note", "Purchasing manager said they are trialling a second supplier "
                                       "for basic jersey fabrics to compare price."))
        items.append(("contract", f"Framework agreement renews in {rng.integers(1, 12)} months; "
                                  "volume rebate tier applies above current annual volume."))
        for dtype, text in items:
            out.append({"id": f"{row.account_id}-{i}", "account_id": row.account_id,
                        "doc_type": dtype, "date": "2026-09-01", "text": text})
            i += 1
    return out
