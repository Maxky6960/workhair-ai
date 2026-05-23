# rag_engine.py
import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class RAGEngine:
    def __init__(self, kb_path: str, cache_dir: str = ".rag_cache"):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.kb_path = kb_path
        self.cache_dir = cache_dir
        self.chunks = self._load_and_chunk(kb_path)
        self.index, self.embeddings = self._build_index()
        print(f"✅ RAG Engine loaded: {len(self.chunks)} chunks")

    def _load_and_chunk(self, path: str) -> list[str]:
        """Load (ขั้นตอน 1) และ Chunk (ขั้นตอน 2)"""
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read().replace("\r\n", "\n")
        except FileNotFoundError:
            print(f"⚠️ Warning: Knowledge base file not found: {path}")
            return []

        # ทำความสะอาดบรรทัดว่าง และรวมเป็นรายการบรรทัด
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        # สร้าง chunk ขนาดจำกัด (characters) พร้อม overlap เพื่อให้การค้นหาละเอียดขึ้น
        max_chars = 450
        overlap_chars = 120
        chunks: list[str] = []
        cur = ""

        def flush_current() -> None:
            nonlocal cur
            if cur.strip():
                chunks.append(cur.strip())
            cur = ""

        def split_long_line(line: str) -> list[str]:
            if len(line) <= max_chars:
                return [line]
            parts = []
            start = 0
            while start < len(line):
                parts.append(line[start : start + max_chars])
                start += max_chars
            return parts

        for line in lines:
            for segment in split_long_line(line):
                if not cur:
                    cur = segment
                    continue

                candidate = cur + " " + segment
                if len(candidate) > max_chars:
                    flush_current()
                    if overlap_chars > 0 and cur:
                        start = max(0, len(cur) - overlap_chars)
                        cur = cur[start:].strip() + " " + segment
                    else:
                        cur = segment
                else:
                    cur = candidate

        flush_current()
        return chunks

    def _cache_paths(self) -> tuple[str, str]:
        os.makedirs(self.cache_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.kb_path))[0]
        return (
            os.path.join(self.cache_dir, f"{base}.faiss"),
            os.path.join(self.cache_dir, f"{base}.json"),
        )

    def _load_cache(self) -> tuple[faiss.Index | None, np.ndarray | None, list[str] | None]:
        index_path, meta_path = self._cache_paths()
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            return None, None, None

        try:
            index = faiss.read_index(index_path)
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            cached_chunks = meta.get("chunks", [])
            cached_embeddings = np.array(meta.get("embeddings", []), dtype="float32")
            if len(cached_chunks) == 0 or len(cached_embeddings) == 0:
                return None, None, None
            if len(cached_chunks) != len(cached_embeddings):
                return None, None, None
            return index, cached_embeddings, cached_chunks
        except Exception:
            return None, None, None

    def _save_cache(self, index: faiss.Index, embeddings: np.ndarray, chunks: list[str]) -> None:
        index_path, meta_path = self._cache_paths()
        faiss.write_index(index, index_path)
        meta = {
            "chunks": chunks,
            "embeddings": embeddings.tolist(),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _build_index(self):
        """Embed (ขั้นตอน 3)"""
        if not self.chunks:
            empty = np.zeros((0, 384), dtype="float32")
            return faiss.IndexFlatL2(empty.shape[1]), empty

        cached_index, cached_embeddings, cached_chunks = self._load_cache()
        if cached_index is not None and cached_chunks == self.chunks:
            return cached_index, cached_embeddings

        embeddings = self.model.encode(self.chunks, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        self._save_cache(index, embeddings, self.chunks)
        return index, embeddings

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """Search (ขั้นตอน 4)"""
        if not self.chunks:
            return []

        k = min(top_k, len(self.chunks))
        q_emb = self.model.encode([query])
        _, indices = self.index.search(np.array(q_emb, dtype="float32"), k)
        
        # Filter out invalid indices and return results
        results = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
        return results
