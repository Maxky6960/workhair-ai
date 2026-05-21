# rag_engine.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class RAGEngine:
    def __init__(self, kb_path: str):
        self.model = SentenceTransformer(EMBED_MODEL)
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

    def _build_index(self):
        """Embed (ขั้นตอน 3)"""
        if not self.chunks:
            empty = np.zeros((0, 384), dtype="float32")
            return faiss.IndexFlatL2(empty.shape[1]), empty

        embeddings = self.model.encode(self.chunks, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
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
