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

    def _load_and_chunk(self, path: str) -> list[str]:
        """Load (ขั้นตอน 1) และ Chunk (ขั้นตอน 2)"""
        with open(path, encoding="utf-8") as f:
            text = f.read().replace("\r\n", "\n")

        # ทำความสะอาดบรรทัดว่าง และรวมเป็นรายการบรรทัด
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        # สร้าง chunk ขนาดจำกัด (characters) พร้อม overlap เพื่อให้การค้นหาละเอียดขึ้น
        max_chars = 450
        overlap_chars = 120
        chunks: list[str] = []
        cur = ""

        for line in lines:
            if cur:
                candidate = cur + " " + line
            else:
                candidate = line

            if len(candidate) > max_chars:
                chunks.append(cur.strip())
                # เริ่ม chunk ใหม่ด้วยส่วน overlap จาก chunk ก่อนหน้า
                if overlap_chars > 0:
                    start = max(0, len(cur) - overlap_chars)
                    cur = cur[start:].strip() + " " + line
                else:
                    cur = line
            else:
                cur = candidate

        if cur:
            chunks.append(cur.strip())

        return [c for c in chunks if c]

    def _build_index(self):
        """Embed (ขั้นตอน 3)"""
        embeddings = self.model.encode(self.chunks, show_progress_bar=False)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings, dtype="float32"))
        return index, embeddings

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """Search (ขั้นตอน 4)"""
        q_emb = self.model.encode([query])
        _, indices = self.index.search(np.array(q_emb, dtype="float32"), top_k)
        return [self.chunks[i] for i in indices[0]]