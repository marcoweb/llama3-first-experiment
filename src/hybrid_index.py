"""
src/hybrid_index.py
Índice Híbrido: Recuperação Léxica (BM25) + Recuperação Semântica (FAISS com embeddings Ollama).
Agregação e fusão de candidatos por RRF (Reciprocal Rank Fusion).
"""

import os
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import requests
import faiss
from rank_bm25 import BM25Okapi
from tqdm import tqdm


STOPWORDS = {
    "the", "of", "in", "and", "a", "to", "for", "with", "on", "at", "by",
    "from", "an", "is", "are", "was", "were", "that", "this", "it", "as",
    "be", "or", "which", "its", "into", "their", "such", "than", "used", "using"
}


def tokenize_text(text: str) -> List[str]:
    """Tokenizador com remoção de stopwords comuns para o BM25."""
    tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    return [w for w in tokens if w not in STOPWORDS]


class OllamaClient:
    def __init__(self, host: str = "http://127.0.0.1:11434", embed_model: str = "mxbai-embed-large"):
        self.host = host.rstrip("/")
        self.embed_model = embed_model

    def get_embedding(self, text: str) -> np.ndarray:
        """Gera embedding para um texto individual."""
        res = self.get_embeddings_batch([text])
        return res[0]

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Gera embeddings em lotes para lista de textos."""
        all_embeddings = []
        url = f"{self.host}/api/embed"

        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            payload = {"model": self.embed_model, "input": chunk}
            resp = requests.post(url, json=payload, timeout=120)
            if resp.status_code != 200:
                raise RuntimeError(f"Erro ao gerar embeddings no Ollama ({resp.status_code}): {resp.text}")
            data = resp.json()
            embeddings = data.get("embeddings", [])
            all_embeddings.extend(embeddings)

        arr = np.array(all_embeddings, dtype=np.float32)
        # Normalização L2 para similaridade de cosseno via produto interno (Inner Product)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        return arr


class HybridIndex:
    def __init__(
        self,
        vocabulary: List[str],
        cache_dir: Path,
        name: str = "vocab",
        ollama_client: Optional[OllamaClient] = None
    ):
        self.vocabulary = vocabulary
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.ollama = ollama_client or OllamaClient()

        self.bm25: Optional[BM25Okapi] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.embeddings: Optional[np.ndarray] = None

    def build_or_load_bm25(self) -> None:
        """Inicializa o índice léxico BM25 sobre os descritores do vocabulário."""
        print(f"Construindo índice BM25 sobre {len(self.vocabulary)} termos do vocabulário...")
        tokenized_corpus = [tokenize_text(term) for term in self.vocabulary]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("Índice BM25 pronto.")

    def build_or_load_faiss(self, force_rebuild: bool = False, max_terms: Optional[int] = None) -> None:
        """Gera embeddings dos descritores ou carrega do cache local de forma incremental."""
        emb_path = self.cache_dir / f"{self.name}_embeddings.npy"
        terms_path = self.cache_dir / f"{self.name}_terms.json"

        vocab_to_index = self.vocabulary
        if max_terms is not None and max_terms < len(vocab_to_index):
            vocab_to_index = vocab_to_index[:max_terms]

        if not force_rebuild and emb_path.exists() and terms_path.exists():
            with open(terms_path, "r", encoding="utf-8") as f:
                cached_terms = json.load(f)
            if cached_terms == vocab_to_index:
                print(f"Carregando embeddings cacheados de {emb_path} ({len(cached_terms)} termos)...")
                self.embeddings = np.load(emb_path)
                dim = self.embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dim)
                self.faiss_index.add(self.embeddings)
                print("Índice FAISS carregado com sucesso do cache.")
                return

        print(f"Gerando embeddings para {len(vocab_to_index)} termos via Ollama ({self.ollama.embed_model})...")
        batch_size = 100
        all_embeddings = []
        
        # Se já existe arquivo parcial com os mesmos termos iniciais, carregar
        start_idx = 0
        if emb_path.exists() and terms_path.exists() and not force_rebuild:
            try:
                with open(terms_path, "r", encoding="utf-8") as f:
                    partial_terms = json.load(f)
                partial_embs = np.load(emb_path)
                if len(partial_terms) < len(vocab_to_index) and vocab_to_index[:len(partial_terms)] == partial_terms:
                    print(f"Retomando geração a partir do termo {len(partial_terms)}/{len(vocab_to_index)}...")
                    all_embeddings.append(partial_embs)
                    start_idx = len(partial_terms)
            except Exception:
                all_embeddings = []
                start_idx = 0

        for i in tqdm(range(start_idx, len(vocab_to_index), batch_size), desc="Gerando Embeddings"):
            chunk = vocab_to_index[i : i + batch_size]
            chunk_embs = self.ollama.get_embeddings_batch(chunk, batch_size=len(chunk))
            all_embeddings.append(chunk_embs)
            
            # Salvar checkpoint periodicamente a cada 1000 termos
            if (i - start_idx + len(chunk)) % 1000 == 0 or (i + batch_size >= len(vocab_to_index)):
                curr_stacked = np.vstack(all_embeddings).astype(np.float32)
                np.save(emb_path, curr_stacked)
                with open(terms_path, "w", encoding="utf-8") as f:
                    json.dump(vocab_to_index[: curr_stacked.shape[0]], f, ensure_ascii=False)

        self.embeddings = np.vstack(all_embeddings).astype(np.float32)
        print(f"Salvando embeddings finais em cache: {emb_path} ({self.embeddings.shape[0]} vetores)")
        np.save(emb_path, self.embeddings)
        with open(terms_path, "w", encoding="utf-8") as f:
            json.dump(vocab_to_index, f, ensure_ascii=False, indent=2)

        dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.embeddings)
        print("Índice FAISS construído e indexado com sucesso.")

    def retrieve_lexical(self, query: str, top_k: int = 30) -> List[Tuple[str, float]]:
        """Recupera os top-K termos por similaridade lexical (BM25)."""
        if self.bm25 is None:
            self.build_or_load_bm25()
        tokens = tokenize_text(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(self.vocabulary[idx], float(scores[idx])) for idx in top_indices if scores[idx] > 0]
        return results

    def retrieve_semantic(self, query: str, top_k: int = 30) -> List[Tuple[str, float]]:
        """Recupera os top-K termos por similaridade semântica (FAISS)."""
        if self.faiss_index is None:
            raise RuntimeError("Índice FAISS não inicializado. Chame build_or_load_faiss primeiro.")
        query_emb = self.ollama.get_embedding(query)
        query_emb = np.expand_dims(query_emb, axis=0)  # shape (1, dim)
        distances, indices = self.faiss_index.search(query_emb, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.vocabulary):
                results.append((self.vocabulary[idx], float(dist)))
        return results

    def retrieve_hybrid_candidates(
        self,
        query: str,
        top_k_lex: int = 30,
        top_k_sem: int = 30,
        total_candidates: int = 35,
        rrf_k: int = 60
    ) -> List[str]:
        """
        Combina candidatos léxicos e semânticos através de Reciprocal Rank Fusion (RRF).
        Retorna lista consolidada de termos candidatos únicos.
        """
        lex_results = self.retrieve_lexical(query, top_k=top_k_lex)
        sem_results = self.retrieve_semantic(query, top_k=top_k_sem)

        rrf_scores: Dict[str, float] = {}

        # Pontuação RRF para resultados léxicos
        for rank, (term, _) in enumerate(lex_results, start=1):
            rrf_scores[term] = rrf_scores.get(term, 0.0) + (1.0 / (rrf_k + rank))

        # Pontuação RRF para resultados semânticos
        for rank, (term, _) in enumerate(sem_results, start=1):
            rrf_scores[term] = rrf_scores.get(term, 0.0) + (1.0 / (rrf_k + rank))

        # Ordenar candidatos por pontuação RRF decrescente
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        final_candidates = [term for term, _ in sorted_candidates[:total_candidates]]
        return final_candidates
