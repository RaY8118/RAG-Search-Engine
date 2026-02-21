import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from lib.search_utils import load_movies
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = Path("cache/movie_embedding.npy")

    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        movie_strings = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        if self.embeddings_path.exists():
            self.embeddings = np.load(self.embeddings_path)
            if len(self.documents) == len(self.embeddings):
                return self.embeddings

        return self.build_embeddings(documents)

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("Must have text to create embedding")

        return self.model.encode([text])[0]

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        qry_emb = self.generate_embedding(query)

        similarities = []
        for doc_emb, doc in zip(self.embeddings, self.documents):  # type: ignore
            _similarty = cosine_similarity(qry_emb, doc_emb)
            similarities.append((_similarty, doc))

        similarities.sort(key=lambda x: x[0], reverse=True)
        res = []
        for sc, doc in similarities[:limit]:
            res.append(
                {"score": sc, "title": doc["title"], "description": doc["description"]}
            )
        return res


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_embeddings_path = Path("cache/chunk_embeddings.npy")
        self.chunk_metadata = None
        self.chunk_metadata_path = Path("cache/chunk_metadata.json")

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        all_chunks = []
        chunk_metadata_list = []

        for doc in documents:
            if doc["description"].strip() == "":
                continue
            _chunks = semantic_chunking(doc["description"], overlap=1, max_chunk_size=4)
            all_chunks += _chunks
            for cidx in range(len(_chunks)):
                chunk_metadata_list.append(
                    {
                        "movie_idx": doc["id"],
                        "chunk_idx": cidx,
                        "total_chunks": len(_chunks),
                    }
                )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)

        self.chunk_metadata = {
            "chunks": chunk_metadata_list,
            "total_chunks": len(all_chunks),
        }

        np.save(self.chunk_embeddings_path, self.chunk_embeddings)

        with open(self.chunk_metadata_path, "w") as f:
            json.dump(self.chunk_metadata, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        if self.chunk_metadata_path.exists() and self.chunk_embeddings_path.exists():
            try:
                self.chunk_embeddings = np.load(self.chunk_embeddings_path)
                with open(self.chunk_metadata_path, "r") as f:
                    self.chunk_metadata = json.load(f)

                if self.chunk_metadata is not None and "chunks" in self.chunk_metadata:
                    if len(self.chunk_embeddings) == len(self.chunk_metadata["chunks"]):
                        return self.chunk_embeddings
                    else:
                        print(
                            "Mismatch in number of chunk embeddings and metadata chunks. Rebuilding."
                        )
                else:
                    print("Chunk metadata is invalid or None. Rebuilding.")
            except Exception as e:
                print(f"Error loading cached embeddings or metadata: {e}. Rebuilding.")

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "No chunk embeddings or metadata loaded. Call `load_or_create_chunk_embeddings` first."
            )
        qry_emb = self.generate_embedding(query)

        chunk_scores = []
        movie_scores = defaultdict(lambda: 0.0)

        for idx in range(len(self.chunk_embeddings)):
            chunk_embeddings = self.chunk_embeddings[idx]
            metadata = self.chunk_metadata["chunks"][idx]
            midx, cidx = metadata["movie_idx"], metadata["chunk_idx"]
            sim = cosine_similarity(qry_emb, chunk_embeddings)
            chunk_scores.append({"movie_idx": midx, "chunk_idx": cidx, "score": sim})
            movie_scores[midx] = max(movie_scores[midx], sim)
        movie_scores_sorted = sorted(
            movie_scores.items(), key=lambda x: x[1], reverse=True
        )
        res = []
        for midx, score in movie_scores_sorted[:limit]:
            doc = self.document_map[midx]

            res.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "document": doc["description"][:100],
                    "score": round(score, 4),
                }
            )
        return res


def search_chunked(query: str, limit=5):
    movies = load_movies()
    css = ChunkedSemanticSearch()
    _ = css.load_or_create_chunk_embeddings(movies)
    results = css.search_chunks(query, limit)
    for i, res in enumerate(results):
        print(f"{i + 1}. {res['title']} (score: {res['score']:.4f})")
        print(f"   {res['document']}...")


def embed_chunks():
    movies = load_movies()
    css = ChunkedSemanticSearch()
    embeddings = css.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")


def semantic_chunking(
    text,
    overlap=0,
    max_chunk_size=4,
):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    step_size = max_chunk_size - overlap
    for i in range(0, len(sentences), step_size):
        chunk_sentences = sentences[i : i + max_chunk_size]
        if len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
    return chunks


def chunk_text_semantic(
    text: str,
    overlap: int = 0,
    max_chunk_size: int = 4,
):
    chunks = semantic_chunking(text, overlap, max_chunk_size)
    print(f"Semantic chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")


def fixed_sized_chunking(text, overlap, chunk_size=200):
    words = text.split()
    chunks = []
    step_size = chunk_size - overlap
    for i in range(0, len(words), step_size):
        chunk_words = words[i : i + chunk_size]
        if len(chunk_words) <= overlap:
            break
        chunks.append(" ".join(chunk_words))
    return chunks


def chunk_text(text: str, overlap: int, chunk_size: int = 200):
    chunks = fixed_sized_chunking(text, overlap, chunk_size)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i}. {chunk}")


def search(query: str, limit: int = 5):
    ss = SemanticSearch()
    movies = load_movies()
    ss.load_or_create_embeddings(movies)
    search_result = ss.search(query, limit)
    for idx, res in enumerate(search_result):
        print(f"{idx}. {res['title']} (score: {res['score']:.4f})")
        print(res["description"][:100])


def embed_query_text(query: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def embed_text(text: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded : {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")
