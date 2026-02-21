import json
import re
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
        self.document_map = {}
        for doc in documents:
            self.document_map = {doc["id"]: doc for doc in documents}
        all_chunks = []
        chunk_metadata = []

        for midx, doc in enumerate(documents):
            if doc["description"].strip == "":
                continue
            _chunks = semantic_chunking(doc["description"], overlap=1, max_chunk_size=4)
            all_chunks += _chunks
            for cidx in range(len(_chunks)):
                chunk_metadata.append(
                    {"movie_idx": midx, "chunk_idx": cidx, "total_chunks": len(_chunks)}
                )

        self.chunk_embeddings = self.model.encode(all_chunks)
        self.chunk_metadata = chunk_metadata

        np.save(self.chunk_embeddings_path, self.chunk_embeddings)

        with open(self.chunk_metadata_path, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2
            )
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}

        if self.chunk_metadata_path.exists() and self.chunk_metadata_path.exists():
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "r") as f:
                self.chunk_metadata = json.load(f)
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)


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
