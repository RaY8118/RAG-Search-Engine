# Project: `rag-search-engine`

## Project Overview

CLI for keyword and semantic search on movie data. Provides inverted index building, keyword search, BM25 scoring, and semantic search using sentence transformers.

**Key Technologies:** Python 3.12+, NLTK (Porter Stemmer, stopwords), NumPy, Sentence Transformers (all-MiniLM-L6-v2), Argparse (CLI), Pickle (caching)

**Architecture:**
- `cli/keyword_search_cli.py` - Keyword search CLI
- `cli/semantic_search_cli.py` - Semantic search CLI  
- `cli/lib/keyword_search.py` - Inverted index, TF/IDF/BM25
- `cli/lib/semantic_search.py` - Semantic embeddings
- `cli/lib/search_utils.py` - Shared utilities

---

## Build, Run & Test Commands

```bash
# Prerequisites
pip install -e .
python -c "import nltk; nltk.download('stopwords')"

# Build index
python cli/keyword_search_cli.py build

# Keyword search
python cli/keyword_search_cli.py search "query"
python cli/keyword_search_cli.py bm25search "query"

# Metrics
python cli/keyword_search_cli.py tf <doc_id> <term>
python cli/keyword_search_cli.py idf <term>
python cli/keyword_search_cli.py tfidf <doc_id> <term>
python cli/keyword_search_cli.py bm25idf <term>
python cli/keyword_search_cli.py bm25tf <doc_id> <term>

# Semantic search
python cli/semantic_search_cli.py embed
python cli/semantic_search_cli.py search "query"

# Linting
pip install ruff
ruff check .
ruff format .

# Testing (no tests exist yet)
pip install pytest pytest-cov
pytest                              # all tests
pytest tests/test_file.py           # single file
pytest tests/test_file.py::test_fn  # single function
pytest --cov=cli/lib                # with coverage
```

---

## Code Style Guidelines

### Imports

Standard library first, then third-party, then local:

```python
import json
import math
import os
import pickle
import string
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer

from lib.search_utils import load_movies, load_stopwords
```

### Formatting

- Line length: 100 characters max
- 4 spaces for indentation (no tabs)
- Black-compatible formatting (Ruff handles this)
- One blank line between top-level definitions

### Type Hints

```python
def tokenize_text(text: str) -> list[str]:
    ...

def get_tf(self, doc_id: int, term: str) -> int:
    ...

def search(self, query: str, limit: int = 5) -> list[dict]:
    ...
```

### Naming Conventions

- **Functions/variables**: `snake_case` (e.g., `tokenize_text`)
- **Classes**: `PascalCase` (e.g., `InvertedIndex`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `BM25_K1`)
- **Private methods**: leading underscore (e.g., `_filter`)

### Classes

- Use `__init__` for instance attributes with type hints
- Group related functionality into classes

### Error Handling

- Validate inputs early, raise `ValueError` for invalid arguments
- Include descriptive messages: `raise ValueError("Can only have 1 token")`

### CLI Design

- Use `argparse` with subparsers
- Type-check arguments (e.g., `type=int`)
- Use match/case for command dispatch

### Caching

- Use Pickle for serialization (`cache/*.pkl`)
- Use NumPy `.npy` for embeddings
- Check existence before rebuilding cached data

---

## Commit Message Conventions

Follow Conventional Commits:

```
<type>: <description>

[optional body]
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`

Examples:
```
feat: Add BM25 TF calculation
fix: Handle empty stopwords file gracefully
```

---

## File Locations

| Purpose | Path |
|---------|------|
| Movie data | `data/movies.json` |
| Stopwords | `data/stopwords.txt` |
| Inverted index | `cache/index.pkl` |
| Document map | `cache/docmap.pkl` |
| Term frequencies | `cache/term_frequencies.pkl` |
| Doc lengths | `cache/doc_lengths.pkl` |
| Embeddings | `cache/movie_embedding.npy` |
| Chunk embeddings | `cache/chunk_embeddings.npy` |
