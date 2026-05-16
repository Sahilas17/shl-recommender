"""
build_index.py
Run this ONCE to build the TF-IDF retrieval index from shl_catalog.json.
Output: shl_retrieval.pkl

Usage:
    python build_index.py
"""

import json
import pickle
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def build_text_for_product(p: dict) -> str:
    """Create a rich text representation of a product for TF-IDF indexing."""
    # Repeat name twice to give it higher weight
    name = p.get("name", "")
    description = p.get("description", "")
    test_types = " ".join(p.get("test_types", []))
    job_levels = " ".join(p.get("job_levels", []))
    skills = " ".join(p.get("skills_measured", []))
    duration = p.get("duration", "")

    # Map type codes to readable words for better matching
    type_words = {
        "A": "ability aptitude reasoning cognitive",
        "K": "knowledge skills technical domain",
        "P": "personality behavior traits questionnaire",
        "S": "simulation situational judgment coding"
    }
    type_expansion = " ".join(
        type_words.get(t, t) for t in p.get("test_types", [])
    )

    return (
        f"{name} {name} {name} "  # 3x weight for name
        f"{description} "
        f"{test_types} {type_expansion} "
        f"{job_levels} "
        f"{skills} {skills} "      # 2x weight for skills
        f"{duration}"
    ).lower()


def build_index(catalog_path: str = "shl_catalog.json",
                output_path: str = "shl_retrieval.pkl") -> None:
    """Build and save the TF-IDF retrieval index."""

    # Load catalog
    if not os.path.exists(catalog_path):
        print(f"ERROR: {catalog_path} not found. Cannot build index.")
        sys.exit(1)

    with open(catalog_path) as f:
        products = json.load(f)

    print(f"Loaded {len(products)} products from {catalog_path}")

    # Build text corpus
    texts = [build_text_for_product(p) for p in products]

    # Fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),      # unigrams and bigrams
        max_features=8000,        # vocabulary size
        sublinear_tf=True,        # log normalization
        min_df=1,                 # include all terms
        analyzer="word",
        token_pattern=r"(?u)\b[a-zA-Z0-9#+.][a-zA-Z0-9#+.]*\b"  # keep C#, C++, etc.
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    # Save
    payload = {
        "products": products,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "texts": texts
    }

    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Index saved to {output_path}")

    # Quick smoke test
    print("\n--- Smoke Tests ---")
    test_queries = [
        "Java developer mid level",
        "personality assessment leadership manager",
        "data scientist Python machine learning",
        "entry level customer service",
        "numerical reasoning graduate",
        "agile project manager stakeholder",
    ]

    for query in test_queries:
        qvec = vectorizer.transform([query.lower()])
        scores = cosine_similarity(qvec, tfidf_matrix).flatten()
        top3 = np.argsort(scores)[::-1][:3]
        print(f"\nQuery: '{query}'")
        for idx in top3:
            print(f"  [{scores[idx]:.3f}] {products[idx]['name']}")


if __name__ == "__main__":
    build_index()
