# -*- coding: utf-8 -*-
"""
Near-duplicate detection and grouping.

Uses TF-IDF char n-gram cosine similarity + Union-Find clustering to group
near-duplicate texts so they can be kept in the same split.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict


# ── Union-Find ──────────────────────────────────────────────────

class UnionFind:
    """Disjoint-set / Union-Find with path compression and union by rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


# ── Near-dup clustering ────────────────────────────────────────

def build_near_dup_groups(
    texts: List[str],
    threshold: float = 0.92,
    k: int = 20,
    ngram_range: tuple = (4, 6),
    min_df: int = 2,
) -> np.ndarray:
    """
    Cluster near-duplicates using TF-IDF char-ngram cosine similarity.

    Returns
    -------
    group_ids : np.ndarray of shape (n,)
        Cluster id for each text; near-dups share the same id.
    """
    n = len(texts)
    if n == 0:
        return np.array([], dtype=int)

    # Build TF-IDF matrix (char n-grams respecting word boundaries)
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=min_df,
        dtype=np.float32,
    )
    X = vec.fit_transform(texts)

    # Find k nearest neighbors using cosine distance
    nn = NearestNeighbors(
        n_neighbors=min(k, n),
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    nn.fit(X)
    dists, idxs = nn.kneighbors(X, return_distance=True)

    # Union neighbors with similarity >= threshold
    uf = UnionFind(n)
    for i in range(n):
        for dist, j in zip(dists[i], idxs[i]):
            if j == i:
                continue
            similarity = 1.0 - float(dist)
            if similarity >= threshold:
                uf.union(i, int(j))

    # Compress root ids to consecutive integers
    roots = np.array([uf.find(i) for i in range(n)], dtype=int)
    unique_roots = {}
    group_ids = np.zeros(n, dtype=int)
    counter = 0
    for i, root in enumerate(roots):
        if root not in unique_roots:
            unique_roots[root] = counter
            counter += 1
        group_ids[i] = unique_roots[root]

    return group_ids


# ── Leak reporting ──────────────────────────────────────────────

def report_leak_exact(
    a_texts: List[str], b_texts: List[str], name: str
) -> int:
    """Count exact duplicate texts between two sets."""
    overlap = len(set(a_texts) & set(b_texts))
    print(f"  Exact leak {name}: {overlap}")
    return overlap


def report_leak_near(
    a_texts: List[str],
    b_texts: List[str],
    threshold: float = 0.92,
    ngram_range: tuple = (4, 6),
    min_df: int = 2,
) -> Dict[str, float]:
    """
    Near-duplicate leak report between two text sets.

    For each text in B, find the most similar text in A.
    Report statistics of those max-similarities.
    """
    if len(a_texts) == 0 or len(b_texts) == 0:
        return {"mean": 0, "median": 0, "p95": 0, "max": 0, "count_ge_thr": 0}

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=min_df,
        dtype=np.float32,
    )
    X_a = vec.fit_transform(a_texts)
    X_b = vec.transform(b_texts)

    nn = NearestNeighbors(
        n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=-1
    ).fit(X_a)
    dists, _ = nn.kneighbors(X_b, return_distance=True)
    similarities = 1.0 - dists.reshape(-1)

    stats = {
        "mean": float(np.mean(similarities)),
        "median": float(np.median(similarities)),
        "p95": float(np.quantile(similarities, 0.95)),
        "max": float(np.max(similarities)),
        "count_ge_thr": int(np.sum(similarities >= threshold)),
    }
    return stats
