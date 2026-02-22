from __future__ import annotations

from datetime import date

import hdbscan
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from umap import UMAP


def date_to_int(value: date) -> int:
    year = value.year - 2000
    return year * 10_000 + value.month * 100 + value.day


def _compute_density(embeddings: np.ndarray, centroid: np.ndarray) -> float:
    if embeddings.shape[0] == 0:
        return 0.0
    distances = cosine_distances(embeddings, centroid.reshape(1, -1)).flatten()
    median_distance = float(np.median(distances)) if distances.size else 0.0
    return float(embeddings.shape[0] / (median_distance * median_distance + 1e-5))


def _cluster_day_embeddings(embeddings: np.ndarray) -> np.ndarray:
    n_samples = embeddings.shape[0]
    if n_samples < 4:
        return np.arange(n_samples)

    n_components = max(2, min(50, n_samples - 2))
    n_neighbors = max(2, min(10, n_samples - 1))

    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    reduced = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min(5, n_samples),
        min_samples=min(5, n_samples),
        cluster_selection_epsilon=0.05,
        metric="euclidean",
        cluster_selection_method="leaf",
    )
    labels = clusterer.fit_predict(reduced)

    if hasattr(clusterer, "probabilities_") and hasattr(clusterer, "outlier_scores_"):
        probs = clusterer.probabilities_
        outlier_scores = clusterer.outlier_scores_
        labels = np.where((probs < 0.3) | (outlier_scores > 0.9), -1, labels)

    if np.all(labels == -1):
        return np.arange(n_samples)

    return labels


def cluster_articles_daily(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    articles = df.copy()
    articles["cluster_id"] = -1

    cluster_rows: list[dict[str, object]] = []

    for current_date, group in articles.groupby("published_date", sort=True):
        group = group.copy()
        vectors = np.vstack(group["embedding_vec"].to_numpy())

        try:
            labels = _cluster_day_embeddings(vectors)
        except Exception:
            labels = np.arange(vectors.shape[0])

        unique_labels = sorted({int(x) for x in labels if int(x) != -1})
        label_to_cluster: dict[int, int] = {}
        day_offset = date_to_int(current_date) * 1000
        for idx, local_label in enumerate(unique_labels, start=1):
            label_to_cluster[local_label] = day_offset + idx

        cluster_ids = np.array(
            [label_to_cluster.get(int(lbl), -1) for lbl in labels],
            dtype=np.int64,
        )

        articles.loc[group.index, "cluster_id"] = cluster_ids

        for cluster_id in sorted(set(cluster_ids.tolist())):
            if cluster_id == -1:
                continue

            mask = cluster_ids == cluster_id
            cluster_vectors = vectors[mask]
            centroid = cluster_vectors.mean(axis=0)
            density = _compute_density(cluster_vectors, centroid)

            cluster_rows.append(
                {
                    "id": int(cluster_id),
                    "date": current_date,
                    "num_articles": int(cluster_vectors.shape[0]),
                    "density": float(density),
                    "centroid": centroid,
                }
            )

    clusters_df = pd.DataFrame(cluster_rows)
    if not clusters_df.empty:
        clusters_df = clusters_df.sort_values(["date", "density"], ascending=[True, False]).reset_index(
            drop=True
        )

    return articles.reset_index(drop=True), clusters_df
