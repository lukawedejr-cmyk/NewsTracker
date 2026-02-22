from __future__ import annotations

from datetime import timedelta
from itertools import combinations
from typing import Any

import hdbscan
import numpy as np
import pandas as pd
from umap import UMAP

from pipeline.io import parse_embedding_cell
from pipeline.labeling import Labeler


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return -1.0
    return float(np.dot(vec_a, vec_b) / denom)


def _median_pairwise_distance(vectors: list[np.ndarray]) -> float:
    vectors = [np.asarray(v, dtype=float) for v in vectors if np.asarray(v).size > 0]
    if len(vectors) < 2:
        return 1.0

    distances = [float(np.linalg.norm(a - b)) for a, b in combinations(vectors, 2)]
    if not distances:
        return 1.0
    return max(float(np.median(distances)), 1e-5)


def assign_themes(
    df: pd.DataFrame,
    similarity_threshold: float = 0.15,
    lookback_days: int | None = 0,
    min_cluster_size: int = 2,
    min_samples: int = 1,
    umap_neighbors: int = 15,
    umap_components: int = 50,
) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["theme"] = pd.Series(dtype="int64")
        return out

    res = df.copy()
    res["date"] = pd.to_datetime(res["date"], utc=True, errors="coerce").dt.date
    res = res[res["date"].notna()].copy()
    res["embedding"] = res["embedding"].apply(parse_embedding_cell)
    res = res.sort_values(["date", "id"]).reset_index(drop=True)
    res["theme"] = -1

    theme_counter = 0
    history: pd.DataFrame | None = None

    for current_date, group in res.groupby("date", sort=True):
        group = group.copy()

        if history is None:
            embeddings = np.vstack(group["embedding"].to_numpy())
            n_samples = embeddings.shape[0]

            if n_samples < 4:
                for idx in group.index:
                    theme_counter += 1
                    res.at[idx, "theme"] = theme_counter
                history = res.loc[group.index, ["date", "embedding", "theme"]].copy()
                continue

            local_neighbors = max(2, min(umap_neighbors, n_samples - 1))
            local_components = max(2, min(umap_components, n_samples - 2))

            try:
                reducer = UMAP(
                    n_neighbors=local_neighbors,
                    n_components=local_components,
                    min_dist=0.1,
                    random_state=42,
                )
                reduced = reducer.fit_transform(embeddings)

                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min(min_cluster_size, n_samples),
                    min_samples=min(min_samples, n_samples),
                    metric="euclidean",
                    cluster_selection_method="leaf",
                )
                labels = clusterer.fit_predict(reduced)
            except Exception:
                labels = np.arange(n_samples)

            label_to_theme: dict[int, int] = {}
            for idx, label in zip(group.index, labels):
                label = int(label)
                if label == -1:
                    theme_counter += 1
                    res.at[idx, "theme"] = theme_counter
                else:
                    if label not in label_to_theme:
                        theme_counter += 1
                        label_to_theme[label] = theme_counter
                    res.at[idx, "theme"] = label_to_theme[label]

            history = res.loc[group.index, ["date", "embedding", "theme"]].copy()
            continue

        candidate_pool = history
        if lookback_days is not None and lookback_days > 0:
            cutoff = current_date - timedelta(days=lookback_days)
            candidate_pool = history[history["date"] >= cutoff].copy()
            if candidate_pool.empty:
                candidate_pool = history

        for idx, row in group.iterrows():
            current_vec = np.asarray(row["embedding"], dtype=float)
            best_sim = -1.0
            best_theme = None

            for _, prev_row in candidate_pool.iterrows():
                prev_vec = np.asarray(prev_row["embedding"], dtype=float)
                sim = _cosine_similarity(current_vec, prev_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_theme = int(prev_row["theme"])

            if best_theme is not None and best_sim >= similarity_threshold:
                res.at[idx, "theme"] = best_theme
            else:
                theme_counter += 1
                res.at[idx, "theme"] = theme_counter

        day_history = res.loc[group.index, ["date", "embedding", "theme"]].copy()
        history = pd.concat([history, day_history], ignore_index=True)

    res["theme"] = res["theme"].astype(int)
    return res


def compute_theme_relevancy(themed_df: pd.DataFrame) -> pd.DataFrame:
    if themed_df.empty:
        return pd.DataFrame(
            columns=[
                "theme",
                "relevancy_score",
                "sum_density",
                "median_centroid_distance",
                "num_clusters",
                "total_articles",
            ]
        )

    df = themed_df.copy()
    df["density"] = pd.to_numeric(df["density"], errors="coerce").fillna(0.0)
    df["num_articles"] = pd.to_numeric(df["num_articles"], errors="coerce").fillna(0)
    df["centroid"] = df["centroid"].apply(parse_embedding_cell)

    rows: list[dict[str, Any]] = []
    for theme, group in df.groupby("theme"):
        sum_density = float(group["density"].sum())
        vectors = group["centroid"].tolist()
        median_distance = _median_pairwise_distance(vectors)
        score = float(sum_density / median_distance)
        rows.append(
            {
                "theme": int(theme),
                "relevancy_score": score,
                "sum_density": sum_density,
                "median_centroid_distance": median_distance,
                "num_clusters": int(len(group)),
                "total_articles": int(group["num_articles"].sum()),
            }
        )

    return pd.DataFrame(rows).sort_values("relevancy_score", ascending=False).reset_index(drop=True)


def build_theme_payload(
    themed_df: pd.DataFrame,
    relevancy_df: pd.DataFrame,
    labeler: Labeler,
    top_k: int = 20,
) -> dict[str, Any]:
    if themed_df.empty or relevancy_df.empty:
        return {"top_k": top_k, "themes": []}

    selected = relevancy_df.head(top_k).copy()
    selected_themes = set(selected["theme"].astype(int).tolist())

    df_selected = themed_df[themed_df["theme"].isin(selected_themes)].copy()
    df_selected["date"] = pd.to_datetime(df_selected["date"], utc=True, errors="coerce")

    payload: dict[str, Any] = {"top_k": top_k, "themes": []}

    for _, row in selected.iterrows():
        theme_id = int(row["theme"])
        theme_df = df_selected[df_selected["theme"] == theme_id].copy()
        if theme_df.empty:
            continue

        theme_df = theme_df.sort_values(["date", "density"], ascending=[True, False])

        theme_title, theme_summary = labeler.generate_theme_title_summary(theme_df)
        if not theme_summary:
            representative = theme_df.iloc[0]
            theme_summary = str(representative.get("summary", "")).strip()

        clusters = []
        for _, cluster in theme_df.iterrows():
            cluster_date = pd.to_datetime(cluster["date"], utc=True, errors="coerce")
            clusters.append(
                {
                    "id": int(cluster["id"]),
                    "date": cluster_date.strftime("%Y-%m-%d") if pd.notna(cluster_date) else None,
                    "title": str(cluster.get("title", "")).strip(),
                    "summary": str(cluster.get("summary", "")).strip(),
                    "num_articles": int(cluster.get("num_articles", 0)),
                    "density": float(cluster.get("density", 0.0)),
                }
            )

        timeseries_df = (
            theme_df.groupby(theme_df["date"].dt.date)
            .agg(total_articles=("num_articles", "sum"), num_clusters=("id", "count"))
            .reset_index()
            .sort_values("date")
        )
        timeseries = [
            {
                "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                "total_articles": int(a),
                "num_clusters": int(c),
            }
            for d, a, c in zip(
                timeseries_df["date"],
                timeseries_df["total_articles"],
                timeseries_df["num_clusters"],
            )
        ]

        payload["themes"].append(
            {
                "theme": theme_id,
                "title": theme_title,
                "summary": theme_summary,
                "relevancy_score": float(row["relevancy_score"]),
                "sum_density": float(row["sum_density"]),
                "median_centroid_distance": float(row["median_centroid_distance"]),
                "num_clusters": int(row["num_clusters"]),
                "total_articles": int(row["total_articles"]),
                "clusters": clusters,
                "timeseries": timeseries,
            }
        )

    payload["themes"] = sorted(payload["themes"], key=lambda x: x["relevancy_score"], reverse=True)
    return payload
