from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Iterable, Protocol

import numpy as np
import pandas as pd
from openai import OpenAI

from pipeline.config import PipelineConfig


@dataclass
class ClusterLabel:
    title: str
    summary: str
    relevant_article_ids: list[int]


class Labeler(Protocol):
    def label_cluster(self, articles: pd.DataFrame) -> ClusterLabel:
        ...

    def embed_text(self, text: str) -> np.ndarray:
        ...

    def generate_theme_title(self, clusters: pd.DataFrame) -> str:
        ...

    def generate_theme_title_summary(self, clusters: pd.DataFrame) -> tuple[str, str]:
        ...


def _safe_title(text: str, max_words: int = 10) -> str:
    words = text.strip().split()
    if not words:
        return "Unlabeled cluster"
    return " ".join(words[:max_words])


def _heuristic_cluster_label(articles: pd.DataFrame) -> ClusterLabel:
    article_ids = articles["id"].astype(int).tolist()
    first_title = str(articles.iloc[0].get("title", "")).strip() if not articles.empty else ""
    title = _safe_title(first_title or "Cluster storyline")

    top_titles = [str(t).strip() for t in articles["title"].head(3).tolist()]
    top_titles = [t for t in top_titles if t]
    if top_titles:
        summary = " | ".join(top_titles)
    else:
        summary = "This cluster groups related articles from the same publication day."

    return ClusterLabel(title=title, summary=summary, relevant_article_ids=article_ids)


def _deterministic_vector(text: str, dims: int = 256) -> np.ndarray:
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    vec = rng.normal(0, 1, size=dims)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z]{3,}", text.lower())
    stop = {
        "the",
        "and",
        "with",
        "from",
        "that",
        "this",
        "into",
        "after",
        "over",
        "under",
        "amid",
        "news",
        "says",
        "said",
        "report",
    }
    return [w for w in words if w not in stop]


class MockLabeler:
    def label_cluster(self, articles: pd.DataFrame) -> ClusterLabel:
        return _heuristic_cluster_label(articles)

    def embed_text(self, text: str) -> np.ndarray:
        return _deterministic_vector(text, dims=256)

    def generate_theme_title(self, clusters: pd.DataFrame) -> str:
        bag: Counter[str] = Counter()
        for _, row in clusters.head(12).iterrows():
            bag.update(_tokenize(str(row.get("title", ""))))
            bag.update(_tokenize(str(row.get("summary", ""))))
        top = [word for word, _ in bag.most_common(4)]
        if not top:
            return "Cross-day theme"
        return _safe_title(" ".join(top), max_words=8).title()

    def generate_theme_title_summary(self, clusters: pd.DataFrame) -> tuple[str, str]:
        title = self.generate_theme_title(clusters)

        snippets: list[str] = []
        for _, row in clusters.head(5).iterrows():
            t = str(row.get("title", "")).strip()
            s = str(row.get("summary", "")).strip()
            if t:
                snippets.append(t)
            if s:
                snippets.append(s)
            if len(snippets) >= 3:
                break

        if snippets:
            summary = " ".join(snippets[:3])
        else:
            summary = "This theme combines related clusters across dates."
        return title, summary


class OpenAILabeler:
    def __init__(self, config: PipelineConfig):
        self._client = OpenAI(api_key=config.openai_api_key)
        self._chat_model = config.openai_chat_model
        self._embed_model = config.openai_embed_model

    def label_cluster(self, articles: pd.DataFrame) -> ClusterLabel:
        if articles.empty:
            return ClusterLabel("Unlabeled cluster", "No articles in cluster.", [])

        prompt_chunks = []
        for _, row in articles.iterrows():
            prompt_chunks.append(
                "\n".join(
                    [
                        f"ID: {int(row['id'])}",
                        f"Title: {str(row.get('title') or '').strip()}",
                        f"Summary: {str(row.get('summary') or '').strip()}",
                    ]
                )
            )
        content = "\n\n".join(prompt_chunks)

        try:
            response = self._client.chat.completions.create(
                model=self._chat_model,
                temperature=0.2,
                max_tokens=500,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You label clusters of related finance/news articles. "
                            "Return concise JSON with fields: title (<=10 words), summary (2-3 sentences), "
                            "relevant_article_ids (subset of input IDs). Output valid JSON only."
                        ),
                    },
                    {"role": "user", "content": content},
                ],
            )
            payload = response.choices[0].message.content or "{}"
            data = json.loads(payload)

            title = _safe_title(str(data.get("title", "")).strip() or "Unlabeled cluster")
            summary = str(data.get("summary", "")).strip() or "Cluster summary unavailable."
            raw_ids = data.get("relevant_article_ids", [])
            if isinstance(raw_ids, str):
                try:
                    raw_ids = json.loads(raw_ids)
                except json.JSONDecodeError:
                    raw_ids = [x.strip() for x in raw_ids.split(",") if x.strip()]
            if not isinstance(raw_ids, list):
                raw_ids = []
            ids = []
            for value in raw_ids:
                try:
                    ids.append(int(value))
                except (TypeError, ValueError):
                    continue

            return ClusterLabel(title=title, summary=summary, relevant_article_ids=ids)
        except Exception:
            return _heuristic_cluster_label(articles)

    def embed_text(self, text: str) -> np.ndarray:
        try:
            response = self._client.embeddings.create(model=self._embed_model, input=text)
            return np.asarray(response.data[0].embedding, dtype=float)
        except Exception:
            return _deterministic_vector(text, dims=256)

    def generate_theme_title(self, clusters: pd.DataFrame) -> str:
        title, _ = self.generate_theme_title_summary(clusters)
        return title

    def generate_theme_title_summary(self, clusters: pd.DataFrame) -> tuple[str, str]:
        if clusters.empty:
            return "Cross-day theme", "No theme content available."

        snippets: list[str] = []
        for _, row in clusters.head(15).iterrows():
            snippets.append(f"Title: {row.get('title', '')}\nSummary: {row.get('summary', '')}")
        prompt = "\n\n".join(snippets)

        try:
            response = self._client.chat.completions.create(
                model=self._chat_model,
                temperature=0.2,
                max_tokens=200,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are creating metadata for a cross-day news theme. "
                            "Given cluster titles and summaries, return JSON with:\n"
                            "- title: concise theme title (max 8 words)\n"
                            "- summary: 2-3 sentence theme overview that synthesizes the shared storyline.\n"
                            "Return valid JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            payload = response.choices[0].message.content or "{}"
            data = json.loads(payload)
            title = _safe_title(str(data.get("title", "")).strip() or "Cross-day theme", max_words=8)
            summary = str(data.get("summary", "")).strip() or "Theme summary unavailable."
            return title, summary
        except Exception:
            return MockLabeler().generate_theme_title_summary(clusters)


def build_labeler(config: PipelineConfig) -> Labeler:
    if config.mock_openai:
        return MockLabeler()
    return OpenAILabeler(config)


def _compute_density(embeddings: np.ndarray, centroid: np.ndarray) -> float:
    if embeddings.shape[0] == 0:
        return 0.0
    distances = 1 - np.dot(embeddings, centroid) / (
        (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(centroid)) + 1e-12
    )
    median_distance = float(np.median(distances)) if distances.size else 0.0
    return float(embeddings.shape[0] / (median_distance * median_distance + 1e-5))


def label_clusters(
    articles_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    labeler: Labeler,
    top_n_per_day: int | None = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    articles = articles_df.copy()
    labeled_rows: list[dict[str, object]] = []

    ordered_clusters = clusters_df.sort_values(["date", "density"], ascending=[True, False]).copy()
    if top_n_per_day is not None and top_n_per_day > 0:
        ordered_clusters = (
            ordered_clusters.groupby("date", group_keys=False)
            .head(top_n_per_day)
            .reset_index(drop=True)
        )

    for _, cluster in ordered_clusters.iterrows():
        cluster_id = int(cluster["id"])
        cluster_articles = articles[articles["cluster_id"] == cluster_id].copy()
        if cluster_articles.empty:
            continue

        label = labeler.label_cluster(cluster_articles)
        valid_ids = set(cluster_articles["id"].astype(int).tolist())
        relevant_ids = [aid for aid in label.relevant_article_ids if aid in valid_ids]
        if not relevant_ids:
            relevant_ids = sorted(valid_ids)

        outlier_ids = [aid for aid in valid_ids if aid not in set(relevant_ids)]
        if outlier_ids:
            articles.loc[articles["id"].isin(outlier_ids), "cluster_id"] = -1

        relevant_articles = cluster_articles[cluster_articles["id"].isin(relevant_ids)].copy()
        if relevant_articles.empty:
            continue

        vectors = np.vstack(relevant_articles["embedding_vec"].to_numpy())
        centroid = vectors.mean(axis=0)
        density = _compute_density(vectors, centroid)

        title = _safe_title(label.title)
        summary = label.summary.strip() or "Cluster summary unavailable."
        embedding_text = f"{title}\n{summary}".strip()
        cluster_embedding = labeler.embed_text(embedding_text)

        labeled_rows.append(
            {
                "id": cluster_id,
                "date": cluster["date"],
                "num_articles": int(vectors.shape[0]),
                "density": float(density),
                "centroid": centroid,
                "embedding": cluster_embedding,
                "title": title,
                "summary": summary,
            }
        )

    labeled_df = pd.DataFrame(labeled_rows)
    if not labeled_df.empty:
        labeled_df = labeled_df.sort_values(["date", "density"], ascending=[True, False]).reset_index(
            drop=True
        )

    return articles.reset_index(drop=True), labeled_df


def prepare_clusters_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in ["centroid", "embedding"]:
        if column in out.columns:
            out[column] = out[column].apply(lambda vec: json.dumps(list(map(float, vec)), separators=(",", ":")))
    return out
