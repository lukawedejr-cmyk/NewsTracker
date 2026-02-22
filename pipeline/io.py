from __future__ import annotations

from datetime import date
import ast
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"id", "title", "published", "embedding"}
OPTIONAL_COLUMNS = {"summary", "link", "source", "fetched_at", "cluster_id"}


def parse_embedding_cell(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value.astype(float)
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=float)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Empty embedding string")
        parsed = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                cleaned = text.replace("[", " ").replace("]", " ").replace("\n", " ")
                arr = np.fromstring(cleaned, sep=",", dtype=float)
                if arr.size == 0:
                    arr = np.fromstring(cleaned, sep=" ", dtype=float)
                if arr.size == 0:
                    raise ValueError("Could not parse embedding string")
                return arr
        arr = np.asarray(parsed, dtype=float)
    else:
        raise ValueError(f"Unsupported embedding type: {type(value).__name__}")

    arr = np.ravel(arr)
    if arr.size == 0:
        raise ValueError("Embedding is empty")
    if not np.isfinite(arr).all():
        raise ValueError("Embedding contains non-finite values")
    return arr


def vector_to_json(vector: Iterable[float]) -> str:
    return json.dumps(list(map(float, vector)), separators=(",", ":"))


def load_articles_csv(
    input_path: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> tuple[pd.DataFrame, int]:
    csv_path = Path(input_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(csv_path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    for column in OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = None

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df[df["id"].notna()].copy()
    df["id"] = df["id"].astype(np.int64)

    df["title"] = df["title"].astype(str).str.strip()
    df = df[df["title"] != ""].copy()

    df["summary"] = df["summary"].fillna("").astype(str)

    df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    df = df[df["published"].notna()].copy()
    df["published_date"] = df["published"].dt.date

    if start_date is not None:
        df = df[df["published_date"] >= start_date].copy()
    if end_date is not None:
        df = df[df["published_date"] <= end_date].copy()

    df = df.sort_values(["published", "id"]).drop_duplicates(
        subset=["published_date", "title"], keep="first"
    )

    embeddings: list[np.ndarray] = []
    invalid_rows: list[int] = []
    for i, value in enumerate(df["embedding"].tolist()):
        try:
            embeddings.append(parse_embedding_cell(value))
        except ValueError:
            invalid_rows.append(i)

    if invalid_rows:
        preview = invalid_rows[:10]
        raise ValueError(
            f"Failed to parse {len(invalid_rows)} embedding rows. Sample row indexes: {preview}"
        )

    dimensions = {vec.size for vec in embeddings}
    if len(dimensions) != 1:
        raise ValueError(
            f"Embedding dimensions are inconsistent: {sorted(dimensions)}"
        )

    embedding_dim = dimensions.pop()
    df = df.copy()
    df["embedding_vec"] = embeddings

    return df.reset_index(drop=True), embedding_dim


def prepare_articles_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "embedding_vec" in out.columns:
        out["embedding"] = out["embedding_vec"].apply(vector_to_json)
        out = out.drop(columns=["embedding_vec"])
    out = out.sort_values(["published", "id"]).reset_index(drop=True)
    return out
