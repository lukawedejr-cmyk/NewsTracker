from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.clustering import cluster_articles_daily
from pipeline.config import load_config
from pipeline.io import load_articles_csv, prepare_articles_for_csv
from pipeline.labeling import build_labeler, label_clusters, prepare_clusters_for_csv
from pipeline.report import render_report
from pipeline.theming import assign_themes, build_theme_payload, compute_theme_relevancy


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return pd.to_datetime(value, format="%Y-%m-%d", errors="raise").date()


def _compute_tidbits(
    themes_payload: dict[str, Any],
    themed_clusters: pd.DataFrame,
    articles: pd.DataFrame,
) -> dict[str, str]:
    if themed_clusters.empty or not themes_payload.get("themes"):
        return {
            "most_persistent_theme": "N/A",
            "fastest_growing_theme": "N/A",
            "highest_density_cluster": "N/A",
            "most_diverse_source_theme": "N/A",
        }

    theme_title_map = {int(t["theme"]): str(t.get("title", f"Theme {t['theme']}")) for t in themes_payload["themes"]}

    by_theme_dates = themed_clusters.groupby("theme")["date"].nunique()
    persistent_theme = int(by_theme_dates.idxmax())
    persistent_days = int(by_theme_dates.loc[persistent_theme])

    growth_theme = None
    growth_value = -10**9
    for theme_id, theme_group in themed_clusters.groupby("theme"):
        daily = (
            theme_group.groupby("date")["num_articles"]
            .sum()
            .sort_index()
            .astype(int)
        )
        if len(daily) < 2:
            continue
        local_growth = int(daily.diff().fillna(0).max())
        if local_growth > growth_value:
            growth_value = local_growth
            growth_theme = int(theme_id)

    highest_row = themed_clusters.sort_values("density", ascending=False).iloc[0]
    highest_cluster_text = (
        f"{highest_row.get('title', 'Cluster')} (ID {int(highest_row['id'])}, density {float(highest_row['density']):.2f})"
    )

    merged = articles[["cluster_id", "source"]].merge(
        themed_clusters[["id", "theme"]],
        left_on="cluster_id",
        right_on="id",
        how="inner",
    )
    merged["source"] = merged["source"].fillna("Unknown").astype(str)
    source_diversity = merged.groupby("theme")["source"].nunique()

    if not source_diversity.empty:
        diverse_theme = int(source_diversity.idxmax())
        diverse_count = int(source_diversity.loc[diverse_theme])
        diverse_text = f"{theme_title_map.get(diverse_theme, f'Theme {diverse_theme}')} ({diverse_count} sources)"
    else:
        diverse_text = "N/A"

    if growth_theme is None:
        growth_text = "N/A"
    else:
        growth_text = f"{theme_title_map.get(growth_theme, f'Theme {growth_theme}')} (+{growth_value} articles)"

    persistent_text = f"{theme_title_map.get(persistent_theme, f'Theme {persistent_theme}')} ({persistent_days} days)"

    return {
        "most_persistent_theme": persistent_text,
        "fastest_growing_theme": growth_text,
        "highest_density_cluster": highest_cluster_text,
        "most_diverse_source_theme": diverse_text,
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, Path]:
    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)

    config = load_config(mock_openai=args.mock_openai, env_file=args.env_file)
    labeler = build_labeler(config)

    articles, embedding_dim = load_articles_csv(
        input_path=args.input,
        start_date=start_date,
        end_date=end_date,
    )

    clustered_articles, clusters = cluster_articles_daily(articles)
    label_top_per_day = getattr(args, "label_top_per_day", 50)
    labeled_articles, labeled_clusters = label_clusters(
        clustered_articles,
        clusters,
        labeler,
        top_n_per_day=label_top_per_day,
    )

    themed_clusters = assign_themes(
        labeled_clusters,
        similarity_threshold=args.theme_sim_threshold,
        lookback_days=args.theme_lookback_days,
    )

    relevancy = compute_theme_relevancy(themed_clusters)
    themes_payload = build_theme_payload(
        themed_clusters,
        relevancy,
        labeler,
        top_k=args.top_k,
    )

    theme_lookup = themed_clusters[["id", "theme"]].rename(columns={"id": "cluster_id"})
    articles_with_theme = labeled_articles.merge(theme_lookup, on="cluster_id", how="left")
    articles_with_theme["theme"] = articles_with_theme["theme"].fillna(-1).astype(int)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    articles_csv_path = output_dir / "articles_clustered.csv"
    clusters_csv_path = output_dir / "clusters_labeled.csv"
    themes_json_path = output_dir / "themes.json"
    summary_json_path = output_dir / "summary_stats.json"
    report_html_path = output_dir / "report.html"

    prepare_articles_for_csv(articles_with_theme).to_csv(articles_csv_path, index=False)
    prepare_clusters_for_csv(themed_clusters).to_csv(clusters_csv_path, index=False)

    themes_json_path.write_text(json.dumps(themes_payload, indent=2), encoding="utf-8")

    summary_stats = {
        "articles_total": int(len(articles)),
        "embedding_dimension": int(embedding_dim),
        "articles_clustered": int((articles_with_theme["cluster_id"] != -1).sum()),
        "clusters_labeled": int(len(themed_clusters)),
        "themes_count": int(len(themes_payload.get("themes", []))),
        "label_top_per_day": int(label_top_per_day) if label_top_per_day is not None else None,
        "theme_similarity_threshold": float(args.theme_sim_threshold),
        "theme_lookback_days": int(args.theme_lookback_days) if args.theme_lookback_days is not None else None,
        "date_start": str(articles["published_date"].min()) if not articles.empty else None,
        "date_end": str(articles["published_date"].max()) if not articles.empty else None,
        "report_scope": args.report_scope,
        "tidbits": _compute_tidbits(themes_payload, themed_clusters, articles_with_theme),
    }

    summary_json_path.write_text(json.dumps(summary_stats, indent=2), encoding="utf-8")

    top_clusters_report = themed_clusters.sort_values("density", ascending=False).head(args.top_k)
    render_report(
        output_path=str(report_html_path),
        summary_stats=summary_stats,
        top_clusters_df=top_clusters_report,
        themes_payload=themes_payload,
    )

    return {
        "articles_clustered": articles_csv_path,
        "clusters_labeled": clusters_csv_path,
        "themes_json": themes_json_path,
        "summary_stats": summary_json_path,
        "report_html": report_html_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CSV-only news clustering and theme timeline pipeline.")
    parser.add_argument("--input", default="example_articles.csv", help="Path to input articles CSV.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated artifacts.")
    parser.add_argument("--start-date", default=None, help="Optional start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=None, help="Optional end date (YYYY-MM-DD).")
    parser.add_argument("--theme-sim-threshold", type=float, default=0.15, help="Theme carry-over similarity threshold.")
    parser.add_argument(
        "--theme-lookback-days",
        type=int,
        default=0,
        help="How many prior days to consider for theme matching (1 = previous day only, 0 = full history).",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Top number of clusters/themes to include in report payload.")
    parser.add_argument(
        "--label-top-per-day",
        type=int,
        default=50,
        help="Limit labeling (and therefore themes) to top N clusters per day by density. Use 0 to label all.",
    )
    parser.add_argument("--report-scope", default="daily", choices=["daily"], help="Report scope. Daily only for this project.")
    parser.add_argument(
        "--include-example-outputs",
        action="store_true",
        help="Compatibility flag for regenerating checked-in sample artifacts.",
    )
    parser.add_argument(
        "--mock-openai",
        action="store_true",
        help="Use deterministic local mock labeling/embeddings instead of OpenAI calls.",
    )
    parser.add_argument("--env-file", default=None, help="Optional path to .env file.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    outputs = run_pipeline(args)
    print("Pipeline completed successfully.")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
