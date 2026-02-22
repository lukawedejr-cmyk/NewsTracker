from __future__ import annotations

from datetime import datetime
import html
from pathlib import Path

import pandas as pd


def _fmt_number(value: float | int) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.2f}"


def _build_timeline_svg(themes: list[dict]) -> str:
    all_dates = sorted({point["date"] for theme in themes for point in theme.get("timeseries", []) if point.get("date")})
    if not all_dates:
        return "<p>No timeline data available.</p>"

    width, height = 1060, 360
    margin_left, margin_right, margin_top, margin_bottom = 70, 20, 20, 60
    inner_w = width - margin_left - margin_right
    inner_h = height - margin_top - margin_bottom

    max_y = 0
    for theme in themes:
        for point in theme.get("timeseries", []):
            max_y = max(max_y, int(point.get("total_articles", 0)))
    if max_y <= 0:
        max_y = 1

    x_lookup = {date: i for i, date in enumerate(all_dates)}
    x_div = max(len(all_dates) - 1, 1)

    def x_pos(date_str: str) -> float:
        return margin_left + inner_w * (x_lookup[date_str] / x_div)

    def y_pos(count: int) -> float:
        return margin_top + inner_h * (1 - (count / max_y))

    colors = [
        "#0f766e",
        "#b91c1c",
        "#1d4ed8",
        "#7c3aed",
        "#b45309",
        "#0e7490",
        "#166534",
        "#9f1239",
        "#4338ca",
        "#92400e",
    ]

    lines = []
    legend = []

    # axes
    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top + inner_h}" x2="{margin_left + inner_w}" y2="{margin_top + inner_h}" stroke="#334155" stroke-width="1" />'
    )
    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + inner_h}" stroke="#334155" stroke-width="1" />'
    )

    for i in range(5):
        tick_value = int(round(max_y * i / 4))
        y = y_pos(tick_value)
        lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + inner_w}" y2="{y:.2f}" stroke="#e2e8f0" stroke-width="1" />'
        )
        lines.append(
            f'<text x="{margin_left - 8}" y="{y + 4:.2f}" text-anchor="end" fill="#334155" font-size="11">{tick_value}</text>'
        )

    label_step = max(1, len(all_dates) // 8)
    for idx, date_str in enumerate(all_dates):
        if idx % label_step != 0 and idx != len(all_dates) - 1:
            continue
        x = x_pos(date_str)
        lines.append(
            f'<text x="{x:.2f}" y="{height - 20}" text-anchor="middle" fill="#334155" font-size="10">{html.escape(date_str)}</text>'
        )

    for idx, theme in enumerate(themes):
        color = colors[idx % len(colors)]
        points = theme.get("timeseries", [])
        if not points:
            continue

        points_lookup = {p["date"]: int(p.get("total_articles", 0)) for p in points if p.get("date")}
        poly_points = []
        for date_str in all_dates:
            count = points_lookup.get(date_str, 0)
            poly_points.append(f"{x_pos(date_str):.2f},{y_pos(count):.2f}")

        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(poly_points)}" />'
        )

        default_title = f"Theme {theme.get('theme')}"
        legend_title = html.escape(str(theme.get("title", default_title)))
        legend.append(
            f'<div class="legend-item"><span class="legend-dot" style="background:{color}"></span>'
            f"<span>{legend_title}</span></div>"
        )

    svg = (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Theme timeline">'
        + "".join(lines)
        + "</svg>"
    )

    return (
        "<div class=\"timeline-wrap\">"
        + svg
        + "<div class=\"legend\">"
        + "".join(legend)
        + "</div></div>"
    )


def render_report(
    output_path: str,
    summary_stats: dict,
    top_clusters_df: pd.DataFrame,
    themes_payload: dict,
) -> None:
    themes = themes_payload.get("themes", [])
    timeline_markup = _build_timeline_svg(themes)

    tidbits = summary_stats.get("tidbits", {})

    cluster_rows = []
    for _, row in top_clusters_df.iterrows():
        cluster_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('date', '')))}</td>"
            f"<td>{int(row.get('id', 0))}</td>"
            f"<td>{_fmt_number(float(row.get('density', 0.0)))}</td>"
            f"<td>{int(row.get('num_articles', 0))}</td>"
            f"<td>{html.escape(str(row.get('title', '')))}</td>"
            f"<td>{html.escape(str(row.get('summary', '')))}</td>"
            "</tr>"
        )

    theme_cards = []
    for theme in themes:
        clusters = theme.get("clusters", [])
        sample_clusters = clusters[:5]
        cluster_list = "".join(
            f"<li><strong>{html.escape(str(c.get('date')))}:</strong> "
            f"{html.escape(str(c.get('title', '')))} "
            f"({_fmt_number(int(c.get('num_articles', 0)))} articles)</li>"
            for c in sample_clusters
        )
        theme_cards.append(
            "<article class='theme-card'>"
            f"<h3>{html.escape(str(theme.get('title', '')))}</h3>"
            f"<p class='theme-meta'>Theme #{int(theme.get('theme', 0))} · Relevancy {_fmt_number(float(theme.get('relevancy_score', 0.0)))}</p>"
            f"<p>{html.escape(str(theme.get('summary', '')))}</p>"
            f"<p class='theme-meta'>Clusters: {int(theme.get('num_clusters', 0))} · Articles: {int(theme.get('total_articles', 0))}</p>"
            "<ul>"
            f"{cluster_list}"
            "</ul>"
            "</article>"
        )

    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>News Cluster Report</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --card: #ffffff;
      --ink: #0f172a;
      --muted: #475569;
      --line: #cbd5e1;
      --accent: #0f766e;
    }}
    body {{ margin: 0; font-family: "Avenir Next", "Segoe UI", sans-serif; color: var(--ink); background: radial-gradient(circle at top right, #dbeafe 0%, var(--bg) 40%); }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 28px 18px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 2rem; }}
    h2 {{ margin: 28px 0 14px; font-size: 1.35rem; }}
    .sub {{ margin: 0; color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap: 12px; margin-top: 14px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px; box-shadow: 0 6px 18px rgba(15,23,42,0.06); }}
    .kpi {{ font-size: 1.6rem; font-weight: 700; margin: 3px 0; }}
    .label {{ color: var(--muted); font-size: 0.9rem; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--card); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid #e2e8f0; text-align: left; vertical-align: top; font-size: 0.92rem; }}
    th {{ background: #f1f5f9; }}
    .timeline-wrap {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 12px; }}
    .legend {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr)); gap: 6px; margin-top: 8px; }}
    .legend-item {{ display: flex; align-items: center; gap: 8px; color: #334155; font-size: 0.86rem; }}
    .legend-dot {{ display:inline-block; width: 11px; height: 11px; border-radius: 99px; }}
    .tidbits {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr)); gap: 12px; }}
    .theme-list {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(320px,1fr)); gap: 12px; }}
    .theme-card {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px; }}
    .theme-card h3 {{ margin: 0 0 8px; color: var(--accent); }}
    .theme-meta {{ color: var(--muted); margin: 0 0 8px; font-size: 0.9rem; }}
    ul {{ margin: 8px 0 0 16px; padding: 0; }}
  </style>
</head>
<body>
  <main>
    <h1>News Clusters and Theme Timeline</h1>
    <p class="sub">Generated {generated_at}</p>

    <section>
      <h2>Run Summary</h2>
      <div class="grid">
        <div class="card"><div class="label">Input Articles</div><div class="kpi">{_fmt_number(int(summary_stats.get('articles_total', 0)))}</div></div>
        <div class="card"><div class="label">Clustered Articles</div><div class="kpi">{_fmt_number(int(summary_stats.get('articles_clustered', 0)))}</div></div>
        <div class="card"><div class="label">Labeled Clusters</div><div class="kpi">{_fmt_number(int(summary_stats.get('clusters_labeled', 0)))}</div></div>
        <div class="card"><div class="label">Themes</div><div class="kpi">{_fmt_number(int(summary_stats.get('themes_count', 0)))}</div></div>
      </div>
    </section>

    <section>
      <h2>Interesting Tidbits</h2>
      <div class="tidbits">
        <div class="card"><strong>Most Persistent Theme</strong><p>{html.escape(str(tidbits.get('most_persistent_theme', 'N/A')))}</p></div>
        <div class="card"><strong>Fastest Growing Theme (DoD)</strong><p>{html.escape(str(tidbits.get('fastest_growing_theme', 'N/A')))}</p></div>
        <div class="card"><strong>Highest Density Cluster</strong><p>{html.escape(str(tidbits.get('highest_density_cluster', 'N/A')))}</p></div>
        <div class="card"><strong>Most Diverse Source Theme</strong><p>{html.escape(str(tidbits.get('most_diverse_source_theme', 'N/A')))}</p></div>
      </div>
    </section>

    <section>
      <h2>Theme Timeline (Top {int(themes_payload.get('top_k', 0))})</h2>
      {timeline_markup}
    </section>

    <section>
      <h2>Top Clusters</h2>
      <table>
        <thead>
          <tr><th>Date</th><th>Cluster ID</th><th>Density</th><th>Articles</th><th>Title</th><th>Summary</th></tr>
        </thead>
        <tbody>
          {''.join(cluster_rows) if cluster_rows else '<tr><td colspan="6">No cluster rows available.</td></tr>'}
        </tbody>
      </table>
    </section>

    <section>
      <h2>Theme Details</h2>
      <div class="theme-list">
        {''.join(theme_cards) if theme_cards else '<p>No themes available.</p>'}
      </div>
    </section>
  </main>
</body>
</html>
"""

    Path(output_path).write_text(html_doc, encoding="utf-8")
