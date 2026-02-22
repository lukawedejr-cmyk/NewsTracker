# CSV-Only News Clustering and Theme Timeline

This project is a GitHub-safe demo of a news-clustering workflow.

## What It Does
- Loads `example_articles.csv` (articles + precomputed embeddings).
- Cleans and deduplicates records (parses timestamps, validates embedding shape).
- Clusters articles by day using UMAP + HDBSCAN.
- Labels only the top 50 clusters per day (by density) with OpenAI (`title`, `summary`, relevant article IDs).
- Builds cross-day themes from those labeled clusters by merging similar cluster embeddings over time.
- Generates theme-level titles and summaries from each theme's cluster titles/summaries.
- Exports machine-readable artifacts and a local HTML report.

## Project Structure
- `run_pipeline.py`: CLI entrypoint.
- `pipeline/config.py`: env/config loading.
- `pipeline/io.py`: CSV loading and embedding parsing.
- `pipeline/clustering.py`: daily clustering + cluster metrics.
- `pipeline/labeling.py`: OpenAI/mock labeling and cluster embedding.
- `pipeline/theming.py`: cross-day theme assignment + relevancy scoring.
- `pipeline/report.py`: static HTML report generation.

## How To View Outputs
- HTML report:
  - macOS: `open outputs/report.html`
  - Linux: `xdg-open outputs/report.html`
  - Windows (PowerShell): `start outputs/report.html`

## Setup
1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
4. Fill `.env` with your OpenAI key:
   - `OPENAI_API_KEY` (required unless using `--mock-openai`)
   - Optional model overrides:
     - `OPENAI_CHAT_MODEL` (default: `gpt-4o-mini`)
     - `OPENAI_EMBED_MODEL` (default: `text-embedding-3-small`)

## Run
Primary command:
```bash
python run_pipeline.py --input example_articles.csv --output-dir outputs --top-k 20
```

Optional flags:
- `--start-date YYYY-MM-DD`
- `--end-date YYYY-MM-DD`
- `--theme-sim-threshold 0.15`
- `--theme-lookback-days 0` (`0` = full history, `1` = previous day only)
- `--top-k 20`
- `--label-top-per-day 50` (use `0` to label all clusters)
- `--report-scope daily`
- `--include-example-outputs`
- `--env-file /path/to/.env`
- `--mock-openai` (deterministic local fallback; no API calls)

## Input CSV Schema
Required columns:
- `id`
- `title`
- `published`
- `embedding`

Optional columns (used if present):
- `summary`
- `link`
- `source`
- `fetched_at`
- `cluster_id`

## Outputs
The pipeline writes:
- `outputs/articles_clustered.csv`
- `outputs/clusters_labeled.csv`
- `outputs/themes.json`
- `outputs/summary_stats.json`
- `outputs/report.html`

The files in `outputs/` committed in this repo are example outputs so they can be viewed immediately.

For your own use case, you should hook up your own article dataset and embeddings:
- Provide your own input CSV with the required schema (`id`, `title`, `published`, `embedding`).
- If your articles are not embedded yet, generate embeddings first and populate the `embedding` column.
- Re-run the pipeline to generate outputs tailored to your data.

## Notes
- This repository no longer uses Postgres, pgvector, Azure Functions, or Azure-specific auth.
- OpenAI is the only external API dependency for labeling and theme-title generation.
- For reproducible local/offline demos, use `--mock-openai`.
