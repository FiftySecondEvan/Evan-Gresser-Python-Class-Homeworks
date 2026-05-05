# Final Project: Rap Game

This folder contains my final project for the Python class.

Chronological lyrical analysis of charting rap songs from 1989 to 2024.

This project builds a source-ladder dataset of top rap songs by year, computes lyric-level metrics (vocabulary, rhyme, profanity, pronouns, and thematic rates), and visualizes long-run trends in a Jupyter notebook.


## GitHub Upload Note

Large raw and processed CSV datasets are not included in this GitHub upload because they exceed GitHub's standard file size limits. Regenerate them locally by running the build script and notebook workflow documented below.

## Project Structure

- `scripts/build_billboard_rap_50.py`: builds the core source-ladder dataset (50 songs per year)
- `scripts/build_expanded_rap_source.py`: auxiliary expanded-source builder
- `rap_lyrics_chronological_analysis.ipynb`: main end-to-end analysis notebook
- `data/raw/`: raw inputs and generated source-ladder CSV
- `data/processed/`: processed exports from notebooks/scripts
- `requirements.txt`: Python dependencies

## Requirements

- Python 3.10+
- A virtual environment is recommended
- Optional: `GENIUS_TOKEN` in `.env` for lyric backfill via Genius API

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build the Source-Ladder Dataset

Run from project root:

```bash
.venv/bin/python -u scripts/build_billboard_rap_50.py --skip-genius
```

Optional flags:

```bash
.venv/bin/python -u scripts/build_billboard_rap_50.py --skip-genius --start-year 1989 --end-year 1992
```

If you want lyric backfill from Genius, remove `--skip-genius` and set `GENIUS_TOKEN` in `.env`.

### Generated Files

Primary outputs from `build_billboard_rap_50.py`:

- `data/raw/billboard_rap_50_source_ladder.csv`
- `data/processed/billboard_rap_50_source_ladder_report.csv`

## Run the Main Analysis Notebook

Open and run all cells in:

- `rap_lyrics_chronological_analysis.ipynb`

The notebook reads `data/raw/billboard_rap_50_source_ladder.csv`, computes per-song and time-series metrics, and exports:

- `data/processed/rap_lyrics_cleaned.csv`
- `data/processed/rap_lyrics_decade_summary.csv`

## Notes on Data Confidence

- Notebook analysis covers 1989 to 2024 (2025 excluded due to incomplete coverage).
- The 1989 to 1993 word-frequency block is included but explicitly labeled lower confidence due to sparse and imbalanced early-year data.

## Reproducibility Checklist

1. Install dependencies from `requirements.txt`.
2. Build the source-ladder CSV with `scripts/build_billboard_rap_50.py`.
3. Run all cells in `rap_lyrics_chronological_analysis.ipynb`.
4. Verify exports appear in `data/processed/`.
