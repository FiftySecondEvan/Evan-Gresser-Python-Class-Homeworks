from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"

INPUT_SOURCES = [
    (RAW_DIR / "billboard_rap_with_lyrics.csv", "billboard_rap_with_lyrics", 1),
    (RAW_DIR / "rap_top_songs_with_lyrics.csv", "rap_top_songs_with_lyrics", 2),
    (RAW_DIR / "hiphop_lyrics.csv", "hiphop_lyrics", 3),
]

OUT_ALL = RAW_DIR / "rap_expanded_with_lyrics.csv"
OUT_TARGET = RAW_DIR / "rap_expanded_10plus_1988_2024.csv"
OUT_REPORT = PROC_DIR / "rap_expanded_year_coverage_report.csv"
OUT_FILLED = RAW_DIR / "rap_expanded_coverage_filled_1970_1987_2025.csv"
OUT_FILLED_REPORT = PROC_DIR / "rap_expanded_coverage_filled_report.csv"

TARGET_START_YEAR = 1988
TARGET_END_YEAR = 2024
MIN_SONGS_PER_YEAR = 10
FILL_YEARS = list(range(1970, 1988)) + [2025]


def load_csv_with_fallback(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception:
            continue
    raise RuntimeError(f"Could not read {path} with utf-8/latin-1/cp1252")


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_cols = {c.lower().strip(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lower_cols:
            return lower_cols[candidate]
    return None


def extract_year(value: object) -> int | None:
    if pd.isna(value):
        return None
    match = re.search(r"(19[6-9]\d|20[0-2]\d)", str(value))
    if not match:
        return None
    return int(match.group(1))


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def standardize_source(df: pd.DataFrame, source_name: str, source_priority: int) -> pd.DataFrame:
    artist_col = find_column(df, ["artist", "artist_name", "performer", "rapper"])
    title_col = find_column(df, ["title", "song", "song_name", "track", "track_name", "name"])
    lyrics_col = find_column(df, ["lyrics", "lyric", "text", "song_lyrics"])
    year_col = find_column(
        df,
        [
            "year",
            "release_year",
            "release_date",
            "date",
            "album_year",
            "album_date",
            "album_release_date",
        ],
    )

    if lyrics_col is None:
        raise ValueError(f"No lyrics column found for source: {source_name}")

    out = pd.DataFrame(
        {
            "artist": df[artist_col] if artist_col else "Unknown",
            "title": df[title_col] if title_col else "Untitled",
            "lyrics": df[lyrics_col],
            "year_raw": df[year_col] if year_col else pd.NA,
            "source": source_name,
            "source_priority": source_priority,
        }
    )

    out["release_year"] = out["year_raw"].apply(extract_year)
    out = out.dropna(subset=["lyrics", "release_year"]).copy()
    out["lyrics"] = out["lyrics"].astype(str)
    out = out[out["lyrics"].str.strip().ne("")].copy()
    out = out[out["release_year"].between(1970, 2025)].copy()

    out["artist_norm"] = out["artist"].apply(normalize_text)
    out["title_norm"] = out["title"].apply(normalize_text)

    out = out.drop_duplicates(subset=["artist_norm", "title_norm", "release_year"], keep="first")
    return out


def build_datasets() -> None:
    frames: list[pd.DataFrame] = []

    for file_path, source_name, source_priority in INPUT_SOURCES:
        if not file_path.exists():
            print(f"Skipping missing source: {file_path}")
            continue
        raw_df = load_csv_with_fallback(file_path)
        std_df = standardize_source(raw_df, source_name, source_priority)
        frames.append(std_df)
        print(f"Loaded {source_name}: {len(std_df):,} rows")

    if not frames:
        raise RuntimeError("No valid sources were loaded.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(
        by=["source_priority", "release_year", "artist_norm", "title_norm"]
    ).copy()

    combined = combined.drop_duplicates(
        subset=["artist_norm", "title_norm", "release_year"], keep="first"
    ).copy()

    combined["rank"] = pd.NA
    combined["year"] = combined["release_year"].astype(int)

    all_export = combined[
        ["rank", "artist", "title", "year", "source", "lyrics", "release_year"]
    ].sort_values(["year", "artist", "title"]).reset_index(drop=True)

    all_export["artist_norm"] = all_export["artist"].apply(normalize_text)
    all_export["title_norm"] = all_export["title"].apply(normalize_text)
    all_export["is_backfill"] = False
    all_export["backfill_method"] = ""
    all_export["lyrics_available"] = all_export["lyrics"].astype(str).str.strip().ne("")

    coverage = (
        all_export.groupby("year", as_index=False)
        .size()
        .rename(columns={"size": "song_count"})
        .sort_values("year")
        .reset_index(drop=True)
    )

    target_mask = all_export["year"].between(TARGET_START_YEAR, TARGET_END_YEAR)
    target = all_export[target_mask].copy()
    target_counts = target.groupby("year").size()

    missing_years = [
        y
        for y in range(TARGET_START_YEAR, TARGET_END_YEAR + 1)
        if int(target_counts.get(y, 0)) < MIN_SONGS_PER_YEAR
    ]

    OUT_ALL.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    all_export.to_csv(OUT_ALL, index=False)
    target.to_csv(OUT_TARGET, index=False)

    report = coverage.copy()
    report["meets_min_10"] = report["song_count"] >= MIN_SONGS_PER_YEAR
    report.to_csv(OUT_REPORT, index=False)

    rap_artists = set(all_export["artist_norm"].dropna().tolist())
    filled = build_filled_dataset(all_export.copy(), rap_artists)

    filled_export = filled[
        [
            "rank",
            "artist",
            "title",
            "year",
            "source",
            "lyrics",
            "release_year",
            "is_backfill",
            "backfill_method",
            "lyrics_available",
        ]
    ].sort_values(["year", "is_backfill", "artist", "title"]).reset_index(drop=True)

    filled_export.to_csv(OUT_FILLED, index=False)

    filled_report = (
        filled_export.groupby("year", as_index=False)
        .agg(
            song_count_total=("title", "size"),
            song_count_with_lyrics=("lyrics_available", "sum"),
            backfill_count=("is_backfill", "sum"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )
    filled_report["meets_min_10_total"] = filled_report["song_count_total"] >= MIN_SONGS_PER_YEAR
    filled_report["meets_min_10_with_lyrics"] = (
        filled_report["song_count_with_lyrics"] >= MIN_SONGS_PER_YEAR
    )
    filled_report.to_csv(OUT_FILLED_REPORT, index=False)

    print("\nSaved files:")
    print(f"- {OUT_ALL}")
    print(f"- {OUT_TARGET}")
    print(f"- {OUT_REPORT}")
    print(f"- {OUT_FILLED}")
    print(f"- {OUT_FILLED_REPORT}")

    print("\nCoverage summary (all years present in merged dataset):")
    print(report.to_string(index=False))

    print(
        f"\nTarget range {TARGET_START_YEAR}-{TARGET_END_YEAR}: "
        f"{target['year'].nunique()} years, {len(target):,} songs"
    )
    if missing_years:
        print(
            "Years below 10 songs in target range: "
            + ", ".join(str(y) for y in missing_years)
        )
    else:
        print("All years in target range meet the 10-song minimum.")

    print("\nRequested fill years summary:")
    fill_summary = filled_report[filled_report["year"].isin(FILL_YEARS)].copy()
    print(fill_summary.to_string(index=False))


def build_filled_dataset(base: pd.DataFrame, rap_artists: set[str]) -> pd.DataFrame:
    hot100_path = RAW_DIR / "billboard_hot100_lyrics.csv"
    hotrap_path = RAW_DIR / "billboard_hot_rap_songs.csv"

    known_keys = set(zip(base["artist_norm"], base["title_norm"], base["year"]))
    out = base.copy()

    hot100 = load_csv_with_fallback(hot100_path) if hot100_path.exists() else pd.DataFrame()
    hotrap = load_csv_with_fallback(hotrap_path) if hotrap_path.exists() else pd.DataFrame()

    for year in FILL_YEARS:
        current_count = int((out["year"] == year).sum())
        needed = max(0, MIN_SONGS_PER_YEAR - current_count)
        if needed == 0:
            continue

        additions: list[pd.DataFrame] = []

        if not hot100.empty:
            c = pd.DataFrame(
                {
                    "rank": hot100.get("Rank", pd.NA),
                    "artist": hot100.get("Artist", "Unknown"),
                    "title": hot100.get("Song", "Untitled"),
                    "year": pd.to_numeric(hot100.get("Year", pd.NA), errors="coerce"),
                    "source": "billboard_hot100_backfill",
                    "lyrics": hot100.get("Lyrics", ""),
                }
            )
            c = c[c["year"] == year].copy()
            c = c.dropna(subset=["artist", "title"])
            c["artist_norm"] = c["artist"].apply(normalize_text)
            c["title_norm"] = c["title"].apply(normalize_text)
            c = c[c["artist_norm"].str.len() > 0]
            c = c[c["title_norm"].str.len() > 0]
            c["lyrics"] = c["lyrics"].fillna("").astype(str)
            c = c[c["lyrics"].str.strip().ne("")]
            c = c[~c.apply(lambda r: (r["artist_norm"], r["title_norm"], int(r["year"])) in known_keys, axis=1)]

            c["release_year"] = c["year"].astype(int)
            c["is_backfill"] = True
            c["lyrics_available"] = True
            c["is_rap_artist_match"] = c["artist_norm"].isin(rap_artists)
            c["backfill_method"] = c["is_rap_artist_match"].map(
                {True: "hot100_rap_artist_match", False: "hot100_proxy_nonrap"}
            )
            c = c.sort_values(["is_rap_artist_match", "rank"], ascending=[False, True])
            take = c.head(needed).copy()
            if not take.empty:
                additions.append(take)
                for row in take.itertuples(index=False):
                    known_keys.add((row.artist_norm, row.title_norm, int(row.year)))
                needed -= len(take)

        if needed > 0 and not hotrap.empty:
            c2 = pd.DataFrame(
                {
                    "rank": hotrap.get("rank", pd.NA),
                    "artist": hotrap.get("artist", "Unknown"),
                    "title": hotrap.get("title", "Untitled"),
                    "year": pd.to_numeric(hotrap.get("year", pd.NA), errors="coerce"),
                    "source": "billboard_hot_rap_chart_backfill",
                    "lyrics": "",
                }
            )
            c2 = c2[c2["year"] == year].copy()
            c2["artist_norm"] = c2["artist"].apply(normalize_text)
            c2["title_norm"] = c2["title"].apply(normalize_text)
            c2 = c2[c2["artist_norm"].str.len() > 0]
            c2 = c2[c2["title_norm"].str.len() > 0]
            c2 = c2[~c2.apply(lambda r: (r["artist_norm"], r["title_norm"], int(r["year"])) in known_keys, axis=1)]

            c2["release_year"] = c2["year"].astype(int)
            c2["is_backfill"] = True
            c2["lyrics_available"] = False
            c2["backfill_method"] = "hot_rap_chart_metadata_no_lyrics"
            take2 = c2.head(needed).copy()
            if not take2.empty:
                additions.append(take2)
                for row in take2.itertuples(index=False):
                    known_keys.add((row.artist_norm, row.title_norm, int(row.year)))

        if additions:
            out = pd.concat([out] + additions, ignore_index=True)

    return out


if __name__ == "__main__":
    build_datasets()
