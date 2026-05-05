from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import lyricsgenius
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"

HOT_RAP_PATH = RAW_DIR / "billboard_hot_rap_songs.csv"
HOT100_PATH = RAW_DIR / "billboard_hot100_lyrics.csv"

LYRICS_SOURCES = [
    (RAW_DIR / "billboard_rap_with_lyrics.csv", "billboard_rap_with_lyrics", 1),
    (RAW_DIR / "rap_top_songs_with_lyrics.csv", "rap_top_songs_with_lyrics", 2),
    (RAW_DIR / "hiphop_lyrics.csv", "hiphop_lyrics", 3),
    (RAW_DIR / "billboard_hot100_lyrics.csv", "billboard_hot100_lyrics", 4),
]

OUT_PATH = RAW_DIR / "billboard_rap_50_source_ladder.csv"
OUT_REPORT_PATH = PROC_DIR / "billboard_rap_50_source_ladder_report.csv"

DEFAULT_START_YEAR = 1989
DEFAULT_END_YEAR = 2025
TARGET_PER_YEAR = 50
GENIUS_DELAY = 1.0

RAP_ARTIST_MANUAL_ADDITIONS = {
    "2 chainz",
    "2pac",
    "50 cent",
    "beastie boys",
    "big sean",
    "black eyed peas",
    "b.o.b",
    "bow wow",
    "busta rhymes",
    "cash out",
    "chingy",
    "chris brown",
    "common",
    "drake",
    "e-40",
    "eminem",
    "eve",
    "fabolous",
    "field mob",
    "flo rida",
    "future",
    "ginuwine",
    "heavy d",
    "ice cube",
    "j. cole",
    "ja rule",
    "jadakiss",
    "jay-z",
    "jibbs",
    "juelz santana",
    "kanye west",
    "kendrick lamar",
    "kirko bangz",
    "lil jon",
    "lil kim",
    "lil wayne",
    "ll cool j",
    "ludacris",
    "marky mark and the funky bunch",
    "missy elliott",
    "nas",
    "nelly",
    "nicki minaj",
    "naughty by nature",
    "outkast",
    "pitbull",
    "public enemy",
    "run-dmc",
    "salt-n-pepa",
    "salt n pepa",
    "sean paul",
    "snoop dogg",
    "sugarhill gang",
    "t-pain",
    "the 2 live crew",
    "the roots",
    "t.i.",
    "t.i",
    "tyga",
    "vanilla ice",
    "wiz khalifa",
    "wu-tang clan",
    "young jeezy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--skip-genius", action="store_true")
    return parser.parse_args()


def load_csv_with_fallback(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception:
            continue
    raise RuntimeError(f"Could not read {path}")


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_cols = {str(col).lower().strip(): col for col in df.columns}
    for candidate in candidates:
        if candidate in lower_cols:
            return lower_cols[candidate]
    return None


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().strip()
    text = text.replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text


def split_artist_tokens(artist: str) -> list[str]:
    artist = normalize_text(artist)
    parts = re.split(r"\s+(?:featuring|feat\.?|ft\.?|with|and|x)\s+|\s*&\s*|\s*/\s*|\s*,\s*", artist)
    return [part.strip() for part in parts if part.strip()]


def title_cleanup(title: str) -> str:
    title = normalize_text(title)
    title = re.sub(r"\s*\((?:remix|radio edit|album version|explicit|clean)[^)]*\)", "", title)
    title = re.sub(r"\s*\[(?:remix|radio edit|album version|explicit|clean)[^]]*\]", "", title)
    return title.strip()


def standardize_lyrics_source(df: pd.DataFrame, source_priority: int) -> pd.DataFrame:
    artist_col = find_column(df, ["artist", "artist_name", "performer", "rapper"])
    title_col = find_column(df, ["title", "song", "song_name", "track", "track_name", "name"])
    lyrics_col = find_column(df, ["lyrics", "lyric", "text", "song_lyrics"])
    year_col = find_column(df, ["year", "release_year", "release_date", "date", "album_year"])

    if lyrics_col is None:
        return pd.DataFrame(columns=["artist", "title", "lyrics", "year", "artist_norm", "title_norm", "source_priority"])

    out = pd.DataFrame(
        {
            "artist": df[artist_col] if artist_col else "Unknown",
            "title": df[title_col] if title_col else "Untitled",
            "lyrics": df[lyrics_col],
            "year": pd.to_numeric(df[year_col], errors="coerce") if year_col else pd.NA,
        }
    )
    out = out.dropna(subset=["lyrics", "year"]).copy()
    out["lyrics"] = out["lyrics"].astype(str)
    out = out[out["lyrics"].str.strip().ne("")].copy()
    out["year"] = out["year"].astype(int)
    out["artist_norm"] = out["artist"].apply(normalize_text)
    out["title_norm"] = out["title"].apply(title_cleanup)
    out["source_priority"] = source_priority
    out = out.drop_duplicates(subset=["artist_norm", "title_norm", "year"], keep="first")
    return out


def build_rap_artist_lexicon() -> set[str]:
    artists: set[str] = set(RAP_ARTIST_MANUAL_ADDITIONS)

    for path, _, _ in LYRICS_SOURCES[:-1]:
        if not path.exists():
            continue
        df = load_csv_with_fallback(path)
        artist_col = find_column(df, ["artist", "artist_name", "performer", "rapper"])
        if artist_col is None:
            continue
        for artist in df[artist_col].dropna().astype(str):
            normalized = normalize_text(artist)
            if normalized:
                artists.add(normalized)
            artists.update(split_artist_tokens(artist))

    if HOT_RAP_PATH.exists():
        hotrap = load_csv_with_fallback(HOT_RAP_PATH)
        for artist in hotrap.get("artist", pd.Series(dtype=object)).dropna().astype(str):
            normalized = normalize_text(artist)
            if normalized:
                artists.add(normalized)
            artists.update(split_artist_tokens(artist))

    return {artist for artist in artists if artist}


def artist_is_rap(artist: str, rap_artists: set[str]) -> bool:
    normalized = normalize_text(artist)
    if normalized in rap_artists:
        return True
    return any(token in rap_artists for token in split_artist_tokens(artist))


def build_lyrics_library() -> tuple[dict[tuple[str, str, int], str], pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path, _, source_priority in LYRICS_SOURCES:
        if not path.exists():
            continue
        frames.append(standardize_lyrics_source(load_csv_with_fallback(path), source_priority))

    if not frames:
        raise RuntimeError(
            "No lyric source files were found. Expected at least one of: "
            + ", ".join(str(path) for path, _, _ in LYRICS_SOURCES)
        )

    library_df = pd.concat(frames, ignore_index=True)
    library_df = library_df.sort_values(["source_priority", "year", "artist_norm", "title_norm"])
    library_df = library_df.drop_duplicates(subset=["artist_norm", "title_norm", "year"], keep="first")
    library = {
        (row.artist_norm, row.title_norm, int(row.year)): row.lyrics
        for row in library_df.itertuples(index=False)
    }
    return library, library_df


def load_hot_rap_chart() -> pd.DataFrame:
    if not HOT_RAP_PATH.exists():
        raise FileNotFoundError(f"Missing required chart input: {HOT_RAP_PATH}")
    hotrap = load_csv_with_fallback(HOT_RAP_PATH)
    out = pd.DataFrame(
        {
            "year": pd.to_numeric(hotrap.get("year", pd.NA), errors="coerce"),
            "rank": pd.to_numeric(hotrap.get("rank", pd.NA), errors="coerce"),
            "artist": hotrap.get("artist", "Unknown"),
            "title": hotrap.get("title", "Untitled"),
            "ranking_source": "billboard_hot_rap_songs",
        }
    )
    out = out.dropna(subset=["year", "rank"]).copy()
    out["year"] = out["year"].astype(int)
    out["rank"] = out["rank"].astype(int)
    out["artist_norm"] = out["artist"].apply(normalize_text)
    out["title_norm"] = out["title"].apply(title_cleanup)
    out = out.drop_duplicates(subset=["artist_norm", "title_norm", "year"], keep="first")
    return out


def load_hot100_rap_candidates(rap_artists: set[str]) -> pd.DataFrame:
    if not HOT100_PATH.exists():
        raise FileNotFoundError(f"Missing required chart input: {HOT100_PATH}")
    hot100 = load_csv_with_fallback(HOT100_PATH)
    out = pd.DataFrame(
        {
            "year": pd.to_numeric(hot100.get("Year", pd.NA), errors="coerce"),
            "rank": pd.to_numeric(hot100.get("Rank", pd.NA), errors="coerce"),
            "artist": hot100.get("Artist", "Unknown"),
            "title": hot100.get("Song", "Untitled"),
            "ranking_source": "billboard_year_end_hot100_rap_artist",
        }
    )
    out = out.dropna(subset=["year", "rank"]).copy()
    out["year"] = out["year"].astype(int)
    out["rank"] = out["rank"].astype(int)
    out["artist_norm"] = out["artist"].apply(normalize_text)
    out["title_norm"] = out["title"].apply(title_cleanup)
    out = out[out["artist"].apply(lambda value: artist_is_rap(value, rap_artists))].copy()
    out = out.drop_duplicates(subset=["artist_norm", "title_norm", "year"], keep="first")
    return out


def load_curated_fallback(library_df: pd.DataFrame) -> pd.DataFrame:
    out = library_df[["artist", "title", "year", "artist_norm", "title_norm"]].copy()
    out["rank"] = pd.NA
    out["ranking_source"] = "curated_rap_lyrics_fallback"
    out = out.drop_duplicates(subset=["artist_norm", "title_norm", "year"], keep="first")
    return out


def select_top_50_for_year(
    year: int,
    hotrap_df: pd.DataFrame,
    hot100_df: pd.DataFrame,
    curated_df: pd.DataFrame,
) -> pd.DataFrame:
    selected: list[dict[str, object]] = []
    seen_keys: set[tuple[str, str, int]] = set()

    def add_rows(rows: pd.DataFrame, tier: int) -> None:
        for row in rows.itertuples(index=False):
            key = (row.artist_norm, row.title_norm, int(row.year))
            if key in seen_keys:
                continue
            selected.append(
                {
                    "year": int(row.year),
                    "artist": row.artist,
                    "title": row.title,
                    "artist_norm": row.artist_norm,
                    "title_norm": row.title_norm,
                    "source_rank": row.rank,
                    "ranking_source": row.ranking_source,
                    "ranking_tier": tier,
                }
            )
            seen_keys.add(key)
            if len(selected) >= TARGET_PER_YEAR:
                break

    add_rows(hotrap_df[hotrap_df["year"] == year].sort_values("rank"), 1)
    if len(selected) < TARGET_PER_YEAR:
        add_rows(hot100_df[hot100_df["year"] == year].sort_values("rank"), 2)
    if len(selected) < TARGET_PER_YEAR:
        fallback_rows = curated_df[curated_df["year"] == year].sort_values(["artist_norm", "title_norm"])
        add_rows(fallback_rows, 3)

    out = pd.DataFrame(selected)
    out["overall_rank"] = range(1, len(out) + 1)
    return out


def attach_existing_lyrics(selected_df: pd.DataFrame, lyrics_library: dict[tuple[str, str, int], str]) -> pd.DataFrame:
    out = selected_df.copy()
    out["lyrics"] = out.apply(
        lambda row: lyrics_library.get((row["artist_norm"], row["title_norm"], int(row["year"]))),
        axis=1,
    )
    out["lyrics_available"] = out["lyrics"].fillna("").astype(str).str.strip().ne("")
    return out


def build_genius_client(token: str) -> lyricsgenius.Genius:
    genius = lyricsgenius.Genius(token)
    genius.timeout = 10
    genius.retries = 2
    genius.sleep_time = GENIUS_DELAY
    genius.remove_section_headers = False
    genius.skip_non_songs = True
    genius.excluded_terms = ["(Remix)", "(Live)", "(Demo)"]
    genius.verbose = False
    return genius


def fetch_genius_lyrics(genius: lyricsgenius.Genius, artist: str, title: str) -> str | None:
    try:
        song = genius.search_song(title, artist)
        time.sleep(GENIUS_DELAY)
        if song and song.lyrics:
            return re.sub(r"\d+Embed$", "", song.lyrics).strip()
    except Exception as exc:
        print(f"[warn] Genius fetch failed for {artist} - {title}: {exc}")
        time.sleep(GENIUS_DELAY)
    return None


def backfill_missing_lyrics(selected_df: pd.DataFrame, skip_genius: bool) -> pd.DataFrame:
    load_dotenv(ROOT / ".env")
    token = os.environ.get("GENIUS_TOKEN", "")
    if skip_genius or not token:
        return selected_df

    genius = build_genius_client(token)
    out = selected_df.copy()
    missing_idx = out.index[~out["lyrics_available"]].tolist()

    for idx in missing_idx:
        artist = out.at[idx, "artist"]
        title = out.at[idx, "title"]
        print(f"Fetching Genius lyrics for {out.at[idx, 'year']} | {artist} - {title}")
        lyrics = fetch_genius_lyrics(genius, artist, title)
        if lyrics:
            out.at[idx, "lyrics"] = lyrics
            out.at[idx, "lyrics_available"] = True

    return out


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("start-year must be <= end-year")

    rap_artists = build_rap_artist_lexicon()
    lyrics_library, library_df = build_lyrics_library()
    hotrap_df = load_hot_rap_chart()
    hot100_df = load_hot100_rap_candidates(rap_artists)
    curated_df = load_curated_fallback(library_df)

    yearly_frames: list[pd.DataFrame] = []
    for year in range(args.start_year, args.end_year + 1):
        yearly = select_top_50_for_year(year, hotrap_df, hot100_df, curated_df)
        yearly_frames.append(yearly)
        print(
            f"{year}: selected {len(yearly):2d} songs "
            f"(hot rap={(yearly['ranking_tier'] == 1).sum()}, "
            f"hot100={(yearly['ranking_tier'] == 2).sum()}, "
            f"curated={(yearly['ranking_tier'] == 3).sum()})"
        )

    selected_df = pd.concat(yearly_frames, ignore_index=True)
    selected_df = attach_existing_lyrics(selected_df, lyrics_library)
    selected_df = backfill_missing_lyrics(selected_df, skip_genius=args.skip_genius)

    selected_df = selected_df[
        [
            "year",
            "overall_rank",
            "source_rank",
            "artist",
            "title",
            "ranking_source",
            "ranking_tier",
            "lyrics",
            "lyrics_available",
        ]
    ].sort_values(["year", "overall_rank"]).reset_index(drop=True)

    report = (
        selected_df.groupby("year", as_index=False)
        .agg(
            song_count=("title", "size"),
            lyrics_count=("lyrics_available", "sum"),
            hot_rap_source_count=("ranking_tier", lambda values: int((values == 1).sum())),
            hot100_source_count=("ranking_tier", lambda values: int((values == 2).sum())),
            curated_source_count=("ranking_tier", lambda values: int((values == 3).sum())),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(OUT_PATH, index=False)
    report.to_csv(OUT_REPORT_PATH, index=False)

    print(f"\nSaved dataset -> {OUT_PATH}")
    print(f"Saved report  -> {OUT_REPORT_PATH}")
    print("\nCoverage summary:")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
