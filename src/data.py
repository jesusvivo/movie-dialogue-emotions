"""Loaders for the tweet emotion dataset and the Cornell Movie Dialogs Corpus."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config

MOVIE_LINES_SEPARATOR = " +++$+++ "
MOVIE_LINES_COLUMNS = ["line_id", "character_id", "movie_id", "character_name", "line"]


def load_tweets(path: Path = config.TWEETS_CSV) -> tuple[pd.Series, pd.Series]:
    """Load the Kaggle 'Emotion Detection from Text' CSV.

    Returns (tweet_text, sentiment_label) as parallel Series.
    """
    df = pd.read_csv(path)
    return df["content"], df["sentiment"]


def load_movie_lines(path: Path = config.MOVIE_LINES_PATH) -> pd.DataFrame:
    """Parse `movie_lines.txt` from the Cornell Movie Dialogs Corpus.

    Each record has five ' +++$+++ '-separated fields. The encoding is
    iso-8859-1 — the corpus is not utf-8 clean.
    """
    rows = []
    with open(path, "r", encoding="iso-8859-1") as f:
        for raw in f:
            fields = raw.split(MOVIE_LINES_SEPARATOR)
            if len(fields) == len(MOVIE_LINES_COLUMNS):
                rows.append(fields)
    df = pd.DataFrame(rows, columns=MOVIE_LINES_COLUMNS)
    df["line"] = df["line"].str.rstrip("\n")
    return df


def get_character_lines(lines_df: pd.DataFrame, movie_id: str, character: str) -> pd.Series:
    """All dialogue lines spoken by `character` in `movie_id`."""
    mask = (lines_df["movie_id"] == movie_id) & (lines_df["character_name"] == character)
    return lines_df.loc[mask, "line"]
