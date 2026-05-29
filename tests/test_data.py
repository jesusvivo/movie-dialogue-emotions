"""Unit tests for the Cornell movie-lines parser and the character filter."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data import (
    MOVIE_LINES_COLUMNS,
    MOVIE_LINES_SEPARATOR,
    get_character_lines,
    load_movie_lines,
)


def _make_sample_lines() -> str:
    """A minimal stand-in for movie_lines.txt: 3 rows, two movies, two characters."""
    rows = [
        ("L1", "u0", "m348", "TYLER", "I see, you wanna live."),
        ("L2", "u1", "m348", "MARLA", "I haven't been laid in a while."),
        ("L3", "u2", "m999", "OTHER", "I am in another movie."),
    ]
    return "\n".join(MOVIE_LINES_SEPARATOR.join(r) for r in rows) + "\n"


def test_load_movie_lines_parses_sample(tmp_path):
    sample = tmp_path / "movie_lines.txt"
    sample.write_text(_make_sample_lines(), encoding="iso-8859-1")

    df = load_movie_lines(sample)

    assert list(df.columns) == MOVIE_LINES_COLUMNS
    assert len(df) == 3
    # Trailing newline must be stripped from the line text.
    assert df.iloc[0]["line"] == "I see, you wanna live."
    assert df.iloc[1]["character_name"] == "MARLA"
    assert df.iloc[2]["movie_id"] == "m999"


def test_get_character_lines_filters_movie_and_character():
    df = pd.DataFrame(
        {
            "line_id": ["L1", "L2", "L3", "L4"],
            "character_id": ["u0", "u0", "u1", "u2"],
            "movie_id": ["m348", "m348", "m348", "m999"],
            "character_name": ["TYLER", "TYLER", "MARLA", "TYLER"],
            "line": ["a", "b", "c", "d"],
        }
    )

    tyler_in_fc = get_character_lines(df, "m348", "TYLER")
    assert tyler_in_fc.tolist() == ["a", "b"]

    # Same character, wrong movie.
    assert get_character_lines(df, "m999", "MARLA").tolist() == []
