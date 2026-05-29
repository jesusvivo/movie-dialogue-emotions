"""End-to-end orchestration.

Usage:
    python -m src.pipeline                # train and analyze
    python -m src.pipeline --train        # train only (persists artifacts)
    python -m src.pipeline --analyze      # analyze only (loads artifacts)
"""
from __future__ import annotations

import argparse
import logging

from src import config
from src.analysis import analyze_character, plot_character_arc
from src.data import get_character_lines, load_movie_lines, load_tweets
from src.model import load_all, persist, train

logger = logging.getLogger(__name__)


def run_train() -> None:
    logger.info("Loading tweets from %s", config.TWEETS_CSV)
    X, y = load_tweets()
    logger.info("Training classifier on %d tweets...", len(X))
    _, label_encoder, classifier, report = train(X, y)
    print(report)
    persist(label_encoder, classifier)
    logger.info("Artifacts saved to %s", config.MODELS_DIR)


def run_analyze(
    movie_id: str = config.FIGHT_CLUB_MOVIE_ID,
    characters: tuple[str, ...] = config.FIGHT_CLUB_CHARACTERS,
) -> None:
    logger.info("Loading model artifacts from %s", config.MODELS_DIR)
    embedder, label_encoder, classifier = load_all()

    logger.info("Loading movie lines from %s", config.MOVIE_LINES_PATH)
    lines_df = load_movie_lines()

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for char in characters:
        lines = get_character_lines(lines_df, movie_id, char)
        if lines.empty:
            logger.warning("No lines for %s in %s — skipping", char, movie_id)
            continue
        logger.info("Analyzing %s (%d lines)...", char, len(lines))
        df = analyze_character(char, lines, embedder, classifier, label_encoder)
        out = config.FIGURES_DIR / f"{char.lower()}_arc.png"
        plot_character_arc(char, df, out)
        logger.info("Wrote %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tweet-trained emotion classifier applied to film dialogue."
    )
    parser.add_argument("--train", action="store_true", help="Train classifier only.")
    parser.add_argument("--analyze", action="store_true", help="Analyze movie characters only.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    run_train_flag = args.train or not (args.train or args.analyze)
    run_analyze_flag = args.analyze or not (args.train or args.analyze)

    if run_train_flag:
        run_train()
    if run_analyze_flag:
        run_analyze()


if __name__ == "__main__":
    main()
