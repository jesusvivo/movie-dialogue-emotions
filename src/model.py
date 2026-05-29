"""Embedder + classifier factories, training, persistence, inference.

The model fleet:
- `all-MiniLM-L6-v2` (sentence-transformers) for 384-dim sentence embeddings.
- `LogisticRegression(class_weight="balanced")` on top, so the 13 imbalanced
  tweet classes don't collapse to majority predictions like the original
  LinearSVC did.

The sentence-transformer model is NOT joblib-pickled. We persist its name in
`config.EMBEDDING_MODEL_NAME` and reload it on demand; the cached weights live
under `~/.cache/huggingface/`. Only the label encoder and the LR classifier go
to disk via joblib.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src import config

logger = logging.getLogger(__name__)

LABEL_ENCODER_PATH = config.MODELS_DIR / "label_encoder.joblib"
CLASSIFIER_PATH = config.MODELS_DIR / "classifier.joblib"


def build_embedder() -> SentenceTransformer:
    """Load the sentence-transformer; auto-uses MPS / CUDA when available."""
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)


def build_label_encoder() -> LabelEncoder:
    return LabelEncoder()


def build_classifier() -> LogisticRegression:
    return LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE)


def _embed(embedder: SentenceTransformer, texts: Iterable[str]) -> np.ndarray:
    return embedder.encode(
        list(texts),
        batch_size=config.EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def train(X: pd.Series, y: pd.Series) -> tuple[SentenceTransformer, LabelEncoder, LogisticRegression, str]:
    """Embed train + test, fit a class-balanced LR, return the held-out report."""
    X_train_text, X_test_text, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    label_encoder = build_label_encoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    embedder = build_embedder()
    logger.info("Embedding %d training tweets...", len(X_train_text))
    X_train_emb = _embed(embedder, X_train_text)
    logger.info("Embedding %d test tweets...", len(X_test_text))
    X_test_emb = _embed(embedder, X_test_text)

    classifier = build_classifier()
    classifier.fit(X_train_emb, y_train)

    y_pred = classifier.predict(X_test_emb)
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
    )
    return embedder, label_encoder, classifier, report


def persist(
    label_encoder: LabelEncoder,
    classifier: LogisticRegression,
    models_dir: Path = config.MODELS_DIR,
) -> None:
    """Persist only the label encoder and classifier. The embedder reloads from its name."""
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(label_encoder, models_dir / "label_encoder.joblib")
    joblib.dump(classifier, models_dir / "classifier.joblib")


def load_all(models_dir: Path = config.MODELS_DIR) -> tuple[SentenceTransformer, LabelEncoder, LogisticRegression]:
    embedder = build_embedder()
    label_encoder = joblib.load(models_dir / "label_encoder.joblib")
    classifier = joblib.load(models_dir / "classifier.joblib")
    return embedder, label_encoder, classifier


def predict_emotions(
    lines: Iterable[str],
    embedder: SentenceTransformer,
    classifier: LogisticRegression,
    label_encoder: LabelEncoder,
) -> list[str]:
    embeddings = _embed(embedder, lines)
    codes = classifier.predict(embeddings)
    return label_encoder.inverse_transform(codes).tolist()
