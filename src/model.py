"""Vectorizer + classifier factories, training, persistence, inference."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from src import config

logger = logging.getLogger(__name__)

VECTORIZER_PATH = config.MODELS_DIR / "vectorizer.joblib"
LABEL_ENCODER_PATH = config.MODELS_DIR / "label_encoder.joblib"
CLASSIFIER_PATH = config.MODELS_DIR / "classifier.joblib"


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        stop_words=config.STOP_WORDS,
        max_features=config.MAX_TFIDF_FEATURES,
    )


def build_label_encoder() -> LabelEncoder:
    return LabelEncoder()


def build_classifier() -> LinearSVC:
    return LinearSVC(random_state=config.RANDOM_STATE, max_iter=1000)


def train(X: pd.Series, y: pd.Series) -> tuple[TfidfVectorizer, LabelEncoder, LinearSVC, str]:
    """Fit vectorizer + encoder + classifier end-to-end.

    Internal train/test split (seeded) so the printed classification report
    reflects held-out performance. The TF-IDF matrix stays sparse throughout —
    `LinearSVC` accepts sparse natively, and densifying it (as the original
    notebook did via `.toarray()`) would materialize a 28k × 5000 array.
    """
    X_train_text, X_test_text, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    label_encoder = build_label_encoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    classifier = build_classifier()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
    )
    return vectorizer, label_encoder, classifier, report


def persist(
    vectorizer: TfidfVectorizer,
    label_encoder: LabelEncoder,
    classifier: LinearSVC,
    models_dir: Path = config.MODELS_DIR,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, models_dir / "vectorizer.joblib")
    joblib.dump(label_encoder, models_dir / "label_encoder.joblib")
    joblib.dump(classifier, models_dir / "classifier.joblib")


def load_all(models_dir: Path = config.MODELS_DIR) -> tuple[TfidfVectorizer, LabelEncoder, LinearSVC]:
    return (
        joblib.load(models_dir / "vectorizer.joblib"),
        joblib.load(models_dir / "label_encoder.joblib"),
        joblib.load(models_dir / "classifier.joblib"),
    )


def predict_emotions(
    lines: Iterable[str],
    vectorizer: TfidfVectorizer,
    classifier: LinearSVC,
    label_encoder: LabelEncoder,
) -> list[str]:
    X = vectorizer.transform(list(lines))
    codes = classifier.predict(X)
    return label_encoder.inverse_transform(codes).tolist()
