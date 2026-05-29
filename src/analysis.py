"""Per-character emotion analysis and figure generation."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.model import predict_emotions


def split_into_thirds(items: Sequence) -> list:
    """Partition `items` into N_TIMELINE_BUCKETS buckets of (nearly) equal size."""
    n = len(items)
    k = config.N_TIMELINE_BUCKETS
    boundaries = [round(i * n / k) for i in range(k + 1)]
    return [items[boundaries[i] : boundaries[i + 1]] for i in range(k)]


def analyze_character(
    name: str,
    lines: pd.Series,
    vectorizer,
    classifier,
    label_encoder,
) -> pd.DataFrame:
    """Predict an emotion for each line and assign each row a timeline bucket."""
    emotions = predict_emotions(lines, vectorizer, classifier, label_encoder)
    bucketed = split_into_thirds(list(range(len(emotions))))
    bucket = [0] * len(emotions)
    for i, indices in enumerate(bucketed):
        for j in indices:
            bucket[j] = i
    return pd.DataFrame({"character": name, "emotion": emotions, "bucket": bucket})


def plot_character_arc(character_name: str, emotions_df: pd.DataFrame, output_path: Path) -> None:
    """Draw 1xN pie charts: top emotions in each timeline bucket. Saves a PNG."""
    fig, axes = plt.subplots(1, config.N_TIMELINE_BUCKETS, figsize=(15, 5))
    if config.N_TIMELINE_BUCKETS == 1:
        axes = [axes]
    fig.suptitle(f"{character_name.title()} — emotional arc through the film", fontsize=14)
    for i, ax in enumerate(axes):
        bucket_df = emotions_df[emotions_df["bucket"] == i]
        if bucket_df.empty:
            ax.set_title(f"Act {i + 1} (no lines)")
            ax.axis("off")
            continue
        top = bucket_df["emotion"].value_counts().head(config.TOP_N_EMOTIONS_IN_PIE)
        top.plot.pie(ax=ax, autopct="%1.1f%%", textprops={"fontsize": 10})
        ax.set_ylabel("")
        ax.set_title(f"Act {i + 1}  ({len(bucket_df)} lines)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
