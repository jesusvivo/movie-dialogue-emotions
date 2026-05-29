# Movie Dialogue Emotions

Train an emotion classifier on Twitter sentiment data, then point it at film dialogue and chart how a character's emotional state shifts across the runtime. The demo is Fight Club (Tyler / Marla / Jack) — but the pipeline accepts any movie ID and character list from the [Cornell Movie Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html).

Originally a UNIMI Information Retrieval final project; refactored here into a clean modular pipeline as a portfolio piece.

## Character arcs

After running the pipeline, `reports/figures/` contains a three-panel emotional arc per character:

| Character | Arc |
| --- | --- |
| Tyler Durden | ![Tyler arc](reports/figures/tyler_arc.png) |
| Marla Singer | ![Marla arc](reports/figures/marla_arc.png) |
| Jack | ![Jack arc](reports/figures/jack_arc.png) |

## Approach

- **Sentence embeddings.** Each tweet (and each line of movie dialogue at inference time) is embedded into a 384-dim vector via `sentence-transformers/all-MiniLM-L6-v2`. This replaced an earlier TF-IDF baseline (see "What changed from the original" below).
- **Classifier.** Logistic regression on the embeddings. Trained on ~28 k tweets from the Kaggle [Emotion Detection from Text](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text) dataset (40 k total, 70/30 split, 13 sentiment classes — heavily imbalanced).
- **Per-character analysis.** Filter the Cornell corpus to a movie ID, pull each character's lines, embed them with the same model, predict an emotion per line.
- **Three-act arc.** Split each character's lines into three contiguous timeline buckets and render a top-emotions pie per act. The original notebook hardcoded the bucket boundaries per character; the refactor generalises to `N_TIMELINE_BUCKETS` (configurable).
- **Persisted artifacts.** Label encoder and classifier are joblib-dumped to `models/` so `--analyze` works without retraining. The sentence-transformer itself is *not* pickled — its name is stored in `config.EMBEDDING_MODEL_NAME` and it reloads from the local Hugging Face cache.

## Results on held-out tweets

| Model | Accuracy | Weighted F1 | Macro F1 |
| --- | --- | --- | --- |
| TF-IDF (5 k) + LinearSVC (original baseline) | 0.31 | 0.29 | 0.18 |
| all-MiniLM-L6-v2 + LogisticRegression (current) | **0.37** | **0.33** | 0.17 |

Big classes (`neutral`, `worry`, `love`, `happiness`) gain ~5–8 F1 points each from the upgrade. Macro F1 stays flat because the four smallest classes (`anger`, `boredom`, `empty`, `enthusiasm`, each with <300 samples in a 40 k corpus) collapse to zero recall under both models — embeddings don't fix severe class imbalance.

## Known limitations

- **Small classes still unrecoverable.** Without targeted sampling or a different loss, the imbalanced minorities don't survive. `class_weight="balanced"` was tested and made things worse (it overcorrects when one class has ~75× the support of another).
- **Tweet-trained / dialogue-applied.** Domain shift between tweets and movie dialogue is real. The arcs are interpretable and entertaining, not clinical.

## What changed from the original notebook

The first refactor mechanically extracted the original TF-IDF + LinearSVC pipeline into modules. The current model layer replaces it with sentence-transformers + LR. Architecture-wise the rest of the pipeline (data loading, per-character analysis, the three-act bucketing, persistence) is unchanged — only the embedding step and the classifier swapped. The original `.toarray()` densification bug that motivated the first cleanup is moot now: ST embeddings are inherently dense and small (384 dims).

## Get the data

The two source datasets are not redistributed in this repo. Download and place them as follows:

```
data/
├── tweet_emotions.csv                          # Kaggle: pashupatigupta/emotion-detection-from-text
└── cornell movie-dialogs corpus/               # unzip the corpus as-is, keep the original folder name
    └── movie_lines.txt
```

## How to run

Prerequisites: Python 3.13. The sentence-transformer model (`all-MiniLM-L6-v2`, ~90 MB) downloads on first use and caches under `~/.cache/huggingface/`. Apple Silicon Macs auto-use MPS; CUDA GPUs auto-detect too.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.pipeline             # train + analyze (default, ~30s on MPS)
python -m src.pipeline --train     # train only — persists to models/
python -m src.pipeline --analyze   # analyze only — loads models/, writes reports/figures/

pytest                              # run the unit tests
```

## Repo

```
movie-dialogue-emotions/
├── data/                  # raw inputs (gitignored)
├── models/                # joblib artifacts (gitignored)
├── reports/figures/       # character arc images (tracked)
├── src/
│   ├── config.py          # paths, hyperparams, movie ID & character list
│   ├── data.py            # tweet loader + Cornell movie_lines parser
│   ├── model.py           # sentence-transformer embedder + LR; train / persist / load / predict
│   ├── analysis.py        # split_into_thirds + analyze_character + plot_character_arc
│   └── pipeline.py        # train(), analyze(), CLI
├── notebooks/
│   └── exploratory.ipynb  # class distribution + sample predictions
├── tests/
│   ├── test_data.py       # movie_lines parser + character filter
│   └── test_analysis.py   # timeline bucketing
└── requirements.txt
```
