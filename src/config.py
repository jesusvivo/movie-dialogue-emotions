"""Project-wide constants. No logic, no I/O."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TWEETS_CSV = DATA_DIR / "tweet_emotions.csv"
MOVIE_LINES_PATH = DATA_DIR / "cornell movie-dialogs corpus" / "movie_lines.txt"

# Training
RANDOM_STATE = 4  # preserved from the original notebook
TEST_SIZE = 0.3
MAX_TFIDF_FEATURES = 5000
STOP_WORDS = "english"

# Movie-dialogue analysis
FIGHT_CLUB_MOVIE_ID = "m348"
FIGHT_CLUB_CHARACTERS = ("TYLER", "MARLA", "JACK")
N_TIMELINE_BUCKETS = 3
TOP_N_EMOTIONS_IN_PIE = 6
