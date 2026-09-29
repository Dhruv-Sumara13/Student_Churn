"""Project assets resolve independently of the shell's working directory."""
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
ORIGINAL_DATA = APP_ROOT / 'data' / 'original' / 'latest.csv'
ORIGINAL_MODELS = APP_ROOT / 'models' / 'original'
