from pathlib import Path

# paths.py lives at src/rnm/paths.py, so the repo root is three levels up.
PROJECT_ROOT = Path(__file__).parents[2]
TMP_DIR = PROJECT_ROOT / ".tmp"
STATIC_IMAGE_PATH = TMP_DIR / "alien.png"
DYNAMIC_IMAGE_PATH = TMP_DIR / "tmp.png"
