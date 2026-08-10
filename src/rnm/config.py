from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
TMP_DIR = PROJECT_ROOT / ".tmp"
TMP_DIR.mkdir(exist_ok=True)
STATIC_IMAGE_PATH = TMP_DIR / "alien.png"
DYNAMIC_IMAGE_PATH = TMP_DIR / "tmp.png"
TTS_MODEL_DIR = TMP_DIR / "tts_model"
TTS_MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_IMAGE_DIR = TMP_DIR / "output"
OUTPUT_IMAGE_DIR.mkdir(exist_ok=True)
