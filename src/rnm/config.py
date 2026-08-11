from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]

TMP_DIR = PROJECT_ROOT / ".tmp"
TMP_DIR.mkdir(exist_ok=True)

STATIC_IMAGE_PATH = TMP_DIR / "alien.png"
if not STATIC_IMAGE_PATH.exists():
    raise FileNotFoundError(f"Static image not found at {STATIC_IMAGE_PATH}. Please ensure the image exists.")

DYNAMIC_IMAGE_PATH = TMP_DIR / "tmp.png"

TTS_MODEL_DIR = TMP_DIR / "tts_model"
TTS_MODEL_DIR.mkdir(exist_ok=True)

OUTPUT_IMAGE_DIR = TMP_DIR / "output"
OUTPUT_IMAGE_DIR.mkdir(exist_ok=True)

# Configuration flags
DEBUG = False
USE_REAL_CAMERA = False
AUDIO_OUTPUT = True

# --- Models that work with the current setup ---
# model = "qwen3.5:4b",
model = "granite4.1:3b"
# model = "qwen3:4b",
# model = "ornith:latest",
