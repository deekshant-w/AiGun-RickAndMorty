import logging
from pathlib import Path

import cv2
from PIL import Image

from src.rnm.dataSpecs import FilePathInput

logger = logging.getLogger(__name__)

# Input Sectioin

PROJECT_ROOT = Path(__file__).parent.parent
STATIC_IMAGE_PATH = PROJECT_ROOT / ".tmp" / "alien.png"
DYNAMIC_IMAGE_PATH = PROJECT_ROOT / ".tmp" / "tmp.png"
USE_REAL_CAMERA = True


def static_camera() -> str:
    """
    Simulate a camera, and return the path to a pre-defined static image.
    """
    return str(STATIC_IMAGE_PATH)


def dynamic_camera() -> str:
    """
    Open the camera and capture a live image, then save it in the tmp directory.
    """
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    if not ret:
        raise RuntimeError("Failed to capture image from camera.")
    cam.release()
    cv2.imwrite(str(DYNAMIC_IMAGE_PATH), frame)
    return str(DYNAMIC_IMAGE_PATH)


def camera() -> FilePathInput:
    """
    Use a camera tool to capture an image and return its path.
    """
    if USE_REAL_CAMERA:
        logger.info("Using real camera feed.")
        image_path = dynamic_camera()
    else:
        logger.info("Using static camera feed.")
        image_path = static_camera()

    return FilePathInput(path=image_path)


# Output Section
def display_image(image_path: FilePathInput):
    """
    Display an image to the user by decoding the base64 string and showing it using PIL.
    """
    img = Image.open(image_path.path)
    img.show()
