import logging
from pathlib import Path

import cv2
from PIL import Image

from rnm.dataSpecs import FilePathInput
from rnm.paths import DYNAMIC_IMAGE_PATH, STATIC_IMAGE_PATH, TMP_DIR

logger = logging.getLogger(__name__)

# Input Sectioin

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
    TMP_DIR.mkdir(exist_ok=True)
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

    return FilePathInput(path=Path(image_path))


# Output Section
def display_image(image_path: FilePathInput):
    """
    Display an image to the user by decoding the base64 string and showing it using PIL.
    """
    img = Image.open(image_path.path)
    img.show()
