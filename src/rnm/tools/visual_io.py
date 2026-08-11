import cv2
from langchain.tools import tool
from PIL import Image

import rnm.config as C


def static_camera() -> str:
    """
    Simulate a camera, and return the path to a pre-defined static image.
    """
    return str(C.STATIC_IMAGE_PATH)


def dynamic_camera() -> str:
    """
    Open the camera and capture a live image, then save it in the tmp directory.
    """
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    if not ret:
        raise RuntimeError("Failed to capture image from camera.")
    cam.release()
    C.TMP_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(C.DYNAMIC_IMAGE_PATH), frame)
    return str(C.DYNAMIC_IMAGE_PATH)


@tool(
    "camera",
    description="Use a camera tool to capture an image and return its path. Display the image to the user after capturing it.",
)
def camera() -> str:
    """
    Use a camera tool to capture an image and return its path.
    """
    if C.USE_REAL_CAMERA:
        print("Using real camera feed.")
        image_path = dynamic_camera()
    else:
        print("Using static camera feed.")
        image_path = static_camera()
    print(f"Image captured at: {image_path}")
    return f"Image captured at: {image_path}. Now display the image to the user using the display tool."


@tool(
    "display_image",
    description="Display an image to the user. Alwasys display the image if the previous tool returns an image path.",
)
def display_image(image_path: str) -> str:
    """
    Display an image to the user.
    """
    try:
        img = Image.open(image_path)
        img.show()
        return "Image displayed."
    except Exception as e:
        return f"Failed to display image: {e}"
