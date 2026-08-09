from collections.abc import Sequence

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, CLIPModel, CLIPProcessor

from rnm.paths import STATIC_IMAGE_PATH

device = "cuda" if torch.cuda.is_available() else "cpu"

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID, device_map="auto")


def classify_face(image_input: str | Image.Image) -> dict:
    labels = [
        "a photograph of a real human face",
        "a cartoon or alien face, animated, illustrated",
        "not a face",
    ]
    if isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        image = Image.open(image_input).convert("RGB")

    inputs = clip_processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=-1).squeeze(0)
    idx = int(probs.argmax())
    return {"label": labels[idx], "confidence": float(probs[idx])}


def classify_faces(image_input: str | Image.Image) -> dict:
    labels = [
        "a photograph of a real human face",
        "a cartoon or alien face, animated, illustrated",
        "not a face",
    ]
    if isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        image = Image.open(image_input).convert("RGB")

    inputs = clip_processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=-1).squeeze(0)
    idx = int(probs.argmax())
    return {"label": labels[idx], "confidence": float(probs[idx])}


def is_alien_face(face_dict: dict) -> bool:
    normalized = face_dict["label"].lower()
    return "alien" in normalized or "cartoon" in normalized


def determine_forehead_point(box: Sequence[float]) -> tuple[int, int]:
    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0
    center_x = x0 + width * 0.5
    forehead_y = y0 + height * 0.2
    return round(center_x), round(forehead_y)


def draw_cross(
    draw: ImageDraw.ImageDraw,
    point: tuple[int, int],
    size: int = 15,
    color: str = "red",
    width: int = 2,
) -> None:
    x, y = point
    draw.line([(x - size, y - size), (x + size, y + size)], fill=color, width=width)
    r = 4
    draw.ellipse((x - r, y - r, x + r, y + r), outline="black", width=width, fill="red")
    r = 10
    draw.ellipse((x - r, y - r, x + r, y + r), outline="black", width=width)
    r = 18
    draw.ellipse((x - r, y - r, x + r, y + r), outline="black", width=width)
    draw.line([(x + size, y - size), (x - size, y + size)], fill=color, width=width)


def main():

    image = Image.open(STATIC_IMAGE_PATH).convert("RGB")
    inputs = dino_processor(images=image, text=[["face"]], return_tensors="pt").to(dino_model.device)
    with torch.no_grad():
        outputs = dino_model(**inputs)

    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.3,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]],
    )
    result = results[0]
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arialbd.ttf", 24)

    def draw_text(position, text, color):
        draw.text((position[0], position[1] - 10), text, fill=color, font=font, anchor="ms")

    for box, _score, _label_text in zip(result["boxes"], result["scores"], result["text_labels"], strict=True):
        box = [round(x, 2) for x in box.tolist()]
        crop = image.crop((box[0], box[1], box[2], box[3]))
        face_dict = classify_face(crop)

        if is_alien_face(face_dict):
            draw.rectangle(box, outline="red", width=3)
            draw_text(((box[0] + box[2]) / 2, box[1]), f"alien:({face_dict['confidence']:.2f})", color="red")
            forehead_point = determine_forehead_point(box)
            draw_cross(draw, forehead_point)
        else:
            draw.rectangle(box, outline="green", width=3)
            draw_text(((box[0] + box[2]) / 2, box[1]), f"human:({face_dict['confidence']:.2f})", color="green")

    image.show()
