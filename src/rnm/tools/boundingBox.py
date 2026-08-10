from collections.abc import Sequence
from uuid import uuid4

import torch
from langchain.tools import tool
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, CLIPModel, CLIPProcessor

from rnm.config import OUTPUT_IMAGE_DIR, STATIC_IMAGE_PATH
from rnm.tools.audio_io import TTS, laser

device = "cuda" if torch.cuda.is_available() else "cpu"

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID).to(device)

LABELS = [
    "a photograph of a real human face",
    "a cartoon or alien face, animated, illustrated",
    "not a face",
]


def classify_faces(images: list[Image.Image], labels: list[str]) -> dict:
    inputs = clip_processor(text=labels, images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=-1)
    idx = probs.argmax(dim=-1)
    return {"label": [labels[i] for i in idx], "confidence": [float(probs[i, j]) for i, j in enumerate(idx)]}


def is_alien_face(label: str) -> bool:
    normalized = label.lower()
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


@tool(
    "find_aliens_and_shoot",
    description="Detects alien faces in the given image (absolute path) and shoot them. A single function that does both detection and shooting together. Returns the path to the image with bounding boxes drawn around detected faces.",
)
def main(image_path: str) -> str:
    TTS.play("Initializing... Scanning for alien faces")
    image = Image.open(image_path).convert("RGB")
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
    faces = classify_faces(
        [image.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3]))) for box in result["boxes"]], LABELS
    )

    def draw_text(position, text, color):
        draw.text((position[0], position[1] - 10), text, fill=color, font=font, anchor="ms")

    count = {
        "alien": 0,
        "human": 0,
    }
    for box, _score, _label_text, label, confidence in zip(
        result["boxes"], result["scores"], result["text_labels"], faces["label"], faces["confidence"], strict=True
    ):
        box = [round(x, 2) for x in box.tolist()]
        if is_alien_face(label):
            draw.rectangle(box, outline="red", width=3)
            draw_text(((box[0] + box[2]) / 2, box[1]), f"alien:({confidence:.2f})", color="red")
            forehead_point = determine_forehead_point(box)
            draw_cross(draw, forehead_point)
            count["alien"] += 1
        else:
            draw.rectangle(box, outline="green", width=3)
            draw_text(((box[0] + box[2]) / 2, box[1]), f"human:({confidence:.2f})", color="green")
            count["human"] += 1

    output_path = OUTPUT_IMAGE_DIR / f"{uuid4()}.png"
    image.save(output_path)
    laser(count=count["alien"])
    return f"{count['alien']} aliens shot, {count['human']} humans identified, proof:{str(output_path)}"


if __name__ == "__main__":
    main(STATIC_IMAGE_PATH)
