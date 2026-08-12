from collections.abc import Sequence
from uuid import uuid4

import torch
from langchain.tools import tool
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, CLIPModel, CLIPProcessor

import rnm.config as CONFIG
from rnm.tools.audio_io import TTS, laser

device = "cuda" if torch.cuda.is_available() else "cpu"

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID).to(device)


def classify_faces(images: list[Image.Image], labels: list[str]) -> dict:
    inputs = clip_processor(text=labels, images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=-1)
    idx = probs.argmax(dim=-1)
    return {"label": [labels[i] for i in idx], "confidence": [float(probs[i, j]) for i, j in enumerate(idx)]}


def is_alien_face(label: str) -> bool:
    return "alien" in label.lower()


def is_human_face(label: str) -> bool:
    return "human" in label.lower()


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


class Box:
    def __init__(self, box: list):
        self.box = [float(x) for x in box]
        self.left = self.box[0]
        self.up = self.box[1]
        self.right = self.box[2]
        self.down = self.box[3]
        assert (self.left <= self.right) and (self.up <= self.down), self
        self.normalize()
        self.area = (self.right - self.left) * (self.down - self.up)
        assert self.area > 0, self

    def __repr__(self):
        return f"<Left:{self.left}, Up:{self.up}, Right:{self.right}, Down:{self.down}>"

    def __call__(self):
        return [self.left, self.up, self.right, self.down]

    def normalize(self):
        self.left = round(self.left)
        self.right = round(self.right)
        self.up = round(self.up)
        self.down = round(self.down)


class Overlap:
    def __init__(self, boxes: list):
        self.boxes: list[Box] = [Box(box) for box in boxes]

    def __call__(self, threshold: float = 0.4):
        self.merge(threshold)
        return [i() for i in self.boxes]

    def intersection(self, A: Box, B: Box) -> float:
        """
        Determine if A overlaps B.
        [Left, Up, Right, Down]

        Arguments:
            A: Box to check
            B: Other Box

        Return:
            Intersection fraction of shared area (0 = No overlap, 1 = Full Overlap)
        """
        hIn = (A.left < B.right) and (A.right > B.left)
        vIn = (A.up < B.down) and (A.down > B.up)

        if not (vIn and hIn):
            return 0

        Xs = sorted([A.left, A.right, B.left, B.right])
        Ys = sorted([A.up, A.down, B.up, B.down])

        iArea = (Xs[2] - Xs[1]) * (Ys[2] - Ys[1])
        # totalArea = A.area+B.area-iArea  # If a big box encapsulates multiple small ones, this area can prevent it (IOU)
        totalArea = min(A.area, B.area)
        ratio = iArea / totalArea
        return ratio

    def combine(self, A: Box, B: Box) -> Box:
        combined = Box([min(A.left, B.left), min(A.up, B.up), max(A.right, B.right), max(A.down, B.down)])
        return combined

    def merge(self, threshold: float):
        """
        Comare all boxes to all boxes recursively
        """
        if len(self.boxes) < 2:
            return
        while len(self.boxes):
            changed = False
            for i in range(len(self.boxes) - 1):
                a = self.boxes[i]
                for j in range(i + 1, len(self.boxes)):
                    b = self.boxes[j]
                    if self.intersection(a, b) >= threshold:
                        changed = True
                        c = self.combine(a, b)
                        self.boxes.pop(j)
                        self.boxes[i] = c
                        break
                if changed:
                    break
            if not changed:
                break


@tool(
    "find_aliens_and_shoot",
    description="Detects alien faces in the given image (absolute path) and shoot them. A single function that does both detection and shooting together. Returns the path to the image with bounding boxes drawn around detected faces.",
)
def algorithm(image_path: str) -> str:
    TTS.play("Initializing... Scanning for alien faces")
    image = Image.open(image_path).convert("RGB")

    # Extract Faces
    inputs = dino_processor(images=image, text=[["face"]], return_tensors="pt").to(dino_model.device)
    with torch.no_grad():
        outputs = dino_model(**inputs)
    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.2,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]],
    )
    result = results[0]
    if len(result["boxes"]) == 0:
        raise Exception("No faces found")

    # Asign labels
    faces = classify_faces(
        [image.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3]))) for box in result["boxes"]],
        CONFIG.CLASSIFICATION_LABELS,
    )

    # Seperate into categories
    objects = {"alien": [], "human": [], "other": []}
    for box, _score, _label_text, label, _confidence in zip(
        result["boxes"], result["scores"], result["text_labels"], faces["label"], faces["confidence"], strict=True
    ):
        if is_alien_face(label):
            objects["alien"].append(box)
        elif is_human_face(label):
            objects["human"].append(box)
        else:
            objects["other"].append(box)

    # Merge boxes
    objects["alien"] = Overlap(objects["alien"])(CONFIG.MERGE_THRESHOLD)
    objects["human"] = Overlap(objects["human"])(CONFIG.MERGE_THRESHOLD)
    objects["other"] = Overlap(objects["other"])(CONFIG.MERGE_THRESHOLD)

    # Draw the image with aim
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arialbd.ttf", 24)

    def draw_text(position, text, color):
        draw.text((position[0], position[1] - 10), text, fill=color, font=font, anchor="ms")

    for label, boxes in objects.items():
        for box in boxes:
            if label == "other":
                continue
            elif label == "alien":
                color = "red"
                forehead_point = determine_forehead_point(box)
                draw_cross(draw, forehead_point)
            elif label == "human":
                color = "green"
            else:
                raise ValueError(f"{label} is not a valid object.")

            draw.rectangle(box, outline=color, width=3)
            draw_text(((box[0] + box[2]) / 2, box[1]), label, color=color)

    output_path = CONFIG.OUTPUT_IMAGE_DIR / f"{uuid4()}.png"
    image.save(output_path)
    image.show()
    print(f"Output image saved at: {output_path}")
    laser(len(objects))
    return f"{len(objects['alien'])} aliens shot, {len(objects['human'])} humans identified, proof:{str(output_path)}"


def main():
    # Disable algorithm tool decorator to do the testing
    CONFIG.AUDIO_OUTPUT = False
    algorithm(str(CONFIG.TMP_DIR / "alien1.png"))
    algorithm(str(CONFIG.TMP_DIR / "alien2.jpg"))
    algorithm(str(CONFIG.TMP_DIR / "alien3.jpg"))
    algorithm(str(CONFIG.TMP_DIR / "alien4.jpg"))
