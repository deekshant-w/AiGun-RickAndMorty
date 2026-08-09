from collections.abc import Sequence

import clip
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)


def classify_face(image_input: str | Image.Image) -> str:

    # Labels — tune these prompts for better accuracy
    labels = [
        "a photograph of a real human face",
        "a cartoon or alien face, animated, illustrated",
        "not a face",
    ]
    if isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        image = Image.open(image_input).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)
    text = clip.tokenize(labels).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)

        # Cosine similarity
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).squeeze(0)

    probs = similarity.softmax(dim=0)
    idx = probs.argmax().item()
    return f"{labels[idx]}  ({probs[idx]:.2%} confidence)"


def is_alien_face(face_label: str) -> bool:
    normalized = face_label.lower()
    return "alien" in normalized or "cartoon" in normalized


def determine_forehead_point(box: Sequence[float]) -> tuple[int, int]:
    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0
    center_x = x0 + width * 0.5
    forehead_y = y0 + height * 0.18
    return round(center_x), round(forehead_y)


def draw_cross(
    draw: ImageDraw.ImageDraw,
    point: tuple[int, int],
    size: int = 10,
    color: str = "cyan",
    width: int = 2,
) -> None:
    x, y = point
    draw.line([(x - size, y), (x + size, y)], fill=color, width=width)
    draw.line([(x, y - size), (x, y + size)], fill=color, width=width)


model_id = "IDEA-Research/grounding-dino-tiny"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, device_map="auto")

image = Image.open("alien.png").convert("RGB")
text_labels = [["face"]]
flat_text_labels = [item for sublist in text_labels for item in sublist]

inputs = processor(images=image, text=text_labels, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.3,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]],
)

# Retrieve the first image result
result = results[0]

# Create a draw object
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

for box, score, labels in zip(result["boxes"], result["scores"], result["labels"], strict=True):
    box = [round(x, 2) for x in box.tolist()]
    label_text = labels
    if isinstance(labels, int):
        label_text = flat_text_labels[labels] if labels < len(flat_text_labels) else str(labels)
    print(f"Detected {label_text} with confidence {round(score.item(), 3)} at location {box}")

    # Crop the detected region and classify it with the CLIP face classifier
    crop = image.crop((box[0], box[1], box[2], box[3]))
    face_label = classify_face(crop)

    # Draw the bounding box
    draw.rectangle(box, outline="red", width=2)

    # For alien/cartoony faces, determine the forehead point and mark it with a cross
    if is_alien_face(face_label):
        forehead_point = determine_forehead_point(box)
        draw_cross(draw, forehead_point)

    # Add a solid background for the label to improve readability
    try:
        text_bbox = draw.textbbox((0, 0), face_label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    except AttributeError:
        text_width, text_height = font.getsize(face_label)
    label_x = box[0]
    label_y = max(box[1] - text_height - 6, 0)
    draw.rectangle(
        [
            (label_x - 1, label_y - 1),
            (label_x + text_width + 3, label_y + text_height + 3),
        ],
        fill="black",
    )
    draw.text((label_x + 1, label_y + 1), face_label, fill="white", font=font)

# Show the image
image.show()
