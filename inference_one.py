# inference_one.py — predict a single image
# Called via inference_one.run() from client.py
# or run directly: python inference_one.py

from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image

import config


def load_model(device):
    ckpt  = torch.load(config.CKPT_PATH, map_location=device)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model, ckpt


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def predict_image(image_path: Path, model, transform, device) -> dict:
    img    = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    top_idx = int(np.argmax(probs))
    return {
        "predicted_class": config.CLASS_NAMES[top_idx],
        "confidence"     : float(probs[top_idx]),
        "all_probs"      : {cls: float(p)
                            for cls, p in zip(config.CLASS_NAMES, probs)},
    }


def run():
    if config.BASE_PATH is None:
        raise RuntimeError("Paths not set. Run client.py first.")

    if not config.CKPT_PATH.exists():
        print(f"\n  No saved model found at {config.CKPT_PATH}")
        print("  Train the model first (option 2 in the main menu).")
        return

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(device)
    transform   = get_transform()

    print(f"\n  Loading best saved checkpoint  "
          f"(epoch {ckpt['epoch']},  val acc {ckpt['val_acc']:.4f})")
    print(f"  Running on {device}")
    print()

    while True:
        raw = input("  Image path (or q to quit): ").strip()
        if raw.lower() == "q":
            break

        img_path = Path(raw)
        if not img_path.exists():
            print(f"  File not found: {img_path}\n")
            continue
        if img_path.suffix.lower() not in config.IMG_EXTS:
            print(f"  Unsupported format: {img_path.suffix}\n")
            continue

        try:
            result = predict_image(img_path, model, transform, device)
        except Exception as e:
            print(f"  Error processing image: {e}\n")
            continue

        pred  = result["predicted_class"]
        conf  = result["confidence"] * 100
        probs = result["all_probs"]

        print(f"\n  {img_path.name}")
        print(f"  Prediction  : {pred}")
        print(f"  Confidence  : {conf:.2f}%")
        print()
        print("  Class probabilities:")
        for cls, prob in sorted(probs.items(), key=lambda x: -x[1]):
            pct    = prob * 100
            marker = "  <" if cls == pred else ""
            print(f"    {cls:>5s}  {pct:6.2f}%{marker}")
        print()


if __name__ == "__main__":
    if config.BASE_PATH is None:
        path = input("Enter dataset base directory: ").strip()
        config.set_paths(Path(path))
    run()