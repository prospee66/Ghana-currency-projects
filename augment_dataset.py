"""
Augment existing dataset images to expand each class to a target size.
Generates realistic variations: rotations, crops, brightness, blur, etc.

Run from project root:
    python augment_dataset.py
"""

import os
import random
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
TARGET_PER_CLASS = 300   # aim for 300 images per denomination
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def augment_image(img: Image.Image) -> Image.Image:
    """Apply a random combination of augmentations to one image."""
    img = img.convert("RGB")
    w, h = img.size

    # Random rotation (-25° to +25°)
    angle = random.uniform(-25, 25)
    img = img.rotate(angle, expand=False, fillcolor=(200, 200, 200))

    # Random crop (80–100% of each dimension)
    crop_frac = random.uniform(0.80, 1.0)
    new_w = int(w * crop_frac)
    new_h = int(h * crop_frac)
    left  = random.randint(0, w - new_w)
    top   = random.randint(0, h - new_h)
    img   = img.crop((left, top, left + new_w, top + new_h))
    img   = img.resize((w, h), Image.LANCZOS)

    # Random horizontal flip
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Brightness jitter (±30%)
    factor = random.uniform(0.7, 1.3)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # Contrast jitter (±25%)
    factor = random.uniform(0.75, 1.25)
    img = ImageEnhance.Contrast(img).enhance(factor)

    # Saturation jitter (±30%)
    factor = random.uniform(0.7, 1.3)
    img = ImageEnhance.Color(img).enhance(factor)

    # Slight sharpness variation
    factor = random.uniform(0.8, 1.5)
    img = ImageEnhance.Sharpness(img).enhance(factor)

    # Occasional mild blur (20% chance)
    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))

    # Perspective warp via slight resize + pad (simulates tilt)
    if random.random() < 0.3:
        scale = random.uniform(0.88, 0.98)
        small = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        bg = Image.new("RGB", (w, h), (210, 210, 210))
        offset_x = random.randint(0, w - small.width)
        offset_y = random.randint(0, h - small.height)
        bg.paste(small, (offset_x, offset_y))
        img = bg

    return img


def augment_class(class_dir: str, class_name: str):
    # Collect original images only (not previously augmented ones)
    originals = [
        f for f in os.listdir(class_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
        and not f.startswith("aug_")
    ]

    if not originals:
        print(f"  [{class_name}] No original images found, skipping.")
        return

    existing_total = len([
        f for f in os.listdir(class_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
    ])

    needed = TARGET_PER_CLASS - existing_total
    if needed <= 0:
        print(f"  [{class_name}] Already has {existing_total} images — skipping.")
        return

    print(f"  [{class_name}] {len(originals)} originals, {existing_total} total "
          f"-> generating {needed} augmented images ...")

    generated = 0
    idx = existing_total + 1

    while generated < needed:
        src_name = random.choice(originals)
        src_path = os.path.join(class_dir, src_name)
        try:
            img = Image.open(src_path)
            aug = augment_image(img)
            out_name = f"aug_{idx:05d}.jpg"
            aug.save(os.path.join(class_dir, out_name), "JPEG", quality=92)
            generated += 1
            idx += 1
        except Exception as e:
            print(f"    [WARN] Could not process {src_name}: {e}")

    print(f"  [{class_name}] Done — {existing_total + generated} total images")


def main():
    print("=" * 60)
    print("  Ghana Currency Dataset Augmentation")
    print(f"  Target: {TARGET_PER_CLASS} images per class")
    print("=" * 60)

    classes = sorted([
        d for d in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, d))
    ])

    if not classes:
        print(f"[ERROR] No class folders found in {DATASET_PATH}")
        return

    print(f"\nFound {len(classes)} classes: {classes}\n")

    for cls in classes:
        augment_class(os.path.join(DATASET_PATH, cls), cls)

    print("\n" + "=" * 60)
    print("  Augmentation complete!")
    print("  Now retrain:")
    print("    python backend/model/train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
