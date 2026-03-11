"""
Download Ghanaian banknote images for the training dataset.
Uses icrawler (Python 3.14 compatible — no imghdr dependency).

Usage:
    python download_images.py
"""

import os
import sys
import subprocess

# ── Install icrawler if missing ───────────────────────────────────────────────
try:
    from icrawler.builtin import BingImageCrawler
except ImportError:
    print("Installing icrawler ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "icrawler"])
    from icrawler.builtin import BingImageCrawler

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR      = os.path.join(BASE_DIR, "dataset")
IMAGES_PER_CLASS = 100     # 100 images per denomination = 800 total

# Specific queries → Bank of Ghana + GH₵ symbol = more accurate results
DENOMINATIONS = {
    "1_GHS":   "Bank of Ghana GH₵1 one cedi banknote currency note",
    "2_GHS":   "Bank of Ghana GH₵2 two cedis banknote currency note",
    "5_GHS":   "Bank of Ghana GH₵5 five cedis banknote currency note",
    "10_GHS":  "Bank of Ghana GH₵10 ten cedis banknote currency note",
    "20_GHS":  "Bank of Ghana GH₵20 twenty cedis banknote currency note",
    "50_GHS":  "Bank of Ghana GH₵50 fifty cedis banknote currency note",
    "100_GHS": "Bank of Ghana GH₵100 hundred cedis banknote currency note",
    "200_GHS": "Bank of Ghana GH₵200 two hundred cedis banknote currency note",
}

# ── Download ──────────────────────────────────────────────────────────────────
print("=" * 55)
print("  Ghana Currency Dataset Downloader  (icrawler)")
print("=" * 55)
print(f"Saving to : {DATASET_DIR}")
print(f"Images    : {IMAGES_PER_CLASS} per denomination")
print()

for folder, query in DENOMINATIONS.items():
    save_dir = os.path.join(DATASET_DIR, folder)
    os.makedirs(save_dir, exist_ok=True)

    # Count existing images
    existing = len([
        f for f in os.listdir(save_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ])

    if existing >= IMAGES_PER_CLASS:
        print(f"[SKIP] {folder}  already has {existing} images.")
        continue

    needed = IMAGES_PER_CLASS - existing
    print(f"[DOWNLOADING] {folder}  ({query})  — need {needed} more images ...")

    try:
        crawler = BingImageCrawler(
            storage={"root_dir": save_dir},
            downloader_threads=4,
            parser_threads=2,
        )
        crawler.crawl(
            keyword=query,
            max_num=needed,
            min_size=(100, 100),   # skip tiny/broken images
            overwrite=False,
        )

        after = len([
            f for f in os.listdir(save_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ])
        print(f"         -> {after} images now in dataset/{folder}/\n")

    except Exception as exc:
        print(f"  [ERROR] {exc}")
        print(f"  Add images to dataset/{folder}/ manually.\n")

print("=" * 55)
print("Download complete!")
print()
print("Next — train the model:")
print("  python backend/model/train.py")
print("=" * 55)
