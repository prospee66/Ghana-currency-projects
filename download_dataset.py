"""
Download Ghana currency banknote images into the dataset folder.
Uses Bing image search via icrawler.

Run from the project root:
    python download_dataset.py
"""

import os
import time
from icrawler.builtin import BingImageCrawler

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

# Search queries per denomination — targeting current Bank of Ghana series (2019+)
# The current series features the Big Six independence heroes on the front.
# 200 GHS was introduced in 2019. Queries use "new series" / year to avoid old notes.
DENOMINATIONS = {
    "1_GHS":   [
        "Ghana new 1 cedi note 2019 Bank of Ghana current series",
        "GHS 1 cedi banknote Big Six Ghana current",
    ],
    "2_GHS":   [
        "Ghana new 2 cedis note 2019 Bank of Ghana current series",
        "GHS 2 cedis banknote Big Six Ghana current",
    ],
    "5_GHS":   [
        "Ghana new 5 cedis note 2019 Bank of Ghana current series",
        "GHS 5 cedis banknote Big Six Ghana current",
    ],
    "10_GHS":  [
        "Ghana new 10 cedis note 2019 Bank of Ghana current series",
        "GHS 10 cedis banknote Big Six Ghana current",
    ],
    "20_GHS":  [
        "Ghana new 20 cedis note 2019 Bank of Ghana current series",
        "GHS 20 cedis banknote Big Six Ghana current",
    ],
    "50_GHS":  [
        "Ghana new 50 cedis note 2019 Bank of Ghana current series",
        "GHS 50 cedis banknote Big Six Ghana current",
    ],
    "100_GHS": [
        "Ghana new 100 cedis note 2019 Bank of Ghana current series",
        "GHS 100 cedis banknote Big Six Ghana current",
    ],
    "200_GHS": [
        "Ghana 200 cedis note Bank of Ghana 2019 new denomination",
        "GHS 200 cedis banknote Ghana current series 200",
    ],
}

IMAGES_PER_QUERY = 60   # 2 queries × 60 = up to 120 images per class

def download_for_class(class_name, queries):
    save_dir = os.path.join(DATASET_PATH, class_name)
    os.makedirs(save_dir, exist_ok=True)

    # Count existing images so we don't re-download unnecessarily
    existing = len([f for f in os.listdir(save_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))])
    print(f"\n[{class_name}] Existing: {existing} images")

    for query in queries:
        print(f"  Searching: '{query}' ...")
        crawler = BingImageCrawler(
            storage={"root_dir": save_dir},
            downloader_threads=4,
            parser_threads=2,
        )
        try:
            crawler.crawl(
                keyword=query,
                max_num=IMAGES_PER_QUERY,
                min_size=(100, 100),   # skip tiny/corrupt images
                file_idx_offset="auto",
            )
        except Exception as e:
            print(f"  [WARN] Query failed: {e}")
        time.sleep(2)   # be polite to the search engine

    after = len([f for f in os.listdir(save_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))])
    print(f"  [{class_name}] Done — {after} total images (+{after - existing} new)")


def main():
    print("=" * 60)
    print("  Ghana Currency Dataset Downloader")
    print("=" * 60)
    print(f"  Saving to: {DATASET_PATH}\n")

    for class_name, queries in DENOMINATIONS.items():
        download_for_class(class_name, queries)
        time.sleep(3)

    print("\n" + "=" * 60)
    print("  Download complete!")
    print("  Now retrain the model:")
    print("    python backend/model/train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
