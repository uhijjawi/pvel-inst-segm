# Description: Convert COCO annotations to text annotations for MMPretrain classification models.
"""
Convert COCO annotations to text annotations like the following (image path, category id) pairs:
folder_1/xxx.png 0
folder_1/xxy.png 1
123.png 4
nsdf3.png 3
...
"""

# IMPORTS
import argparse
from pathlib import Path
import json
from pycocotools.coco import COCO

# FUNCTIONS
def parse_args():
    parser = argparse.ArgumentParser(description="Convert COCO annotations to text annotations.")
    parser.add_argument(
        "--coco-ann-path",
        type=str,
        required=True,
        help="The path to COCO annotations.",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        required=True,
        help="The output path of text annotations.",
    )
    return parser.parse_args()

def coco_to_text_ann(coco_ann_path, out_path):
    # Convert only unique image_id, category_id pairs
    coco = COCO(coco_ann_path)
    ann_ids = coco.getAnnIds()
    anns = coco.loadAnns(ann_ids)
    img_ids = set()
    img_cat_ids = []
    for ann in anns:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        if img_id not in img_ids:
            img_ids.add(img_id)
            img_cat_ids.append((img_id, cat_id))
    with open(out_path, "w") as f:
        for img_id, cat_id in img_cat_ids:
            try:
                img_info = coco.loadImgs(img_id)
            except KeyError:
                print(f"Image id {img_id} not found in COCO annotations. Skipping...")
                continue
            img_path = img_info[0]["file_name"]
            f.write(f"{img_path} {cat_id}\n")
    print(f"Text annotations saved to {out_path}")

# MAIN
if __name__ == "__main__":
    args = parse_args()
    coco_to_text_ann(args.coco_ann_path, args.out_path)