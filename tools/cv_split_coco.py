# Copyright (c) OpenMMLab. All rights reserved. Modified by @abdol in 2024.
import argparse
import os.path as osp
from pathlib import Path

import numpy as np
from mmengine.fileio import dump, load
from mmengine.utils import mkdir_or_exist
from sklearn.model_selection import KFold

prog_description = """K-Fold coco split.

To split coco data with K-Fold cross validation:
    python tools/cv_split_coco.py --ann_path [PATH TO ANN FILE] --out-dir [PATH TO EXPORTED ANNS] --fold [NUMBER OF FOLDS]
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ann-path",
        type=str,
        help="The path to the coco annotation file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="The output directory of coco annotations.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        help="The number of folds for K-Fold cross validation.",
        default=5,
    )
    args = parser.parse_args()
    return args


def split_coco(ann_path: Path, output_path, folds):
    """Split COCO data for K-Fold cross validation.

    Args:
        ann_path (str): The data root of coco dataset.
        out_path (str): The output directory of coco annotations.
        folds (int): The fold of dataset and set as random seed for data split.
    """

    def save_anns(path, images, annotations):
        sub_anns = dict()
        sub_anns["images"] = images
        sub_anns["annotations"] = annotations
        if "licenses" in anns:
            sub_anns["licenses"] = anns["licenses"]
        sub_anns["categories"] = anns["categories"]
        if "info" in anns:
            sub_anns["info"] = anns["info"]

        # extract file name from path
        out_dir = path.parent
        mkdir_or_exist(out_dir)
        dump(sub_anns, path)

    # set random seed with the fold
    np.random.seed(folds)
    anns = load(ann_path)

    image_list = anns["images"]
    labeled_total = len(image_list)
    # Implement K-Fold cross validation using SKLearn.model_selection.KFold
    kf = KFold(n_splits=folds, shuffle=True)
    labeled_indices = np.arange(labeled_total)
    for i, (train_index, val_index) in enumerate(kf.split(labeled_indices)):
        print(f"Fold {i + 1}/{folds}...")
        train_split_path = Path("train") / f"{i + 1}.json"
        val_split_path = Path("val") / f"{i + 1}.json"
        labeled_train = [image_list[i] for i in train_index]
        labeled_val = [image_list[i] for i in val_index]
        save_anns(output_path / train_split_path, labeled_train, anns["annotations"])
        save_anns(output_path / val_split_path, labeled_val, anns["annotations"])


if __name__ == "__main__":
    args = parse_args()
    split_coco(args.ann_path, args.output_path, args.folds)
