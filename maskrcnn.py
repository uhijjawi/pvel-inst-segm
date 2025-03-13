# Instance Segmentation with MMDetection
# MODEL MAKER (not a config file)

# IMPORTS
from mmdet.utils import register_all_modules
from pycocotools.coco import COCO
import time
import datetime
import os
from mmengine.runner import set_random_seed
from mmengine import Config
from pathlib import Path
import os.path as osp


# CONFIG
## PATHS
TRAIN_MODE = False
USE_AMP = True
DATA_ROOT = Path("../data/")
OUTPUT_PATH = Path("output/")
MODEL_PATH = Path("runs/")
CONFIG_FOLDER = Path("config/")
INFERENCE_MODEL_PATH = "runs/<path-to-checkpoint-file>" 
TRAIN_DATA_FOLDER = "train/"
VAL_DATA_FOLDER = "test/"
TRAIN_ANN_FILENAME = "train_coco.json"
VAL_ANN_FILENAME = "test_coco.json"
TRAIN_DATA_PATH = DATA_ROOT / TRAIN_DATA_FOLDER
VAL_DATA_PATH = DATA_ROOT / VAL_DATA_FOLDER
TRAIN_ANN_PATH = TRAIN_DATA_PATH / TRAIN_ANN_FILENAME
VAL_ANN_PATH = VAL_DATA_PATH / VAL_ANN_FILENAME
MODEL_CONFIG = "mask-rcnn_x101-32x8d_fpn_1x_coco"
CHECKPOINT_FILE = (
    "mask_rcnn_x101_32x8d_fpn_1x_coco_20220630_173841-0aaf329e.pth"
)

## PARAMETERS
NUM_EPOCHS = 100
LEARNING_RATE = 0.0025
WEIGHT_DECAY = 0.0001
OPTIM_TYPE = "SGD"
NUM_WORKERS = 4
BATCH_SIZE = 6
EVAL_INTERVAL = 5
CHECKPOINT_INTERVAL = 5
SCALE_RATIO_RANGE = (0.2, 0.9)
F1_THRESHOLD = 0.25

# DIRECTORY SETUP
if osp.exists(OUTPUT_PATH):
    print(f"Output path {OUTPUT_PATH} already exists, deleting...")
    os.system(f"rm -rf {OUTPUT_PATH}")
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
timestamp = time.strftime("%Y%m%d_%H%M", time.localtime())
model_path = MODEL_PATH / timestamp
config_file = f"{CONFIG_FOLDER}/{MODEL_CONFIG}.py"
checkpoint_file = f"{CONFIG_FOLDER}/{CHECKPOINT_FILE}"
register_all_modules()

# COCO (TO CHECK COCO DATASET VALIDITY)
train_coco = COCO(TRAIN_ANN_PATH.as_posix())
val_coco = COCO(VAL_ANN_PATH.as_posix())
categories = train_coco.loadCats(train_coco.getCatIds())
category_id_to_name = {cat["id"] for cat in categories} 
for category_id in category_id_to_name:
    print(f"Category ID: {category_id}")

# MODEL CONFIGURATION
cfg = Config.fromfile("/mmdetection/configs/mask_rcnn/mask-rcnn_x101-32x8d_fpn_1x_coco.py")
cfg.custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        "mmpretrain.models",
        "metrics.dice",
        "metrics.iou_bbox_metric",
        "metrics.classification_metric",
        "metrics.iou_segm_metric",
    ],
)
cfg.metainfo = {
    "classes": ("Contact", "Crack", "Interconnect", "Corrosion"),
}
cfg.image_size = (640, 640)
cfg.train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="RandomResize",
        scale=cfg.image_size,
        ratio_range=SCALE_RATIO_RANGE,
        keep_ratio=True,
    ),
    dict(
        allow_negative_crop=True,
        crop_size=cfg.image_size,
        crop_type="absolute_range",
        recompute_bbox=True,
        type="RandomCrop",
    ),
    dict(
        type="Rotate",
        level=10,
        prob=0.5,
    ),
    dict(
        type="RandomFlip",
        prob=0.5,
    ),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type="PackDetInputs"),
] 
cfg.val_pipeline = [
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(
        keep_ratio=True,
        scale=cfg.image_size,
        type="Resize",
    ),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
        type="PackDetInputs",
    ),
]
cfg.data_root = DATA_ROOT.as_posix()
cfg.train_dataloader.dataset.ann_file = TRAIN_ANN_PATH.as_posix()
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.data_prefix = dict(img=TRAIN_DATA_FOLDER)
cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline
cfg.train_dataloader.dataset.metainfo = cfg.metainfo
cfg.val_dataloader.dataset.ann_file = VAL_ANN_PATH.as_posix()
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix = dict(img=VAL_DATA_FOLDER)
cfg.val_dataloader.dataset.metainfo = cfg.metainfo
cfg.val_dataloader.dataset.pipeline = cfg.val_pipeline
cfg.train_dataloader.num_workers = NUM_WORKERS
cfg.train_dataloader.batch_size = BATCH_SIZE
cfg.val_dataloader.num_workers = NUM_WORKERS
cfg.val_dataloader.batch_size = BATCH_SIZE
cfg.test_dataloader = cfg.val_dataloader
cfg.val_evaluator = [
    dict(
        ann_file=VAL_ANN_PATH.as_posix(),
        backend_args=None,
        format_only=False,
        metric=[
            "bbox",
            "segm",
        ],
        type="CocoMetric",
    ),
    dict(
        num_classes=len(cfg.metainfo["classes"]),
        type="ClassificationMetric", 
    ),
    dict(
        type="Dice",
    ),
    dict(
        threshold=F1_THRESHOLD,
        type="IoUBBoxMetric",
    ),
    dict(
        threshold=F1_THRESHOLD,
        type="IoUSegmMetric",
    ),
]
cfg.test_evaluator = cfg.val_evaluator
cfg.model.test_cfg.rcnn.score_thr = 0.25 
cfg.model.roi_head.bbox_head.num_classes = len(cfg.metainfo["classes"])
cfg.model.roi_head.mask_head.num_classes = len(cfg.metainfo["classes"])

# Check if checkpoint file exists
if osp.exists(checkpoint_file):
    cfg.load_from = checkpoint_file
cfg.work_dir = (model_path).as_posix()
cfg.train_cfg.val_interval = EVAL_INTERVAL
cfg.default_hooks.checkpoint.interval = CHECKPOINT_INTERVAL 

cfg.workflow = [('train', 1), ('val', 1)]
cfg.optim_wrapper = dict(
    optimizer=dict(type=OPTIM_TYPE, momentum=0.9, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY), type="OptimWrapper"
) 
cfg.optim_wrapper.optimizer.lr = LEARNING_RATE 
cfg.default_hooks.logger.interval = 10
cfg.train_cfg.max_epochs = NUM_EPOCHS
cfg.param_scheduler[1].end = NUM_EPOCHS 
cfg.param_scheduler[1].milestones = [int(NUM_EPOCHS * 0.75), int(NUM_EPOCHS * 0.91)]
cfg.param_scheduler[1].begin = LEARNING_RATE
set_random_seed(0, deterministic=False)
cfg.visualizer.vis_backends.append({"type": "TensorboardVisBackend"})
config = f"{CONFIG_FOLDER}/{MODEL_CONFIG}_modified.py"
with open(config, "w") as f:
    f.write(cfg.pretty_text)

# TRAINING/TESTING COMMAND
if TRAIN_MODE:
    ts = datetime.datetime.now().timestamp()
    timestamp_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    amp = "--amp" if USE_AMP else "" 
    print("Model setup done. Run the following command to start training:")
    print(f"python tools/train.py {config} --work-dir {cfg.work_dir} {amp}") 
else:
    checkpoint_file = INFERENCE_MODEL_PATH
    print(f"Using {checkpoint_file} from {config} to test the model...")
    print("Model setup done. Run the following command to start testing:")
    print(
        f"python tools/test.py {config} {checkpoint_file} --show-dir {(OUTPUT_PATH/'test').as_posix()}"
    )

