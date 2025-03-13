# INSTNACE SEGMENTATION USING MMDETECTION

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
TRAIN_MODE = False
USE_AMP = True
K_FOLDS = None  
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
K_FOLD_TRAIN_FOLDER = "train"
K_FOLD_VAL_FOLDER = "val"
MODEL_CONFIG = "mask-rcnn_convnext-v2-b_fpn_lsj-3x-fcmae_coco"
CHECKPOINT_FILE = (
    "mask-rcnn_convnext-v2-b_fpn_lsj-3x-fcmae_coco_20230113_110947-757ee2dd.pth"
)
NUM_EPOCHS = 100
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.05
NUM_WORKERS = 4
BATCH_SIZE = 6
OPTIM_TYPE = "SGD"
EVAL_INTERVAL = 5
IMAGE_SIZE = (1024, 1024)
IMAGE_RESCALE_RANGE = (0.5, 1.5)
NECK = "FPN" 
CLS_LOSS_FUNCTION = "CrossEntropyLoss"  
MASK_LOSS_FUNCTION = "CrossEntropyLoss" 
BBOX_LOSS_FUNCTION = "L1Loss"  
PYTHON_PATH = "python" 
F1_THRESHOLD = 0.25

# DIRECTORY SETUP
if osp.exists(OUTPUT_PATH):
    print(f"Output path {OUTPUT_PATH} already exists, deleting...")
    os.system(f"rm -rf {OUTPUT_PATH}")
os.makedirs(OUTPUT_PATH, exist_ok=True)
timestamp = time.strftime("%Y%m%d_%H%M", time.localtime())
model_path = MODEL_PATH / timestamp
os.makedirs(model_path, exist_ok=True)
config_file = f"{CONFIG_FOLDER}/{MODEL_CONFIG}.py"
checkpoint_file = f"{CONFIG_FOLDER}/{CHECKPOINT_FILE}"
register_all_modules()

# COCO INIT
if K_FOLDS is None:
    train_coco = COCO(TRAIN_ANN_PATH.as_posix())
    val_coco = COCO(VAL_ANN_PATH.as_posix())
    categories = train_coco.loadCats(train_coco.getCatIds())
    category_id_to_name = {cat["id"] for cat in categories}
    for category_id in category_id_to_name:
        print(f"Category ID: {category_id}")

# MODEL CONFIGURATION
amp = "--amp" if USE_AMP else ""
cfg = Config.fromfile(f"{CONFIG_FOLDER}/{MODEL_CONFIG}.py")
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

cfg.train_dataloader.dataset.metainfo = cfg.metainfo
cfg.val_dataloader.dataset.metainfo = cfg.metainfo
cfg.train_dataloader.num_workers = NUM_WORKERS
cfg.train_dataloader.batch_size = BATCH_SIZE
cfg.val_dataloader.num_workers = NUM_WORKERS
cfg.val_dataloader.batch_size = BATCH_SIZE
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
    dict(type='MultiLabelMetric_c', average='macro'),  # class-wise mean
    dict(type='MultiLabelMetric_c', average='micro'),  # overall mean
]
cfg.model.test_cfg.rcnn.score_thr = 0.15
cfg.model.roi_head.bbox_head.num_classes = len(cfg.metainfo["classes"])
cfg.model.roi_head.mask_head.num_classes = len(cfg.metainfo["classes"])
if osp.exists(checkpoint_file):
    print(f"Loading checkpoint from {checkpoint_file}...")
    cfg.load_from = checkpoint_file
cfg.train_cfg.val_interval = EVAL_INTERVAL
cfg.default_hooks.checkpoint.interval = 5
cfg.image_size = IMAGE_SIZE
cfg.albu_train_transforms = [
    dict(type="CLAHE"),
    dict(
        type="OneOf",
        transforms=[
            dict(
                type="GaussianBlur",
                sigma_limit=0,
                blur_limit=[11, 15],
                p=0.25),
            dict(
                type="RandomBrightnessContrast",
                brightness_limit=[-0.21, 0.21],
                contrast_limit=[-0.25, 0.25],
                p=0.5,
            ),
            dict(
                type="HueSaturationValue",
                hue_shift_limit=5,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=1.0,
            ),
            dict(
                type="RandomGamma",
                gamma_limit=[25, 150],
                p=0.5,
            )
        ],
        p=0.5,
    ),
]
cfg.albu_val_transforms = [
    dict(type="CLAHE"),
]
cfg.train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="RandomResize",
        scale=cfg.image_size,
        ratio_range=IMAGE_RESCALE_RANGE,
        keep_ratio=True,
    ),
    dict(
        allow_negative_crop=True,
        crop_size=cfg.image_size,
        crop_type='absolute_range',
        recompute_bbox=True,
        type='RandomCrop'
    ),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1.0, 1.0)),
    dict(
        type="Rotate",
        level=8,
        prob=0.25,
    ),
    dict(
        type="RandomFlip",
        prob=0.25,
    ),
    dict(type="PackDetInputs"),
]
cfg.optim_wrapper.optimizer.lr = LEARNING_RATE
cfg.optim_wrapper.optimizer.weight_decay = WEIGHT_DECAY
cfg.optim_wrapper = dict(
    optimizer=dict(type=OPTIM_TYPE, momentum=0.9, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
) 
cfg.default_hooks.logger.interval = 10
cfg.max_epochs = NUM_EPOCHS
cfg.train_cfg.max_epochs = NUM_EPOCHS
cfg.param_scheduler[1].end = NUM_EPOCHS
cfg.param_scheduler[1].milestones = [int(NUM_EPOCHS * 0.75), int(NUM_EPOCHS * 0.91)]
cfg.param_scheduler[1].begin = LEARNING_RATE
set_random_seed(42, deterministic=False)
cfg.visualizer.vis_backends.append({"type": "TensorboardVisBackend"})

# Change neck
if NECK == "FPN":
    cfg.model.neck = dict(
        type="FPN",
        in_channels=[
            128,
            256,
            512,
            1024,
        ],
        out_channels=256,
        num_outs=5,
    )
elif NECK == "FPG":
    cfg.image_size = (640, 640)
    cfg.batch_augments = [dict(type="BatchFixedSizePad", size=cfg.image_size)]
    cfg.model.data_preprocessor = dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        pad_size_divisor=64,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        batch_augments=cfg.batch_augments,
        type="DetDataPreprocessor",
    )
    cfg.norm_cfg = dict(type="BN", requires_grad=True)
    cfg.model.neck = dict(
        type="FPG",
        in_channels=[
            128,
            256,
            512,
            1024,
        ],
        out_channels=256,
        inter_channels=256,
        num_outs=5,
        stack_times=9,
        paths=["bu"] * 9,
        same_down_trans=None,
        same_up_trans=dict(
            type="conv",
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=cfg.norm_cfg,
            inplace=False,
            order=("act", "conv", "norm"),
        ),
        across_lateral_trans=dict(
            type="conv",
            kernel_size=1,
            norm_cfg=cfg.norm_cfg,
            inplace=False,
            order=("act", "conv", "norm"),
        ),
        across_down_trans=dict(
            type="interpolation_conv",
            mode="nearest",
            kernel_size=3,
            norm_cfg=cfg.norm_cfg,
            order=("act", "conv", "norm"),
            inplace=False,
        ),
        across_up_trans=None,
        across_skip_trans=dict(
            type="conv",
            kernel_size=1,
            norm_cfg=cfg.norm_cfg,
            inplace=False,
            order=("act", "conv", "norm"),
        ),
        output_trans=dict(
            type="last_conv",
            kernel_size=3,
            order=("act", "conv", "norm"),
            inplace=False,
        ),
        norm_cfg=cfg.norm_cfg,
        skip_inds=[(0, 1, 2, 3), (0, 1, 2), (0, 1), (0,), ()],
    )
elif NECK == "PAFPN":
    cfg.model.neck = dict(
        type="PAFPN",
        in_channels=[
            128,
            256,
            512,
            1024,
        ],
        out_channels=256,
        num_outs=5,
    )
elif NECK == "FPN_CARAFFE":
    cfg.model.data_preprocessor = dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        pad_size_divisor=64,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type="DetDataPreprocessor",
    )
    cfg.model.neck = dict(
        type="FPN_CARAFE",
        in_channels=[
            128,
            256,
            512,
            1024,
        ],
        out_channels=256,
        num_outs=5,
        start_level=0,
        end_level=-1,
        norm_cfg=None,
        act_cfg=None,
        order=("conv", "norm", "act"),
        upsample_cfg=dict(
            type="carafe",
            up_kernel=5,
            up_group=1,
            encoder_kernel=3,
            encoder_dilation=1,
            compressed_channels=64,
        ),
    )
    cfg.model.roi_head.mask_head.upsample_cfg = dict(
        type="carafe",
        scale_factor=2,
        up_kernel=5,
        up_group=1,
        encoder_kernel=3,
        encoder_dilation=1,
        compressed_channels=64,
    )

# Change loss function
if CLS_LOSS_FUNCTION == "CrossEntropyLoss":
    cfg.model.roi_head.bbox_head.loss_cls = dict(
        loss_weight=1.0, type="CrossEntropyLoss", use_sigmoid=False
    )
elif CLS_LOSS_FUNCTION == "SeesawLoss":
    cfg.model.roi_head.bbox_head.loss_cls = dict(
        type="SeesawLoss",
        p=0.8,
        q=2.0,
        num_classes=len(cfg.metainfo["classes"]),
        loss_weight=1.0,
    )
elif CLS_LOSS_FUNCTION == "FocalLoss":
    cfg.model.roi_head.bbox_head.loss_cls = dict(
        type="FocalLoss",
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
    )
if MASK_LOSS_FUNCTION == "CrossEntropyLoss":
    cfg.model.roi_head.mask_head.loss_mask = dict(
        loss_weight=1.0, type="CrossEntropyLoss", use_mask=True
    )
if BBOX_LOSS_FUNCTION == "L1Loss":
    cfg.model.roi_head.bbox_head.loss_bbox = dict(loss_weight=1.0, type="L1Loss")
    cfg.model.rpn_head.loss_bbox = dict(loss_weight=1.0, type="L1Loss")
elif BBOX_LOSS_FUNCTION == "IoULoss":
    cfg.model.roi_head.bbox_head.loss_bbox = dict(type="IoULoss", loss_weight=10.0)
    cfg.model.rpn_head.loss_bbox = dict(type="IoULoss", loss_weight=10.0)

if K_FOLDS is None:
    cfg.data_root = DATA_ROOT.as_posix()
    cfg.train_dataloader.dataset.ann_file = TRAIN_ANN_PATH.as_posix()
    cfg.train_dataloader.dataset.data_root = cfg.data_root
    cfg.train_dataloader.dataset.data_prefix = dict(img=TRAIN_DATA_FOLDER)
    cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline
    cfg.val_dataloader.dataset.ann_file = VAL_ANN_PATH.as_posix()
    cfg.val_dataloader.dataset.data_root = cfg.data_root
    cfg.val_dataloader.dataset.data_prefix = dict(img=VAL_DATA_FOLDER)
    cfg.val_evaluator[0].ann_file = VAL_ANN_PATH.as_posix()
    cfg.test_dataloader = cfg.val_dataloader
    cfg.test_evaluator = cfg.val_evaluator
    cfg.work_dir = (model_path).as_posix()
    timestamp = time.strftime("%Y%m%d_%H%M", time.localtime())
    modified_config_path = f"{CONFIG_FOLDER}/{MODEL_CONFIG}_modified_{timestamp}.py"
    with open(modified_config_path, "w") as f:
        f.write(cfg.pretty_text)
else:
    config_cv_paths = []
    for fold in range(1, K_FOLDS + 1):
        data_root = DATA_ROOT / TRAIN_DATA_FOLDER
        train_ann_fold_path = (
            TRAIN_DATA_PATH / Path(K_FOLD_TRAIN_FOLDER) / f"{fold}.json"
        )
        val_ann_fold_fold = TRAIN_DATA_PATH / Path(K_FOLD_VAL_FOLDER) / f"{fold}.json"
        work_dir_fold_path = MODEL_PATH / timestamp / f"fold-{fold}"
        cfg.data_root = DATA_ROOT.as_posix()
        cfg.train_dataloader.dataset.ann_file = train_ann_fold_path.as_posix()
        cfg.train_dataloader.dataset.data_root = cfg.data_root
        cfg.train_dataloader.dataset.data_prefix = dict(img=TRAIN_DATA_FOLDER)
        cfg.val_dataloader.dataset.ann_file = val_ann_fold_fold.as_posix()
        cfg.val_dataloader.dataset.data_root = cfg.data_root
        cfg.val_dataloader.dataset.data_prefix = dict(img=TRAIN_DATA_FOLDER)
        cfg.val_dataloader.dataset.metainfo = cfg.metainfo
        cfg.val_evaluator.ann_file = val_ann_fold_fold.as_posix()
        cfg.test_evaluator = cfg.val_evaluator
        cfg.work_dir = (work_dir_fold_path).as_posix()
        print(f"Creating config for train fold {fold}/{K_FOLDS}...")
        modified_config_path = f"{CONFIG_FOLDER}/{MODEL_CONFIG}_fold_{fold}.py"
        config_cv_paths.append((modified_config_path, work_dir_fold_path))
        with open(modified_config_path, "w") as f:
            f.write(cfg.pretty_text)
    cv_script_path = ("train_" if TRAIN_MODE else "test_") + MODEL_CONFIG + ".sh"
    if TRAIN_MODE:
        with open(cv_script_path, "w") as f:
            for fold_config in config_cv_paths:
                f.write(
                    f"{PYTHON_PATH} tools/train.py {fold_config[0]} --work-dir {fold_config[1]} {amp}\n"
                )
    else:
        with open(cv_script_path, "w") as f:
            for fold, fold_config in enumerate(config_cv_paths):
                with open(
                    osp.join(fold_config[1].as_posix(), "last_checkpoint"), "r"
                ) as f:
                    checkpoint_file = osp.join(cfg.work_dir, f.read().strip())
                f.write(
                    f"{PYTHON_PATH} tools/test.py {fold_config[0]} {checkpoint_file} --show-dir {fold_config[1]}\n"
                )


# TRAINING/TESTING COMMAND
if TRAIN_MODE:
    if K_FOLDS is None:
        print("Model setup done. Run the following command to start training:")
        print(
            f"{PYTHON_PATH} tools/train.py {modified_config_path} --work-dir {cfg.work_dir} {amp}"
        )
    else:
        print("Model setup done. Run the following command to start training:")
        print(f"sh {cv_script_path}")
else:
    if K_FOLDS is None:
        checkpoint_file = INFERENCE_MODEL_PATH
        print(
            f"Using {checkpoint_file} from {modified_config_path} to test the model..."
        )
        print("Model setup done. Run the following command to start testing:")
        print(
            f"{PYTHON_PATH} tools/test.py {modified_config_path} {checkpoint_file} --show-dir {(OUTPUT_PATH/'test').as_posix()}"
        )
    else:
        print("Model setup done. Run the following command to start testing:")
        print(f"sh {cv_script_path}")
