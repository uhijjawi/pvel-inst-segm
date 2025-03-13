# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import cv2
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar

from mmdet.models.utils import mask2ndarray
from mmdet.registry import DATASETS, VISUALIZERS
from mmdet.structures.bbox import BaseBoxes


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    # # Create a dummy args object for testing
    # class Args:
    #     config = 'config/mask-rcnn_convnext-v2-b_fpn_lsj-3x-fcmae_coco_modified_20240711_1218.py'
    #     output_dir = 'config/img_previews'
    #     not_show = True
    #     show_interval = 2
    #     cfg_options = None
    # args = Args()
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register all modules in mmdet into the registries
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = dataset.metainfo

    progress_bar = ProgressBar(len(dataset))
    for item in dataset:
        img = item['inputs'].permute(1, 2, 0).numpy()
        img_raw = cv2.cvtColor(cv2.imread(item['data_samples'].img_path), cv2.COLOR_BGR2RGB)

        data_sample = item['data_samples'].numpy()
        gt_instances = data_sample.gt_instances
        img_path = osp.basename(item['data_samples'].img_path)
        img_ext = osp.splitext(img_path)[1]
        
        out_file_raw = osp.join(
            args.output_dir,
            osp.basename(img_path).split('.')[0] + '_0_raw' + img_ext) if args.output_dir is not None else None
        
        out_file_no_ann = osp.join(
            args.output_dir,
            osp.basename(img_path).split('.')[0] + '_1_no_ann' + img_ext) if args.output_dir is not None else None
        
        out_file = osp.join(
            args.output_dir,
            osp.basename(img_path).split('.')[0] + '_2_ann' + img_ext) if args.output_dir is not None else None
        
        img = img[..., [2, 1, 0]]  # bgr to rgb
        gt_bboxes = gt_instances.get('bboxes', None)
        if gt_bboxes is not None and isinstance(gt_bboxes, BaseBoxes):
            gt_instances.bboxes = gt_bboxes.tensor
        gt_masks = gt_instances.get('masks', None)
        if gt_masks is not None:
            masks = mask2ndarray(gt_masks)
            gt_instances.masks = masks.astype(bool)
        data_sample.gt_instances = gt_instances

        visualizer.add_datasample(
            osp.basename(img_path),
            img,
            data_sample,
            draw_pred=False,
            show=not args.not_show,
            wait_time=args.show_interval,
            out_file=out_file)
        
        visualizer.add_datasample(
            osp.basename(img_path),
            img,
            None,
            draw_pred=False,
            show=not args.not_show,
            wait_time=args.show_interval,
            out_file=out_file_no_ann)

        visualizer.add_datasample(
            osp.basename(img_path),
            img_raw,
            None,
            draw_pred=False,
            show=not args.not_show,
            wait_time=args.show_interval,
            out_file=out_file_raw)
        progress_bar.update()


if __name__ == '__main__':
    main()
